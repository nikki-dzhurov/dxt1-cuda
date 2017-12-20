#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "dxt1.h"
#include "types.h"
#include "colors.h"

cudaError_t compressWithCuda(BYTE*, BYTE*, int, int);

__global__ void compressDXT1(BYTE *dev_image, BYTE *dev_compressedImage, long imageSize, int maxDXT1BlocksPerRow, int width)
{
	long currBlockIdx = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	long blockRow = currBlockIdx / maxDXT1BlocksPerRow;
	long blockCol = currBlockIdx % maxDXT1BlocksPerRow;

	long outBuffOffset = currBlockIdx*8;
	long imageOffset = blockRow* width* 16 + blockCol *16;

	if (imageOffset  > imageSize - 64) {
		return;
	}

	BYTE* blockStart = dev_image + imageOffset;
	BYTE* outBuff = dev_compressedImage + outBuffOffset;
	BYTE block[64];
	BYTE minColor[3];
	BYTE maxColor[3];
	DWORD colorIndices;

	extractBlock(blockStart, width, block);
	getMinMaxColors(block, minColor, maxColor);
	getColorIndices(block, minColor, maxColor, &colorIndices);

	writeWord(outBuff, dev_rgb888_to_rgb565(maxColor));
	writeWord(outBuff, dev_rgb888_to_rgb565(minColor));
	writeDoubleWord(outBuff, colorIndices);
}

int main()
{
	char inPath[] = "./images/pretty.png";
	char outPath[] = "./output/out.png";
	int width, height, bytesPerPixel;
	long pixelCount;
	BYTE *imageData;
	cudaError_t cudaStatus;

	BYTE *oldImageData = stbi_load(inPath, &width, &height, &bytesPerPixel, 0);
	if (oldImageData == NULL) {
		printf("Invalid image!");
		return  1;
	}
	if (width % 4 != 0 || height % 4 != 0) {
		printf("Invalid dimensions");
		return  1;
	}
	if (width > 32768 || height > 32768) {
		printf("Max image dimensions: 32768x32768");
		return  1;
	}

	pixelCount = width*height;

	// Check if image is RGB and convert to RGBA
	imageData = (BYTE *)malloc(pixelCount * 4);
	if (bytesPerPixel == 3) {
		bytesPerPixel = 4;
		rgb_to_rgba_image(oldImageData, imageData, pixelCount);
	} else if (bytesPerPixel == 4) {
		memcpy(imageData, oldImageData, pixelCount*bytesPerPixel);
	} else {
		printf("Invalid pixel size! %d %d %d", bytesPerPixel, width, height);
		return 1;
	}

	BYTE *compressedImage = (BYTE*)malloc(pixelCount * 4 / 8);
	BYTE* decompressedImage = (BYTE*)malloc(pixelCount * 4);

	clock_t compressStart = clock();
	// Compress image DXT1(8:1 compression ratio)
	cudaStatus = compressWithCuda(imageData, compressedImage, width, height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Compress failed!");
		return 1;
	}
	clock_t compressEnd = clock();


	// Decompress image DXT1
	DecompressImageDXT1(width, height, compressedImage, decompressedImage);

	clock_t decompressEnd = clock();

	printf("compress: %.3fsec\n", double(compressEnd - compressStart) / CLOCKS_PER_SEC);
	printf("decompress: %.3fsec\n", double(decompressEnd - compressEnd) / CLOCKS_PER_SEC);
	printf("total: %.3fsec\n", double(decompressEnd - compressStart) / CLOCKS_PER_SEC);

	//if (stbi_write_jpg(outPath, width, height, bytesPerPixel, imageData, 100)) {
	if (stbi_write_png(outPath, width, height, bytesPerPixel, decompressedImage, width*bytesPerPixel)) {
		printf("WRITE SUCCESS!");
	} else {
		printf("WRITE ERROR!");
	}

	stbi_image_free(imageData);
	stbi_image_free(oldImageData);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t compressWithCuda(BYTE *image, BYTE *compressedImage, int width, int height)
{
	time_t totalStart = clock();

	BYTE *dev_image = 0;
	BYTE *dev_compressedImage = 0;
	long imageSize = width*height*4;
	int maxDXT1BlocksPerRow = width / 4;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_image, imageSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_compressedImage, imageSize / 8);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_image, image, imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	int threadsCount = 256;
	dim3 blocks(1,1,1);

	long totalBlocks = ceil(height*width / 16 / float(threadsCount));
	if (totalBlocks <= 1024) {
		blocks.x = totalBlocks;
	} else {
		blocks.x = 1024;
		blocks.y = ceil(totalBlocks / 1024.f);
	}

	time_t compressStart = clock();

    // Launch a kernel on the GPU with one thread for each element.
    compressDXT1<<<blocks, threadsCount>>>(dev_image, dev_compressedImage, imageSize, maxDXT1BlocksPerRow, width);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	time_t compressEnd = clock();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(compressedImage, dev_compressedImage, width*height*4 / 8, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	time_t totalEnd = clock();

	printf("Compress time: %.3lfs\n", double(compressEnd - compressStart) / CLOCKS_PER_SEC);
	printf("Total compress time(+memory allocation): %.3lfs\n", double(totalEnd - totalStart) / CLOCKS_PER_SEC);

Error:
    cudaFree(dev_compressedImage);
    cudaFree(dev_image);
    
    return cudaStatus;
}