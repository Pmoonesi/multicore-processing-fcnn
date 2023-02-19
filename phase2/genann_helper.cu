// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "genann.h"
#include <math.h>

// CUDA runtime
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

cudaError_t cudaStatus;

__global__ void
matrixMultiply(double *d_output, double *d_weights, int current_layer, int next_layer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int length = gridDim.x * blockDim.x;
	
	while (i < next_layer) {
		double *start_w = d_weights + (current_layer + 1) * i;
		double sum = *start_w++ * -1.0;
		for (int j = 0; j < current_layer; j++)
			sum += *start_w++ * *(d_output + j);
		*(d_output + current_layer + i) = 1 / (1.0 + exp(sum));
		i += length;
	}
}

__global__ void setupKernel(curandState *state, long long seed)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, i, 0, &state[i]);
}

__global__ void
matrixAddRandom(double* d_matrix, int length, curandState *states)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	double my_rand;

	int i = id;
	while (i < length) {
		my_rand = curand_uniform_double(&states[id]) - 0.5;
		*(d_matrix + i) += my_rand;
		i += threads;
	}
}

extern "C" void cudaFeedForward(genann *ann) {
	double *d_output, *d_weights;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&d_weights, ann->total_weights * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_output, ann->total_neurons * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(d_weights, ann->weight, ann->total_weights * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_output, ann->output, ann->total_neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	if (!ann->hidden_layers) {
		matrixMultiply << <1, 512 >> > (d_output, d_weights, ann->inputs, ann->outputs);
	}
	else {
		double* d_o = d_output;
		double* d_w = d_weights;
		int i;

		/* Figure input layer */
		matrixMultiply << <1, 512 >> > (d_o, d_w, ann->inputs, ann->hidden);
		d_o += ann->inputs;
		d_w += (ann->inputs + 1) * ann->hidden;

		/* Figure hidden layers, if any. */
		for (i = 1; i < ann->hidden_layers; i++) {
			matrixMultiply << <1, 512 >> > (d_o, d_w, ann->hidden, ann->hidden);
			d_o += ann->hidden;
			d_w += (ann->hidden + 1) * ann->hidden;
		}

		/* Figure output layer. */
		matrixMultiply << <1, 512 >> > (d_o, d_w, ann->hidden, ann->outputs);
		d_o += ann->hidden;
		d_w += (ann->hidden + 1) * ann->outputs;

		// making sure we've been through all weights an neurons
		d_o += ann->outputs;
		assert(d_w - d_weights == ann->total_weights);
		assert(d_o - d_output == ann->total_neurons);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
			cudaStatus);
	}

	cudaStatus = cudaMemcpy(ann->output, d_output, ann->total_neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaFree(d_output);
	cudaFree(d_weights);
}


extern "C" void cudaRandomWeights(genann *ann, long long seed) {
	double *d_weights;
	curandState *dev_random;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&d_weights, ann->total_weights * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_random, 512 * sizeof(curandState));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(d_weights, ann->weight, ann->total_weights * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	setupKernel << <1, 512 >> > (dev_random, seed);
	matrixAddRandom << <1, 512 >> > (d_weights, ann->total_weights, dev_random);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
			cudaStatus);
	}

	cudaStatus = cudaMemcpy(ann->weight, d_weights, ann->total_weights * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaFree(d_weights);
	cudaFree(dev_random);
}