//Contains all CUDA kernels (e.g. matrixMultiply, ReLU)
#include <iostream>
#include <cuda_runtime.h>
#include "mat_op.h"
#include "utils.h"

__global__ void matrixmultiplication(float *t_A, float *t_B, float *c, int batch_size, int m, int n, int k){
	int batch_id = blockIdx.z;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float e=0.0f;
	if (row < m && col < k){
		for(int f1=0; f1<n;f1++){
			e += t_A[ (batch_id * m  * n) + n*row +f1 ] * t_B[ ( batch_id * n  * k ) + (k*f1) + col];
		}
		c[(batch_id * m * k) + row * k + col] = e;
	}
}

__global__ void matrixmultiplication2(float *t_A, float *t_B, float *c, int batch_size, int m, int n, int k, int f){
	int batch_id = blockIdx.z;
	int col = blockIdx.x;
	int row = blockIdx.y;
	float e=0.0f;
	if (row < m && col < k){
		for(int f1=0; f1<f;f1++){
			e += t_A[ (batch_id * m  * n) + n*row +f1 + threadIdx.x*32 ] * t_B[ ( batch_id * n  * k ) + (k*f1) + col + (k*32*threadIdx.x)];
		}
		__shared__ float sdata;
		sdata += e;
		__syncthreads();
		
		c[(batch_id * m * k) + row * k + col] = e;
	}
}

__global__ void matrixmultiplication3(float *t_A, float *t_B, float *c, int batch_size, int m, int n, int k, int f){
	int batch_id = blockIdx.z;
	int col = blockIdx.x;
	int row = blockIdx.y;
	r = batch_id%2;
	float e=0.0f;
	if (row < m && col < k){
		for(int f1=0; f1<f/2;f1++){
			e += t_A[ (batch_id * m  * n) + n*row +f1*(r+1) + threadIdx.x*32 ] * t_B[ ( batch_id * n  * k ) + (k*f1) + col + (k*32*threadIdx.x)];
		}
		__shared__ float sdata;
		sdata += e;
		__syncthreads();
		
		c[(batch_id * m * k) + row * k + col] = e;
	}
}


__global__ void matadd(float *t_A, float *t_b,float *output, int num_threads){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_threads) {
		output[idx] = t_A[idx] +  t_b[idx];

	}
}

Tensor Tensor::matmul(const Tensor& t_A, const Tensor& t_B){
        std::vector<int> shape_A = t_A.get_shape();
        std::vector<int> shape_B = t_B.get_shape();
  
		std::vector<int> shape_output;
		int d=1;

        for(int f1=0; f1<shape_A.size()-2;f1++){
                d*=shape_A[f1];
		shape_output.push_back(shape_A[f1]);
        }

		shape_output.push_back(shape_A[shape_A.size()-2]);
		shape_output.push_back(shape_B[shape_B.size()-1]);

        float* add_A = t_A.device_address();
        float* add_B = t_B.device_address();
        int total_size_C = d * shape_B[shape_B.size()-1] * shape_A[shape_A.size()-2];
        float *add_C;
        float *h_C = new float[total_size_C];
        cudaMalloc((void**)&add_C,total_size_C * sizeof(float));

        dim3 blockDim(32,32);
        dim3 gridDim((shape_B[shape_B.size()-1]+31)/32,(shape_A[shape_A.size()-2]+31)/32 , d);

        matrixmultiplication<<<gridDim, blockDim>>>(add_A,add_B,add_C,d,shape_A[shape_A.size()-2],shape_B[shape_B.size()-2],shape_B[shape_B.size()-1]);

        cudaDeviceSynchronize();
        cudaMemcpy(h_C, add_C, total_size_C * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(add_C);

		Tensor output(shape_output);
		output.from_list(h_C);

        return output;
}

Tensor Tensor::matmul2(const Tensor& t_A, const Tensor& t_B){
        std::vector<int> shape_A = t_A.get_shape();
        std::vector<int> shape_B = t_B.get_shape();
  
		std::vector<int> shape_output;
		int d=1;

        for(int f1=0; f1<shape_A.size()-2;f1++){
                d*=shape_A[f1];
		shape_output.push_back(shape_A[f1]);
        }

		shape_output.push_back(shape_A[shape_A.size()-2]);
		shape_output.push_back(shape_B[shape_B.size()-1]);

        float* add_A = t_A.device_address();
        float* add_B = t_B.device_address();
        int total_size_C = d * shape_B[shape_B.size()-1] * shape_A[shape_A.size()-2];
        float *add_C;
        float *h_C = new float[total_size_C];
        cudaMalloc((void**)&add_C,total_size_C * sizeof(float));

        dim3 blockDim((shape_A[shape_A.size()-1]+31)/32);
        dim3 gridDim((shape_B[shape_B.size()-1]),(shape_A[shape_A.size()-2]), d);

		int f;
		int n = shape_B[shape_B.size()-2];
		if (n<32){
			f = n;
		}else{
			f = 32;
		}
        matrixmultiplication2<<<gridDim, blockDim>>>(add_A,add_B,add_C,d,shape_A[shape_A.size()-2],shape_B[shape_B.size()-2],shape_B[shape_B.size()-1],f);

        cudaDeviceSynchronize();
        cudaMemcpy(h_C, add_C, total_size_C * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(add_C);

		Tensor output(shape_output);
		output.from_list(h_C);

        return output;
}


Tensor Tensor::MatrixAddition(const Tensor& t_A, const Tensor& t_B){

	float* add_A = t_A.device_address();
	float* add_B = t_B.device_address();

	std::vector<int> shape_A = t_A.get_shape();
	std::vector<int> shape_B = t_B.get_shape();

	size_t num_threads = 1;
	for(int i=0; i<shape_A.size();i++){
		num_threads*=shape_A[i];
	}

	float *add_C;
	float *h_C = new float[num_threads];

	cudaMalloc((void**)&add_C,num_threads * sizeof(float));

	int threads_per_block = 32*32;

	dim3 blockDim(threads_per_block);
	dim3 gridDim((num_threads+threads_per_block-1)/threads_per_block);

	matadd<<<gridDim, blockDim>>>(add_A, add_B, add_C, num_threads);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
   		 printf("Kernel launch error: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();

	cudaMemcpy(h_C, add_C, num_threads * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(add_C);

	Tensor output(shape_A);
	output.from_list(h_C);

	return output;
}
