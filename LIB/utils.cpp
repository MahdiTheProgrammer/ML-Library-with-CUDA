#include <iostream>
#include "utils.h"
#include <cuda_runtime.h>
#include <cassert>
#include "mat_op.h"

Tensor::Tensor(const std::vector<int>& shape_) : shape(shape_){
	total_size =1;
        strides.resize(shape.size());

        for(int f1=shape.size()-1; f1>=0;f1--){
        	strides[f1]=total_size;
                total_size*=shape[f1];
        }
        host_data.resize(total_size);
}

Tensor::~Tensor(){
        if (device_data != nullptr) {
            cudaFree(device_data);
        }
}

Tensor::Tensor(const Tensor& other)
    : shape(other.shape),
      strides(other.strides),
      host_data(other.host_data),
      on_gpu(other.on_gpu),
      total_size(other.total_size)
{
    if (other.device_data) {
        cudaMalloc(&device_data, total_size * sizeof(float));
        cudaMemcpy(device_data, other.device_data, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        device_data = nullptr;
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other)
        return *this;

    if (device_data) {
        cudaFree(device_data);
        device_data = nullptr;
    }

    shape = other.shape;
    strides = other.strides;
    host_data = other.host_data;
    on_gpu = other.on_gpu;
    total_size = other.total_size;

    if (other.device_data) {
        cudaMalloc(&device_data, total_size * sizeof(float));
        cudaMemcpy(device_data, other.device_data, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return *this;
}

void Tensor::to_device(){
	if(device_data==nullptr){
		cudaMalloc(&device_data, total_size * sizeof(float));
	}
        cudaMemcpy(device_data, host_data.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
	on_gpu = true;
}

void Tensor::to_host(){
        cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	on_gpu = false;
}

void Tensor::print(){

	if(on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}

	for(int f1=0;f1<shape.size();f1++){
		std::cout<<"[";
	}
	bool b=false;
	for(int f1=0;f1<total_size;f1++){
		for(int f2=0; f2<strides.size()-1;f2++){
			if(f1 % strides[f2]==0 && f1!=0 && f1!=1){
				if(f2<shape.size()-2){
					if(f2<shape.size()-3){
					std::cout<<"]\n";
					}else{
					std::cout<<"]]\n";
					}
					b = true;
				}
				else{
					if(b){
						std::cout<<"\n[[";
						b = false;
					}
					else{
						std::cout<<"]\n[";
					}
				}
			}
		}
		std::cout << host_data[f1]<<", ";
	}

	for(int f1=0;f1<shape.size();f1++){
		std::cout<<"]";
	}
	std::cout<<"\n";
}

int Tensor::flatten_index(const std::vector<int>& indices) const {
	int idx = 0;
	for(int f1=0; f1<indices.size();f1++){
		idx += indices[f1] * strides[f1];
	}
	return idx;
}

float&  Tensor::operator[](const std::vector<int>& indices) {
	if(on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	return host_data[flatten_index(indices)];
}

void Tensor::zeros(){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = 0;
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void Tensor::ones(){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = 1;
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void Tensor::full(const float& value){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = value;
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void Tensor::arrange(const float& start, const float& step){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = start+(step*f1);
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}
//void Tensor::rand(){
//}

//void clone(){

//}
void Tensor::add_padding(int padding, float value){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	int batch = 1;
	for(int f=0; f<shape.size()-2;f++){
		batch*=shape[f];
	}
	int counter = 0;
	int x = shape[shape.size()-2];
	int y = shape[shape.size()-1];

	for(int f=0; f<batch; f++){
		host_data.insert(host_data.begin()+ f*x*y + counter , y+3, value);
		counter+=y+3;

		for(int n=0;n<x-1;n++){
			host_data.insert(host_data.begin()+ (y*(n+1))+ counter+ f*x*y ,2,value);
			counter+=2;
		}

		host_data.insert(host_data.begin() + ((f+1)*x*y) + counter, y+3, value);
		counter+=y+3;
	}
	shape[shape.size()-1] = shape[shape.size()-1] + (padding*2);
	shape[shape.size()-2] = shape[shape.size()-2] + (padding*2);
	total_size = 1;
        for(int f1=shape.size()-1; f1>=0;f1--){
                strides[f1]=total_size;
                total_size*=shape[f1];
        }

	if (on_gpu){
		cudaFree(device_data);
		cudaMalloc(&device_data, total_size * sizeof(float));
		cudaMemcpy(device_data, host_data.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

const std::vector<int>& Tensor::get_shape() const {
	return shape;
}

const std::vector<int>& Tensor::get_strides() const{
	return strides;
}

float* Tensor::device_address() const{
	cudaMemcpy(device_data, host_data.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
	return device_data;
}

bool Tensor::is_on_gpu() const{
	return on_gpu;
}
int Tensor::get_total_size() const{
	return total_size;
}
void Tensor::from_list(float* data){
	for (int i = 0; i < total_size; ++i) {
		host_data[i] = data[i];
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(), total_size * sizeof(float),cudaMemcpyHostToDevice);
	}
//	device_data = data;
//	host_data = *data;
}
const std::vector<float>& Tensor::get_data(){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}

	return host_data;
}
void Tensor::print_2D(){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for (float num : host_data){
		std::cout<<num<<", ";
	}
	std::cout<<"\n";
}
void Tensor::reshape(std::vector<int> new_shape){
	shape=new_shape;
        total_size = 1;
        for(int f1=shape.size()-1; f1>=0;f1--){
                strides[f1]=total_size;
                total_size*=shape[f1];
        }

}

void Tensor::requires_grad(bool requires){
	grad = requires;
}
