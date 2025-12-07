//Starts the program (creates and trains the model)
#include <iostream>
#include "utils.cpp"
#include "layer.h"
#include "model.h"


int  main(){
//	std::vector<int> shape_A = {2,2,6,6};
	std::vector<int> shape_B = {2,2,3};
	std::vector<int> shape_C = {2,3,2};
	// std::vector<int> shape_B = {2,2,2};
	// std::vector<int> shape_C = {2,2,2};
//	Tensor t_A(shape_A);
	Tensor t_B(shape_B);
	Tensor t_C(shape_C);
//	std::vector<float> i_a = {1.0f, 2.0f, 3.0f, 4.0f};;
	t_C.to_device();
	t_B.to_device();
	t_B.arrange(0.0,10.0);
	t_C.arrange(0.0,1.0);
//	t_B.arrange(0,2);
//	t_A.from_list(i_a.data());
//	t_B.from_list(i_b.data());
	Tensor out = Tensor::matmul(t_B,t_C);
//	out.print();
//	std::cout<<"\n";
//	t_B.add_padding(1,0);
	t_B.print();
	std::cout<<"DONE"<<"\n\n";
	t_C.print();	
	std::cout<<"DONE"<<"\n\n";
	out.print();
	std::cout<<"DONE"<<"\n\n";
//    	std::vector<int> shape = out.get_shape();
//	std::cout<<"\n";
//   	 for (int num : shape) {
//        	std::cout << num << " ";
//   	 }
//	Model model;
//	model.add(new softmax());
//	Tensor output = model.forward(t_B);
//	output.print();

//	t_A.add_padding(1,0.0f);
//	t_A.print();
//	std::cout<<"\n";
//	const std::vector<int> shape = output.get_shape();
//	for (float val : shape){
//		std::cout<<val<< " ";
//	}
//	std::cout<<"\n";
//	t_B.print();
//	std::cout<<"\n";
//	t_B.reshape({4,4,4});
//s	t_B.print();
//	std::cout<<"\n";
//	float* i_c = Tensor::matmul(t_A, t_B);
//	t_C.from_list(i_c);
//	t_C.print();
//	float *d_A = t_A.device_address();
//	bool b = t_A.is_on_gpu();
//	std::cout<<"Is on GPU." << b << std::endl;
//	std::cout<<"Address on GPU" <<static_cast<void*>(d_A) << std::endl;
	return 0;
}
