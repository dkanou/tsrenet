#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/Berhu_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

// Berhu Loss layer. Refer "Deeper Depth Prediction with Fully Convolutional Residual Networks" for the loss function. 
// 

namespace caffe {

template <typename Dtype>
__global__ void BerhuLossForwardGPU(const int nthreads,
          Dtype* data_diff, 
          Dtype* loss_data, const Dtype C) {
  CUDA_KERNEL_LOOP(index, nthreads) {
		if (fabs(data_diff[index]) <= C){
			//L1 norm
			loss_data[index] = fabs(data_diff[index]);
		
			// gradients of L1 norm is set to C. Later on, all the gradients are scaled down by C. 
			if (data_diff[index] > 0) data_diff[index]= C;
			else if (data_diff[index] < 0) data_diff[index]= (-C);
						} 
		else{
			//L2 norm			
			loss_data[index] = (((data_diff[index]*data_diff[index])/ C) + C) / Dtype(2);
		}
	}
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // count = NxCxHxW
  int count = bottom[0]->count();

  // diff_.gpu_data contains the gradient. Note: For L2 norm, the difference is the gradient.
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  // Calculate the maximum absolute difference and threshold
  Dtype max_diff = 0;
  max_diff = caffe_gpu_amax(count, diff_.gpu_data(), 1);
  Dtype C = max_diff * Dtype(0.2);

  //Gradient to be altered according to L1 or L2 norm
  Dtype* data_diff = diff_.mutable_gpu_data();

  // Create loss
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  // CUDA macros cannot be accessed without initializing blocks and threads.
  BerhuLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, data_diff, 
      loss_data, C);

  Dtype loss;

  //Sum of all the loss
  caffe_gpu_asum(count, loss_data, &loss);
  loss = loss / bottom[0]->num();

  // Scaling
  //caffe_gpu_scal(count, Dtype(1.0) / C, diff_.mutable_gpu_data());
  top[0]->mutable_cpu_data()[0] = loss;
}

// Backprop similar to Euclidean norm.
template <typename Dtype>
void BerhuLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BerhuLossLayer);

}  // namespace caffe
