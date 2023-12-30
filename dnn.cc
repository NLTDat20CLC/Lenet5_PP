#include"dnn.h"

Network createlenet5_CPU()
{
Network dnn;
  Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5, 1, 2, 2);
  Layer* pool1 = new MaxPooling(6,28, 28, 2, 2, 2);
  Layer* conv2 = new Conv(6,14, 14, 16, 5, 5, 1, 2, 2);
  Layer* pool2 = new MaxPooling(16, 10, 10, 2, 2, 2);
  Layer* fc3 = new FullyConnected(pool2->output_dim(), 120);
  Layer* fc4 = new FullyConnected(120, 84);
  Layer* fc5 = new FullyConnected(84, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* relu4 = new ReLU;
  Layer* softmax = new Softmax;
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool1);
  dnn.add_layer(conv2);
  dnn.add_layer(relu2);
  dnn.add_layer(pool2);
  dnn.add_layer(fc3);
  dnn.add_layer(relu3);
  dnn.add_layer(fc4);
  dnn.add_layer(relu4);
  dnn.add_layer(fc5);
  dnn.add_layer(softmax);
  // loss
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);
  return dnn;
}


Network createlenet5_GPU()
{
  Network dnn_gpu;
  Layer* conv1_gpu = new Conv_Custom(1, 28, 28, 6, 5, 5, 1, 2, 2);
  Layer* pool1_gpu = new MaxPooling(6,28, 28, 2, 2, 2);
  Layer* conv2_gpu = new Conv_Custom(6,14, 14, 16, 5, 5, 1, 2, 2);
  Layer* pool2_gpu = new MaxPooling(16, 10, 10, 2, 2, 2);
  Layer* fc3_gpu = new FullyConnected(pool2_gpu->output_dim(), 120);
  Layer* fc4_gpu = new FullyConnected(120, 84);
  Layer* fc5_gpu = new FullyConnected(84, 10);
  Layer* relu1_gpu = new ReLU;
  Layer* relu2_gpu = new ReLU;
  Layer* relu3_gpu = new ReLU;
  Layer* relu4_gpu = new ReLU;
  Layer* softmax_gpu = new Softmax;
  dnn_gpu.add_layer(conv1_gpu);
  dnn_gpu.add_layer(relu1_gpu);
  dnn_gpu.add_layer(pool1_gpu);
  dnn_gpu.add_layer(conv2_gpu);
  dnn_gpu.add_layer(relu2_gpu);
  dnn_gpu.add_layer(pool2_gpu);
  dnn_gpu.add_layer(fc3_gpu);
  dnn_gpu.add_layer(relu3_gpu);
  dnn_gpu.add_layer(fc4_gpu);
  dnn_gpu.add_layer(relu4_gpu);
  dnn_gpu.add_layer(fc5_gpu);
  dnn_gpu.add_layer(softmax_gpu);
  Loss* loss = new CrossEntropy;
  dnn_gpu.add_loss(loss);
  return dnn_gpu;
}