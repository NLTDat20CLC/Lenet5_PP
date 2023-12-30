/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include "dnn.h"

int main() {
  // data
  MNIST dataset("../data/mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn = createlenet5_CPU();
  // train & test
  SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;
  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                    std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                    std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1) {
        std::cout << ith_batch << "-th grad: " << std::endl;
        dnn.check_gradient(x_batch, target_batch, 10);
      }
      dnn.forward(x_batch);
      dnn.backward(x_batch, target_batch);
      // display
     // if (ith_batch % 50 == 0) {
     //   std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss()
     //   << std::endl;
    //  }
      // optimize
      dnn.update(opt);
    }
    // test
   // dnn.forward(dataset.test_data);
  //  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    //std::cout << std::endl;
    //std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
   // std::cout << std::endl;
  }
    GpuTimer timer;
    timer.Start();
    dnn.forward(dataset.test_data);
    timer.Stop();
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    float ts = timer.Elapsed();
    std::cout << "CPU:" << std::endl;
    std::cout << "Test acc: " << acc << std::endl;
    std::cout << "Time: " << ts << " ms" << std::endl;
    dnn.save_parameters("/content/Lenet5_PP/testinput.bin");
    Network dnn_gpu = createlenet5_GPU();
    dnn_gpu.load_parameters("/content/Lenet5_PP/testinput.bin");
    timer.Start();
    dnn_gpu.forward(dataset.test_data);
    timer.Stop();
    acc = compute_accuracy(dnn_gpu.output(), dataset.test_labels);
    ts = timer.Elapsed();
    std::cout << "GPU:" <<std::endl;
    std::cout << "Test acc: " << acc << std::endl;
    std::cout << "Time: " << ts << " ms" << std::endl;

    //dnngpuoptimize
    Network dnn_gpu_op = createlenet5_GPU_OP();
    dnn_gpu_op.load_parameters("/content/Lenet5_PP/testinput.bin");
    timer.Start();
    dnn_gpu_op.forward(dataset.test_data);
    timer.Stop();
    acc = compute_accuracy(dnn_gpu_op.output(), dataset.test_labels);
    ts = timer.Elapsed();
    std::cout << "GPU Optimize:" <<std::endl;
    std::cout << "Test acc: " << acc << std::endl;
    std::cout << "Time: " << ts << " ms" << std::endl;
  return 0;
}

