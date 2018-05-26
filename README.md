# Tensorflow Neural Network Quantization
This is a school project for the Advanced Computer Architectures class @PoliMi, 2017-2018.

Be careful, the repository contains also the datasets and all the model files.

You can see the main results of MNIST models evaluation looking at these notebooks:
[Model (a)](https://github.com/emanuele-falzone/Tensorflow-Neural-Network-Quantization/blob/master/MNIST/performance-small.ipynb), 
[Model (b)](https://github.com/emanuele-falzone/Tensorflow-Neural-Network-Quantization/blob/master/MNIST/performance-big.ipynb)

The evaluation of CIFAR-10 models is also available:
[Model (c)](https://github.com/emanuele-falzone/Tensorflow-Neural-Network-Quantization/blob/master/CIFAR10/performance-small.ipynb), 
[Model (d)](https://github.com/emanuele-falzone/Tensorflow-Neural-Network-Quantization/blob/master/CIFAR10/performance-big.ipynb)

___

Tensorflow is an open source framework for machine learning applications such as neural networks. We use official python API. The project is developed using:
* Ubuntu 16.04 LTS
* Python 2.7
* Tensorflow 1.8 version with CPU only support

We used a desktop pc assempled with:
* [Inter Core i7 2600K](https://ark.intel.com/it/products/52214/Intel-Core-i7-2600K-Processor-8M-Cache-up-to-3_80-GHz)
* [8 GB DDR3 1333 Mhz RAM](https://www.corsair.com/it/it/Categories/Products/Memory/High-Performance-Memory/Vengeance%C2%AE-Low-Profile-%E2%80%94-8GB-Dual-Channel-DDR3-Memory-Kit/p/CML8GX3M2A1600C9)
* [Nvidia GTX 960 4GB](https://www.evga.com/products/specs/gpu.aspx?pn=dc087073-987f-477e-8258-800938653730)

To install Tensorflow with CPU only support and jupyter notebook run:

```sh
pip install tensorflow
pip install jupyter
```

Then navigate to your code directory and start jupyter
```sh
cd path/to/repo
jupyter notebook
```

___
As mentioned in the project description we are not interested in the training phase. However, we had to train some models.
In order to speed up the training phase we used a [Docker image](https://hub.docker.com/r/tensorflow/tensorflow/) with Nvidia GPU support, officially provided by Tensorflow.

To setup docker and train your model on GPU:
* Install proprietary [_Nvidia driver_](http://www.nvidia.it/Download/index.aspx)
* Install [_docker_](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
* Install [_nvidia-docker_](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

Then pull the latest GPU docker image:
```sh
sudo nvidia-docker pull tensorflow/tensorflow:latest-gpu
```

Create the container with port and data binding
```sh
sudo nvidia-docker run -it  --name tensorflow_docker -d -p 8888:8888 -p 6006:6006 -v ~/tensorflow:/notebooks tensorflow/tensorflow:latest-gpu
```

To start the container:
```sh
sudo nvidia-docker container start tensorflow_docker
```

To stop the container:
```sh
sudo nvidia-docker container stop tensorflow_docker
```

To access Tensorflow docker container bash:
```sh
sudo nvidia-docker exec -it tensorflow_docker /bin/bash
```

To start Tensorboard (in container bash):
```sh
tensorboard --logdir /notebooks/logdir
```

To access jupyter notebook navigate to http://localhost:8888/tree

To access Tensorboard navigate to http://localhost:6006

___
### Useful links:
[Tensorflow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets)

[Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/)
