# Targeted DeepFool

Targeted DeepFool is a a parametrized, simple and effective algorithm for fooling deep neural networks to specific classes.

## Setup

We encourage creating a virtual enviornment to run our code.

### Requirements

torch==2.0.0+cu117  

torchaudio==2.0.1+cu117  

torchmetrics==0.11.4  

torchvision==0.15.1+cu117  

matplotlib==3.7.1

### Dataset

We used the **Imagenet2012 Validation Dataset**[[1]](https://arxiv.org/abs/1409.0575).  

After downloading, unzip the dataset in the `./data` directory

### Issues you might run into:

You might run into a problem when importing the libraries.

The following error might show up: **"cannot import name 'zero_gradients' from 'torch.autograd.gradcheck"**

There will be a link at the end of the error, which should lead you to **gradcheck py** file

Copy and paste the following code the **gradcheck.py** file

```
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)
```

Save the gradcheck.py file and you should be good to go.

## deepfool.py

This function implements the algorithm proposed in [[2]](http://arxiv.org/pdf/1511.04599) using PyTorch to find adversarial perturbations.

**Note**: The final softmax (loss) layer should be removed in order to prevent numerical instabilities.

The parameters of the function are:

- `image`: Image of size `HxWx3d`
- `net`: neural network (input: images, output: values of activation **BEFORE** softmax).
- `num_classes`: limits the number of classes to test against, by default = 10.
- `max_iter`: max number of iterations, by default = 50.

## deepfool_targeted.py

There are two functions here one for testing purpose and another for reproducing experiments on our paper.

The function here is a modified verison of the original algorithm.

**Note**: The final softmax (loss) layer should be removed in order to prevent numerical instabilities.

The parameters of the function are:

- `image`: Image of size `HxWx3d`
- `net`: neural network (input: images, output: values of activation **BEFORE** softmax).
- `target_class`: target class that the image should be misclassified as.
- `overshoot`: used as a termination criterion to prevent vanishing updates (default = 0.02).
- `min_confidence`: used to set minimum amount of confidence.
- `max_iter`: max number of iterations, by default = 100.

## test_deepfool_targeted.ipynb

This file contains demo code that computes targeted adversarial perturbation for a test image from Imagenet dataset.
It includes code to test on the models we covered in our paper.

## experiment_imagenet_val.ipynb

This file contains the code to reproduce our experiments.

## Reference

[1] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (\* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.

[2] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
_DeepFool: a simple and accurate method to fool deep neural networks_. In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.
