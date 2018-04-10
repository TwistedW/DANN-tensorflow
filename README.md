# Domain-Adversarial Training of Neural Networks in Tensorflow

Requires TensorFlow>=1.0 and tested with Python 3.5

## MNIST Experiments

This experiment is tested under the mnist dataset

### Build MNIST-M dataset

First you should download data from http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

and run
```
python create_mnistm.py
```

This may take a couple minutes and should result in a `mnistm.pkl` file containing generated images.


## Training

```
python main.py
```

## statement
This experimental idea from ["Domain-Adversarial Training of Neural Networks"](https://arxiv.org/abs/1505.07818) and this is test version.

 
