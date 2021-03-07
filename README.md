# RFN-Nest: An end-to-end residual fusion network for infrared and visible images

[Information Fusion](https://doi.org/10.1016/j.inffus.2021.02.023)(IF:13.669)


## Platform
Python 3.7  
Pytorch >=0.4.1  

The testing datasets are included in "images".

The results iamges are included in "outputs".

## Fusion framework

[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) (T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) is utilized to train our auto-encoder network.

[KAIST](https://sites.google.com/view/multispectral/home) (S. Hwang, J. Park, N. Kim, Y. Choi, I. So Kweon, Multispectral pedestrian detection: Benchmark dataset and baseline, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1037–1045.) is utilized to train the RFN modules.

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/framework.png" width="600">


### Decoder architecture

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/decoder.png" width="600">



### Training RFN modules

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/training-rfn.png" width="600">


### Fusion results

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/results-umbrella.png" width="600">


### RFN for RGBT tracking - framework

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/tracking-framework.png" width="600">


### RFN for RGBT tracking - results

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/results-tracking.png" width="600">
