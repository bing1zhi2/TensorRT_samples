# RESNET 18 网络定义方式转换trt模型

对此代码的说明参见博客：
[resnet api](https://blog.csdn.net/bing1zhi2/article/details/108801025)

## 环境  

trt 7.0-7.1应该都行  （如果trt的api未变的情况下应该都行）

## develop docker env

 you can use nvidia official docker image : [nvidia NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt/tags), eg:

 ```shell
    docker pull nvcr.io/nvidia/tensorrt:20.03-py3
 ```