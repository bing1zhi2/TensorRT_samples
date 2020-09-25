'''
Author: 
Date: 2020-09-04 16:03:33
LastEditTime: 2020-09-24 10:15:56
LastEditors: 
Description: In User Settings Edit
FilePath: /onnx_graphsurgeon/4_resnet18_trt_module.py
'''
# import model
import common
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import torch
import sys
import os
# sys.path.insert(1, os.path.join(sys.path[0], ".."))

# You can set the logger severity higher to suppress messages (or lower to display more messages).
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class ModelData(object):
    INPUT_NAME = "input"
    # INPUT_SHAPE = (1, 3, 640, 640)
    INPUT_SHAPE = (1, 3, 320, 320)
    # INPUT_SHAPE = (1, 64, -1, -1)
    OUTPUT_NAME = "output"
    # OUTPUT_SIZE = 10
    DTYPE = trt.float32

def conv2d(input_tensor, out_channel, network, conv0_w, kernel_size, stride=1,padding=None):
    conv0 = network.add_convolution(
        input=input_tensor, num_output_maps=out_channel, kernel_shape= (kernel_size,kernel_size), kernel=conv0_w)
    conv0.stride = (stride, stride)
    if padding is not None:
        conv0.padding = (padding, padding)
    return conv0
    

def conv3x3(input_tensor, out_channel, network, conv0_w, stride=1):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)
    # conv0_w = weights['conv1.weight'].cpu().numpy()
    conv0 = network.add_convolution(
        input=input_tensor, num_output_maps=out_channel, kernel_shape=(3, 3), kernel=conv0_w)
    conv0.stride = (stride, stride)
    conv0.padding = (1, 1)

    return conv0


def batchnorm(input_tensor, network, bn_gamma2, bn_bias2, bn_mean2, bn_var2):
    # bn_gamma2 = weights['cnn.batchnorm2.weight'].cpu().numpy()        # bn gamma
    # bn_bias2  = weights['cnn.batchnorm2.bias'].cpu().numpy()          # bn beta
    # bn_mean2  = weights['cnn.batchnorm2.running_mean'].cpu().numpy()  # bn mean
    # bn_var2   = weights['cnn.batchnorm2.running_var'].cpu().numpy()   # bn var sqrt
    eps = 1e-05
    bn_var2 = np.sqrt(bn_var2 + eps)
    bn_scale = bn_gamma2 / bn_var2
    bn_shift = - bn_mean2 / bn_var2 * bn_gamma2 + bn_bias2
    batchnorm2 = network.add_scale(
        input=input_tensor, mode=trt.ScaleMode.CHANNEL, shift=bn_shift, scale=bn_scale)

    return batchnorm2


def relu(input_tensor, network):
    relu1 = network.add_activation(
        input=input_tensor, type=trt.ActivationType.RELU)
    return relu1


def tensor_add(tensor1, tensor2, network):
    return network.add_elementwise(tensor1, tensor2, trt.ElementWiseOperation.SUM)


def basic_block_network(input_tensor, network, weights, out_channel, stride=1, layerweight_id=1,layerweight_sub_id=0,downsample=None):

    residual = input_tensor

    weight_key_prefix_first = "layer" + str(layerweight_id) + "."+str(layerweight_sub_id)+"."  # layerweight_id = 1-4 layerweight_sub_id = 0 or 1

    print("---> weight_key_prefix_first:", weight_key_prefix_first)
    print("input tensor shape ", input_tensor.shape)

    conv1_w = weights[weight_key_prefix_first + 'conv1.weight'].cpu().numpy()
    conv1 = conv3x3(input_tensor, out_channel, network, conv1_w, stride)
    conv1.name = weight_key_prefix_first + 'conv1'

    print("-------------", conv1.name)

    print(weight_key_prefix_first+"conv1 .shape:", conv1.get_output(0).shape)

    bn_gamma1 = weights[weight_key_prefix_first + 'bn1.weight'].cpu().numpy()        # bn gamma
    bn_bias1 = weights[weight_key_prefix_first + 'bn1.bias'].cpu().numpy()          # bn beta
    bn_mean1 = weights[weight_key_prefix_first + 'bn1.running_mean'].cpu().numpy()  # bn mean
    bn_var1 = weights[weight_key_prefix_first + 'bn1.running_var'].cpu().numpy()   # bn var sqrt
    bn1 = batchnorm(conv1.get_output(0), network, bn_gamma1, bn_bias1, bn_mean1, bn_var1)
    bn1.name = weight_key_prefix_first + 'bn1'

    print("-------------", bn1.name)


    relu1 = relu(bn1.get_output(0), network)
    relu1.name = weight_key_prefix_first + "relu1"
    print("-------------", relu1.name)


    conv2_w = weights[weight_key_prefix_first + 'conv2.weight'].cpu().numpy()
    conv2 = conv3x3(relu1.get_output(0), out_channel, network, conv2_w)
    conv2.name = weight_key_prefix_first + "conv2"
    print("-------------", conv2.name)


    bn_gamma2 = weights[weight_key_prefix_first + 'bn2.weight'].cpu().numpy()        # bn gamma
    bn_bias2 = weights[weight_key_prefix_first + 'bn2.bias'].cpu().numpy()          # bn beta
    bn_mean2 = weights[weight_key_prefix_first + 'bn2.running_mean'].cpu().numpy()  # bn mean
    bn_var2 = weights[weight_key_prefix_first + 'bn2.running_var'].cpu().numpy()   # bn var sqrt
    bn2 = batchnorm(conv2.get_output(0), network,
                    bn_gamma2, bn_bias2, bn_mean2, bn_var2)
    bn2.name =  weight_key_prefix_first + "bn2"

    print("----------------bn2.shpae: ",bn2.get_output(0).shape)
    print("-------------", bn2.name)

    # out_tensor = bn2.get_output(0) + input_tensor
    if downsample:
        # 
        conv_downsample_weight = weights[weight_key_prefix_first + "downsample."+ "0.weight"].cpu().numpy() 
        # conv_downsample = network.add_convolution(input=bn2.get_output(0), num_output_maps=out_channel, kernel_shape=(1, 1), kernel=conv_downsample_weight)
        
        conv_downsample = conv2d(input_tensor, out_channel,network,conv_downsample_weight,1)
        conv_downsample.stride = (stride, stride)
        conv_downsample.name = weight_key_prefix_first + "downsample."+ "conv0"
        print("-------------", conv_downsample.name)


        bn_downsample_gamma = weights[weight_key_prefix_first + 'downsample.' + '1.weight'].cpu().numpy()        # bn gamma
        bn_downsample_bias = weights[weight_key_prefix_first + "downsample."+ '1.bias'].cpu().numpy()          # bn beta
        bn_downsample_mean = weights[weight_key_prefix_first + "downsample."+ '1.running_mean'].cpu().numpy()  # bn mean
        bn_downsample_var = weights[weight_key_prefix_first + "downsample."+ '1.running_var'].cpu().numpy()   # bn var sqrt
        bn_downsample = batchnorm(conv_downsample.get_output(0), network,
                        bn_downsample_gamma, bn_downsample_bias, bn_downsample_mean, bn_downsample_var)
        bn_downsample.name = weight_key_prefix_first + 'downsample.' + '1'
        print("-------------", bn_downsample.name)

        
        residual =  bn_downsample.get_output(0)
        residual.name = weight_key_prefix_first + "residual"


        print( " downsampl residual shape:", residual.shape)


    
    residual_tensor = tensor_add(bn2.get_output(0), residual, network)

    relu_out = relu(residual_tensor.get_output(0), network)
    relu_out.name = weight_key_prefix_first + "relu_out"
    print("-------------", relu_out.name)
    # out_tensor = relu_out.get_output(0)

    return relu_out

    """
layer1.0.conv1.weight torch.Size([64, 64, 3, 3])
layer1.0.bn1.weight torch.Size([64])
layer1.0.bn1.bias torch.Size([64])
layer1.0.bn1.running_mean torch.Size([64])
layer1.0.bn1.running_var torch.Size([64])
layer1.0.bn1.num_batches_tracked torch.Size([])
layer1.0.conv2.weight torch.Size([64, 64, 3, 3])
layer1.0.bn2.weight torch.Size([64])
layer1.0.bn2.bias torch.Size([64])
layer1.0.bn2.running_mean torch.Size([64])
layer1.0.bn2.running_var torch.Size([64])
layer1.0.bn2.num_batches_tracked torch.Size([])
layer1.1.conv1.weight torch.Size([64, 64, 3, 3])
layer1.1.bn1.weight torch.Size([64])
layer1.1.bn1.bias torch.Size([64])
layer1.1.bn1.running_mean torch.Size([64])
layer1.1.bn1.running_var torch.Size([64])
layer1.1.bn1.num_batches_tracked torch.Size([])
layer1.1.conv2.weight torch.Size([64, 64, 3, 3])
layer1.1.bn2.weight torch.Size([64])
layer1.1.bn2.bias torch.Size([64])
layer1.1.bn2.running_mean torch.Size([64])
layer1.1.bn2.running_var torch.Size([64])
layer1.1.bn2.num_batches_tracked torch.Size([])
layer2.0.conv1.weight torch.Size([128, 64, 3, 3])
layer2.0.bn1.weight torch.Size([128])
layer2.0.bn1.bias torch.Size([128])
layer2.0.bn1.running_mean torch.Size([128])
layer2.0.bn1.running_var torch.Size([128])
layer2.0.bn1.num_batches_tracked torch.Size([])
layer2.0.conv2.weight torch.Size([128, 128, 3, 3])
layer2.0.bn2.weight torch.Size([128])
layer2.0.bn2.bias torch.Size([128])
layer2.0.bn2.running_mean torch.Size([128])
layer2.0.bn2.running_var torch.Size([128])
layer2.0.bn2.num_batches_tracked torch.Size([])
layer2.0.downsample.0.weight torch.Size([128, 64, 1, 1])
layer2.0.downsample.1.weight torch.Size([128])
layer2.0.downsample.1.bias torch.Size([128])
layer2.0.downsample.1.running_mean torch.Size([128])
layer2.0.downsample.1.running_var torch.Size([128])
layer2.0.downsample.1.num_batches_tracked torch.Size([])
layer2.1.conv1.weight torch.Size([128, 128, 3, 3])
layer2.1.bn1.weight torch.Size([128])
layer2.1.bn1.bias torch.Size([128])
layer2.1.bn1.running_mean torch.Size([128])
layer2.1.bn1.running_var torch.Size([128])
layer2.1.bn1.num_batches_tracked torch.Size([])
layer2.1.conv2.weight torch.Size([128, 128, 3, 3])
layer2.1.bn2.weight torch.Size([128])
layer2.1.bn2.bias torch.Size([128])
layer2.1.bn2.running_mean torch.Size([128])
layer2.1.bn2.running_var torch.Size([128])
layer2.1.bn2.num_batches_tracked torch.Size([])
layer3.0.conv1.weight torch.Size([256, 128, 3, 3])
layer3.0.bn1.weight torch.Size([256])
layer3.0.bn1.bias torch.Size([256])
layer3.0.bn1.running_mean torch.Size([256])
layer3.0.bn1.running_var torch.Size([256])
layer3.0.bn1.num_batches_tracked torch.Size([])
layer3.0.conv2.weight torch.Size([256, 256, 3, 3])
layer3.0.bn2.weight torch.Size([256])
layer3.0.bn2.bias torch.Size([256])
layer3.0.bn2.running_mean torch.Size([256])
layer3.0.bn2.running_var torch.Size([256])
layer3.0.bn2.num_batches_tracked torch.Size([])
layer3.0.downsample.0.weight torch.Size([256, 128, 1, 1])
layer3.0.downsample.1.weight torch.Size([256])
layer3.0.downsample.1.bias torch.Size([256])
layer3.0.downsample.1.running_mean torch.Size([256])
layer3.0.downsample.1.running_var torch.Size([256])
layer3.0.downsample.1.num_batches_tracked torch.Size([])
layer3.1.conv1.weight torch.Size([256, 256, 3, 3])
layer3.1.bn1.weight torch.Size([256])
layer3.1.bn1.bias torch.Size([256])
layer3.1.bn1.running_mean torch.Size([256])
layer3.1.bn1.running_var torch.Size([256])
layer3.1.bn1.num_batches_tracked torch.Size([])
layer3.1.conv2.weight torch.Size([256, 256, 3, 3])
layer3.1.bn2.weight torch.Size([256])
layer3.1.bn2.bias torch.Size([256])
layer3.1.bn2.running_mean torch.Size([256])
layer3.1.bn2.running_var torch.Size([256])
layer3.1.bn2.num_batches_tracked torch.Size([])
layer4.0.conv1.weight torch.Size([512, 256, 3, 3])
layer4.0.bn1.weight torch.Size([512])
layer4.0.bn1.bias torch.Size([512])
layer4.0.bn1.running_mean torch.Size([512])
layer4.0.bn1.running_var torch.Size([512])
layer4.0.bn1.num_batches_tracked torch.Size([])
layer4.0.conv2.weight torch.Size([512, 512, 3, 3])
layer4.0.bn2.weight torch.Size([512])
layer4.0.bn2.bias torch.Size([512])
layer4.0.bn2.running_mean torch.Size([512])
layer4.0.bn2.running_var torch.Size([512])
layer4.0.bn2.num_batches_tracked torch.Size([])
layer4.0.downsample.0.weight torch.Size([512, 256, 1, 1])
layer4.0.downsample.1.weight torch.Size([512])
layer4.0.downsample.1.bias torch.Size([512])
layer4.0.downsample.1.running_mean torch.Size([512])
layer4.0.downsample.1.running_var torch.Size([512])
layer4.0.downsample.1.num_batches_tracked torch.Size([])
layer4.1.conv1.weight torch.Size([512, 512, 3, 3])
layer4.1.bn1.weight torch.Size([512])
layer4.1.bn1.bias torch.Size([512])
layer4.1.bn1.running_mean torch.Size([512])
layer4.1.bn1.running_var torch.Size([512])
layer4.1.bn1.num_batches_tracked torch.Size([])
layer4.1.conv2.weight torch.Size([512, 512, 3, 3])
layer4.1.bn2.weight torch.Size([512])
layer4.1.bn2.bias torch.Size([512])
layer4.1.bn2.running_mean torch.Size([512])
layer4.1.bn2.running_var torch.Size([512])
layer4.1.bn2.num_batches_tracked torch.Size([])
    """
def make_layer(network, weights, input_tensor, out_channel, layer_id, block_sizes=2, stride=1):
    downsample = None
    if stride != 1 or 64 != out_channel :
        downsample = True
    sub1 = basic_block_network(input_tensor, network, weights, out_channel, stride=stride, layerweight_id=layer_id, layerweight_sub_id=0, downsample=downsample)
    sub2 = basic_block_network(sub1.get_output(0), network, weights, out_channel, stride=1, layerweight_id=layer_id, layerweight_sub_id=1)
    return sub2

"""
conv1.weight torch.Size([64, 3, 7, 7])
bn1.weight torch.Size([64])
bn1.bias torch.Size([64])
bn1.running_mean torch.Size([64])
bn1.running_var torch.Size([64])
bn1.num_batches_tracked torch.Size([])
"""
def populate_network(network, weights):
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    out_planes = 64
    
     # conv0 relu
    conv0_w = weights['conv1.weight'].cpu().numpy()
    # conv0_b = weights['cnn.conv_0.bias'].cpu().numpy()
    conv0 = network.add_convolution(input=input_tensor, num_output_maps=out_planes, kernel_shape=(7, 7), kernel=conv0_w)
    conv0.stride = (2, 2)
    conv0.padding = (3, 3)

    #bn1
    bn1_gamma1 = weights['bn1.weight'].cpu().numpy()        # bn gamma
    bn1_bias1 = weights['bn1.bias'].cpu().numpy()          # bn beta
    bn1_mean1 = weights['bn1.running_mean'].cpu().numpy()  # bn mean
    bn1_var1 = weights['bn1.running_var'].cpu().numpy()   # bn var sqrt
    bn1 = batchnorm(conv0.get_output(0), network, bn1_gamma1, bn1_bias1, bn1_mean1, bn1_var1)
    bn1.name =  'bn1'

    #relu0
    relu0 = network.add_activation(input=bn1.get_output(0), type=trt.ActivationType.RELU)
    #pulling1
    pooling1 = network.add_pooling(relu0.get_output(0), trt.PoolingType.MAX, (3, 3))
    pooling1.stride = (2, 2)
    pooling1.padding = (1, 1)

    
    # print("----------------pooling1.shpae: ",pooling1.get_output(0).shape)


    layer1_out = make_layer(network,weights,pooling1.get_output(0),64,layer_id=1, stride=1)
    print("----------------layer1_out.shpae: ",layer1_out.get_output(0).shape)

    layer2_out = make_layer(network,weights,layer1_out.get_output(0),128,layer_id=2, stride=2)
    print("----------------layer2_out.shpae: ",layer2_out.get_output(0).shape)

    layer3_out = make_layer(network,weights,layer2_out.get_output(0),256,layer_id=3, stride=2)
    print("----------------layer3_out.shpae: ",layer3_out.get_output(0).shape)

    layer4_out = make_layer(network,weights,layer3_out.get_output(0),512,layer_id=4, stride=2)
    print("----------------layer4_out.shpae: ",layer4_out.get_output(0).shape)

    layer1_out.get_output(0).name = "out1"
    layer2_out.get_output(0).name = "out2"
    layer3_out.get_output(0).name = "out3"
    layer4_out.get_output(0).name = "out4"
    network.mark_output(tensor=layer1_out.get_output(0))  # (1, 64, 80, 80
    network.mark_output(tensor=layer2_out.get_output(0))  # (1, 128, 40, 40
    network.mark_output(tensor=layer3_out.get_output(0))  # (1, 256, 20, 20
    network.mark_output(tensor=layer4_out.get_output(0))   # (1, 512, 10, 10






def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network:
        builder.max_workspace_size = common.GiB(1)

        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.

        engine = builder.build_cuda_engine(network)

        return engine


if __name__ == "__main__":
    weights = torch.load("resnet18.pth")
    #   weights = checkpoint["state_dict"]
    for k in weights:
        print(k, weights[k].size())

    with build_engine(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:

            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            # input_numpy = np.ones((1, 64, 3, 3))
            input_numpy = np.ones((1, 3, 320, 320))

            input_numpy = input_numpy.astype(np.float32)

            input_numpy = input_numpy.ravel()  # 4816896
            print("-----------------input ---------------")
            print(input_numpy)
            inputs[0].host = input_numpy
            infer_outputs = common.do_inference(
                context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            # pred = np.argmax(output)
            # print(infer_outputs)
            # print("Test Case: " + str(case_num))
            # print("Prediction: " + str(pred))
            # print(output.reshape(1, 64, 3, 3))
            # for one_out in infer_outputs:
            #     print(one_out.shape)
            #     if on
            #     print(one_out)

            # print(infer_outputs)

            out1 = infer_outputs[0].reshape(1, 64, 80, 80)
            out2 = infer_outputs[1].reshape(1, 128, 40, 40)
            out3 = infer_outputs[2].reshape(1, 256, 20, 20)
            out4 = infer_outputs[3].reshape(1, 512, 10, 10)

            print("---out1---")
            print(out1)
            # print("---out2---")
            # print(out2)

            # print("---out3---")
            # print(out3)

            # print("---out4---")
            # print(out4)






'''

'''
