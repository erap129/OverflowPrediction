import copy
import random
import numpy as np
from braindecode.torch_ext.util import np_to_var
from model_generation.abstract_layers import *
from model_generation.custom_modules import *
from torch import nn
from torch.nn import init
from utilities.misc import get_index_of_last_layertype

from model_generation.custom_modules import _squeeze_final_output


def new_model_from_structure_pytorch(layer_collection, applyFix=False, check_model=False):
    model = nn.Sequential()
    if global_vars.get('channel_dim') != 'channels' or global_vars.get('exp_type') == 'target':
        model.add_module('dimshuffle', _transpose(shape=[0, 3, 2, 1]))
    if global_vars.get('time_factor') != -1:
        model.add_module('stack_by_time', Expression(_stack_input_by_time))
    activations = {'elu': nn.ELU, 'softmax': nn.Softmax, 'sigmoid': nn.Sigmoid}
    input_shape = (2, global_vars.get('eeg_chans'), global_vars.get('input_height'), global_vars.get('input_width'))
    for i in range(len(layer_collection)):
        layer = layer_collection[i]
        if i > 0:
            out = model.forward(np_to_var(np.ones(
                input_shape,
                dtype=np.float32)))
            prev_channels = out.cpu().data.numpy().shape[1]
            prev_height = out.cpu().data.numpy().shape[2]
            prev_width = out.cpu().data.numpy().shape[3]
        else:
            prev_channels = global_vars.get('eeg_chans')
            prev_height = global_vars.get('input_height')
            prev_width = global_vars.get('input_width')
            # if global_vars.get('channel_dim') == 'channels':
            #     prev_channels = global_vars.get('eeg_chans')
            #     prev_eeg_channels = 1
        if isinstance(layer, PoolingLayer):
            while applyFix and (prev_height-layer.pool_height) / layer.stride_height < 1:
                if random.uniform(0,1) < 0.5 and layer.pool_height > 1:
                    layer.pool_height -= 1
                elif layer.stride_height > 1:
                    layer.stride_height -= 1
                if layer.pool_height == 1 and layer.stride_height == 1:
                    break
            # if global_vars.get('channel_dim') == 'channels':
            #     layer.pool_eeg_chan = 1
            model.add_module('%s_%d' % (type(layer).__name__, i), nn.MaxPool2d(kernel_size=(int(layer.pool_height), int(layer.pool_width)),
                                                                      stride=(int(layer.stride_height), 1)))

        elif isinstance(layer, ConvLayer):
            layer_class = nn.Conv2d
            if layer.kernel_height == 'down_to_one' or i >= global_vars.get('num_layers'):
                if global_vars.get('autoencoder'):
                    layer.kernel_height = global_vars.get('input_height') - prev_height + 1
                    layer.kernel_width = global_vars.get('input_width') - prev_width + 1
                    layer_class = nn.ConvTranspose2d
                else:
                    layer.kernel_height = prev_height
                    layer.kernel_width = prev_width
                conv_name = 'conv_classifier'
            else:
                conv_name = '%s_%d' % (type(layer).__name__, i)
                if applyFix and layer.kernel_height > prev_height:
                    layer.kernel_height = prev_height
                if applyFix and layer.kernel_width > prev_width:
                    layer.kernel_width = prev_width
            # if global_vars.get('channel_dim') == 'channels':
            #     layer.kernel_eeg_chan = 1
            model.add_module(conv_name, layer_class(prev_channels, layer.filter_num,
                                                (layer.kernel_height, layer.kernel_width),
                                                stride=1))

        elif isinstance(layer, BatchNormLayer):
            model.add_module('%s_%d' % (type(layer).__name__, i), nn.BatchNorm2d(prev_channels,
                                                                                 momentum=global_vars.get('batch_norm_alpha'),
                                                                                 affine=True, eps=1e-5), )

        elif isinstance(layer, ActivationLayer):
            model.add_module('%s_%d' % (layer.activation_type, i), activations[layer.activation_type]())


        elif isinstance(layer, DropoutLayer):
            model.add_module('%s_%d' % (type(layer).__name__, i), nn.Dropout(p=global_vars.get('dropout_p')))

        elif isinstance(layer, IdentityLayer):
            model.add_module('%s_%d' % (type(layer).__name__, i), IdentityModule())

        elif isinstance(layer, FlattenLayer):
            model.add_module('squeeze', _squeeze_final_output())

    if applyFix:
        return layer_collection
    if check_model:
        return
    if global_vars.get('autoencoder'):
        last_conv_idx = get_index_of_last_layertype(model, nn.ConvTranspose2d)
    else:
        last_conv_idx = get_index_of_last_layertype(model, nn.Conv2d)
    init.xavier_uniform_(list(model._modules.items())[last_conv_idx][1].weight, gain=1)
    init.constant_(list(model._modules.items())[last_conv_idx][1].bias, 0)
    return model


def check_legal_model(layer_collection):
    # if global_vars.get('channel_dim') == 'channels':
    #     input_chans = 1
    # else:
    #     input_chans = global_vars.get('eeg_chans')
    height = global_vars.get('input_height')
    width = global_vars.get('input_width')
    for layer in layer_collection:
        if type(layer) == ConvLayer:
            height = (height - layer.kernel_height) + 1
            width = (width - layer.kernel_width) + 1
        elif type(layer) == PoolingLayer:
            height = (height - layer.pool_height) / layer.stride_height + 1
            width = (width - layer.pool_width) / layer.stride_width + 1
        if height < 1 or width < 1:
            print(f"illegal model, height={height}, width={width}")
            return False
    # if global_vars.get('cropping'):
    #     return check_legal_cropping_model(layer_collection)
    return True


def uniform_model(n_layers, layer_type):
    layer_collection = []
    for i in range(n_layers):
        layer = layer_type()
        layer_collection.append(layer)
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return uniform_model(n_layers, layer_type)


def custom_model(layers):
    layer_collection = []
    for layer in layers:
        layer_collection.append(layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return custom_model(layers)


def random_layer():
    layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
    return layers[random.randint(0, 5)]()


def random_model(n_layers):
    layer_collection = []
    for i in range(n_layers):
        if global_vars.get('simple_start'):
            layer_collection.append(IdentityLayer())
        else:
            layer_collection.append(random_layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return random_model(n_layers)


def add_layer_to_state(new_model_state, layer, index, old_model_state):
    if type(layer).__name__ in ['BatchNormLayer', 'ConvLayer', 'PoolingLayer']:
        for k, v in old_model_state.items():
            if '%s_%d' % (type(layer).__name__, index) in k and \
                    k in new_model_state.keys() and new_model_state[k].shape == v.shape:
                new_model_state[k] = v


def finalize_model(layer_collection):
    if global_vars.get('grid'):
        return ModelFromGrid(layer_collection)
    layer_collection = copy.deepcopy(layer_collection)
    if global_vars.get('cropping'):
        final_conv_time = global_vars.get('final_conv_size')
    else:
        final_conv_time = 'down_to_one'
    conv_layer = ConvLayer(kernel_height=final_conv_time, kernel_width=1,
                           filter_num=global_vars.get('n_classes'))
    layer_collection.append(conv_layer)
    if global_vars.get('problem') == 'classification':
        activation = ActivationLayer('softmax')
        layer_collection.append(activation)
    flatten = FlattenLayer()
    layer_collection.append(flatten)
    return new_model_from_structure_pytorch(layer_collection)
