import os
import sys
sys.path.append("..")
import scipy
import torch
from braindecode.torch_ext.util import np_to_var
from models_generation import target_model
from naiveNAS import NaiveNAS
import globals
from torch import nn
from data_preprocessing import get_train_val_test
from BCI_IV_2a_experiment import get_normal_settings, set_params_by_dataset
import matplotlib.pyplot as plt
import matplotlib
from utils import createFolder
from visualization.cnn_layer_visualization import CNNLayerVisualization
from visualization.pdf_utils import create_pdf, create_pdf_from_story
import numpy as np
from visualization.tf_plot import tf_plot, get_tf_data, get_tf_data_efficient
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hiddenlayer as hl
from datetime import datetime
import models_generation
from reportlab.platypus import Paragraph, Image
from visualization.pdf_utils import get_image
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()
from collections import OrderedDict, defaultdict

matplotlib.use("TkAgg")
plt.interactive(False)
img_name_counter = 1


def get_max_examples_per_channel(data, select_layer, model):
    act_maps = {}
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for idx, example in enumerate(x):
        example_x = example[None, :, :, :]
        for l in modules[:select_layer + 1]:
            example_x = l(example_x)
        act_maps[idx] = example_x
    channels = act_maps[0].shape[1]
    selected_examples = np.zeros(channels)
    for c in range(channels):
        selected_examples[c]\
            = int(np.array([act_map.squeeze()[c].sum() for act_map in act_maps.values()]).argmax())
    return [int(x) for x in selected_examples]


def create_max_examples_per_channel(select_layer, model, steps=500):
    dummy_X = models_generation.get_dummy_input().cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
        dummy_X = l(dummy_X)
    channels = dummy_X.shape[1]
    act_maps = []
    for c in range(channels):
        layer_vis = CNNLayerVisualization(model, select_layer, c)
        act_maps.append(layer_vis.visualise_layer_with_hooks(steps))
    return act_maps


def get_intermediate_act_map(data, select_layer, model):
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
      x = l(x)
    act_map = x.cpu().detach().numpy()
    act_map_avg = np.average(act_map, axis=0).swapaxes(0, 1).squeeze(axis=2)
    return act_map_avg


def plot_tensors(tensor, title, num_cols=8):
    global img_name_counter
    tensor = np.swapaxes(tensor, 1, 2)
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        im = ax1.imshow(tensor[i].squeeze(axis=2), cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.suptitle(f'{title}, Tensor shape: {tensor.shape}')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    img_name = f'temp/{img_name_counter}.png'
    plt.savefig(f'{img_name}')
    plt.close('all')
    img_name_counter += 1
    return img_name


def plot_one_tensor(tensor, title):
    global img_name_counter
    if not tensor.ndim == 2:
        raise Exception("assumes a 2D tensor")
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(tensor.swapaxes(0,1), cmap='gray')
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(f'{title}, Tensor shape: {tensor.shape}')
    img_name = f'temp/{img_name_counter}.png'
    plt.savefig(img_name, bbox_inches='tight')
    plt.close('all')
    img_name_counter += 1
    return img_name


def plot_all_kernels_to_pdf(pretrained_model, date_time):
    img_paths = []
    for index, layer in enumerate(list(pretrained_model.children())):
        if type(layer) == nn.Conv2d:
            im = plot_tensors(layer.weight.detach().cpu().numpy(), f'Layer {index}')
            img_paths.append(im)
    create_pdf(f'results/{date_time}/step1_all_kernels.pdf', img_paths)
    for im in img_paths:
        os.remove(im)


def plot_avg_activation_maps(pretrained_model, train_set, date_time):
    img_paths = []
    left_X = train_set[subject_id].X[np.where(train_set[subject_id].y == 0)]
    right_X = train_set[subject_id].X[np.where(train_set[subject_id].y == 1)]
    for index, layer in enumerate(list(pretrained_model.children())[:-1]):
        left_act_map = plot_one_tensor(get_intermediate_act_map(left_X, index, pretrained_model), f'Layer {index} Left')
        right_act_map = plot_one_tensor(get_intermediate_act_map(right_X, index, pretrained_model), f'Layer {index} Right')
        img_paths.extend([left_act_map, right_act_map])
    create_pdf(f'results/{date_time}/step2_avg_activation_maps.pdf', img_paths)
    for im in img_paths:
        os.remove(im)


def find_optimal_samples_per_filter(pretrained_model, train_set, date_time, eeg_chans=None):
    if eeg_chans is None:
        eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    plot_dict = OrderedDict()
    for layer_idx, layer in enumerate(list(pretrained_model.children())):
        max_examples = get_max_examples_per_channel(train_set[subject_id].X, layer_idx, pretrained_model)
        for chan_idx, example_idx in enumerate(max_examples):
            tf_data = []
            for eeg_chan in eeg_chans:
                tf_data.append(get_tf_data_efficient(train_set[subject_id].X[example_idx][None, :, :], eeg_chan, 250))
            max_value = np.max(np.array(tf_data))
            plot_dict[(layer_idx, chan_idx)] = tf_plot(tf_data,
                                                      f'TF plot of example {example_idx} for layer {layer_idx}, channel {chan_idx}',
                                                       max_value)
    img_paths = list(plot_dict.values())
    story = []
    story.append(Paragraph('<br />\n'.join([f'{x}:{y}' for x,y in pretrained_model._modules.items()]), style=styles["Normal"]))
    for im in img_paths:
        story.append(get_image(im))
    create_pdf_from_story(f'results/{date_time}/step3_tf_plots_real.pdf', story)
    for im in img_paths:
        os.remove(im)


def create_optimal_samples_per_filter(pretrained_model, date_time, eeg_chans=None, steps=500):
    if eeg_chans is None:
        eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    plot_dict = OrderedDict()
    plot_imgs = OrderedDict()
    for layer_idx, layer in enumerate(list(pretrained_model.children())):
        max_examples = create_max_examples_per_channel(layer_idx, pretrained_model, steps=steps)
        max_value = 0
        for chan_idx, example in enumerate(max_examples):
            for eeg_chan in eeg_chans:
                plot_dict[(layer_idx, chan_idx, eeg_chan)] = get_tf_data_efficient(example, eeg_chan, 250)
                max_value = max(max_value, np.max(plot_dict[(layer_idx, chan_idx, eeg_chan)]))
        for chan_idx, example in enumerate(max_examples):
            plot_imgs[(layer_idx, chan_idx)] = tf_plot([plot_dict[(layer_idx, chan_idx, c)] for c in eeg_chans],
                                                       f'TF plot of optimal example'
                                                       f' for layer {layer_idx}, channel {chan_idx}', max_value)
    story = []
    img_paths = list(plot_imgs.values())
    story.append(
        Paragraph('<br />\n'.join([f'{x}:{y}' for x, y in pretrained_model._modules.items()]), style=styles["Normal"]))
    for im in img_paths:
        story.append(get_image(im))
    create_pdf_from_story(f'results/{date_time}/step4_tf_plots_optimal_test.pdf', story)
    for im in img_paths:
        os.remove(im)


def get_avg_class_tf(train_set, date_time, eeg_chans=None):
    if eeg_chans is None:
        eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    left_X = train_set[subject_id].X[np.where(train_set[subject_id].y == 0)]
    right_X = train_set[subject_id].X[np.where(train_set[subject_id].y == 1)]

    left_chan_data = defaultdict(list)
    right_chan_data = defaultdict(list)

    for left_example in left_X:
        for eeg_chan in eeg_chans:
            left_chan_data[eeg_chan].append(get_tf_data_efficient(left_example[None, :, :], eeg_chan, 250))

    for right_example in right_X:
        for eeg_chan in eeg_chans:
            right_chan_data[eeg_chan].append(get_tf_data_efficient(right_example[None, :, :], eeg_chan, 250))

    left_chan_avg_tf = []
    right_chan_avg_tf = []
    for eeg_chan in eeg_chans:
        left_chan_avg_tf.append(np.average(np.array(left_chan_data[eeg_chan]), axis=0))
        right_chan_avg_tf.append(np.average(np.array(right_chan_data[eeg_chan]), axis=0))
    max_value = max(np.max(np.array(left_chan_avg_tf)), np.max(np.array(right_chan_avg_tf)))
    left_img = tf_plot(left_chan_avg_tf, 'average TF for left hand', max_value)
    right_img = tf_plot(right_chan_avg_tf, 'average TF for right hand', max_value)
    story = [get_image(left_img), get_image(right_img)]
    create_pdf_from_story(f'results/{date_time}/step5_tf_plots_avg_per_class.pdf', story)
    os.remove(left_img)
    os.remove(right_img)


if __name__ == '__main__':
    globals.set_dummy_config()
    globals.set('input_time_len', 1125)
    globals.set('n_classes', 4)
    globals.set('eeg_chans', 22)
    globals.set('valid_set_fraction', 0.2)
    globals.set('dataset', 'BCI_IV_2b')
    globals.set('batch_size', 60)
    globals.set('do_early_stop', True)
    globals.set('remember_best', True)
    globals.set('max_epochs', 50)
    globals.set('max_increase_epochs', 1)
    globals.set('final_max_epochs', 1)
    globals.set('final_max_increase_epochs', 80)
    globals.set('cuda', True)
    globals.set('data_folder', '../../data/')
    globals.set('low_cut_hz', 0)
    globals.set('dataset', 'BCI_IV_2b')
    set_params_by_dataset()
    model_selection = 'evolution'
    cnn_layer = {'evolution': 10, 'deep4': 25}
    filter_pos = {'evolution': 0, 'deep4': 0}
    model_dir = '91_x_BCI_IV_2b'
    model_name = 'best_model_5_1_8_7_9_2_3_4_6.th'
    model = {'evolution': torch.load(f'../models/{model_dir}/{model_name}'),
                        'deep4': target_model('deep')}
    subject_id = 1
    train_set = {}
    val_set = {}
    test_set = {}
    train_set[subject_id], val_set[subject_id], test_set[subject_id] = \
        get_train_val_test(globals.get('data_folder'), subject_id, globals.get('low_cut_hz'))

    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    naiveNAS = NaiveNAS(iterator=iterator, exp_folder=None, exp_name=None,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id=subject_id, fieldnames=None, strategy='cross_subject',
                        evolution_file=None, csv_file=None)
    _, _, pretrained_model, _, _ = naiveNAS.evaluate_model(model[model_selection], final_evaluation=True)

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y-%H:%M:%S")
    createFolder(f'results/{date_time}')
    plot_all_kernels_to_pdf(pretrained_model, date_time)
    plot_avg_activation_maps(pretrained_model, train_set, date_time)
    find_optimal_samples_per_filter(pretrained_model, train_set, date_time)
    create_optimal_samples_per_filter(pretrained_model, date_time, steps='max')
    get_avg_class_tf(train_set, date_time)

