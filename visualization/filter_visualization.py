import os
import sys
sys.path.append("..")
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
from visualization.tf_plot import tf_plot, get_tf_data_efficient, subtract_frequency
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import models_generation
from reportlab.platypus import Paragraph
from visualization.pdf_utils import get_image
from reportlab.lib.styles import getSampleStyleSheet
from BCI_IV_2a_experiment import config_to_dict
styles = getSampleStyleSheet()
from collections import OrderedDict, defaultdict
from utils import label_by_idx

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
        print(f'created optimal example for layer {select_layer}, channel {c}')
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
    class_examples = []
    for class_idx in range(globals.get('n_classes')):
        class_examples.append(train_set[subject_id].X[np.where(train_set[subject_id].y == class_idx)])
    for index, layer in enumerate(list(pretrained_model.children())[:-1]):
        act_maps = []
        for class_idx in range(globals.get('n_classes')):
            act_maps.append(plot_one_tensor(get_intermediate_act_map
                                            (class_examples[class_idx], index, pretrained_model),
                                            f'Layer {index}, {label_by_idx(class_idx)}'))
        img_paths.extend(act_maps)
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
            class_str = ''
            if layer_idx >= len(list(pretrained_model.children())) - 3:
                class_str = f', class:{label_by_idx(chan_idx)}'
            plot_dict[(layer_idx, chan_idx)] = tf_plot(tf_data,
                                                      f'TF plot of example {example_idx} for layer '
                                                      f'{layer_idx}, channel {chan_idx}{class_str}',max_value)
            print(f'plot most activating TF for layer {layer_idx}, channel {chan_idx}')

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
        class_str = ''
        if layer_idx >= len(list(pretrained_model.children())) - 3:
            class_str = f', class:{label_by_idx(chan_idx)}'
        for chan_idx, example in enumerate(max_examples):
            plot_imgs[(layer_idx, chan_idx)] = tf_plot([plot_dict[(layer_idx, chan_idx, c)] for c in eeg_chans],
                                                       f'TF plot of optimal example for layer {layer_idx},'
                                                       f' channel {chan_idx}{class_str}', max_value)
            print(f'plot gradient ascent TF for layer {layer_idx}, channel {chan_idx}')

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
    class_examples = []
    for class_idx in range(globals.get('n_classes')):
        class_examples.append(train_set[subject_id].X[np.where(train_set[subject_id].y == class_idx)])
    chan_data = []
    for class_idx in range(globals.get('n_classes')):
        chan_data.append(defaultdict(list))
        for example in class_examples[class_idx]:
            for eeg_chan in eeg_chans:
                chan_data[-1][eeg_chan].append(get_tf_data_efficient(example[None, :, :], eeg_chan, globals.get('frequency')))
    avg_tfs = []
    for class_idx in range(globals.get('n_classes')):
        class_tfs = []
        for eeg_chan in eeg_chans:
            class_tfs.append(np.average(np.array(chan_data[class_idx][eeg_chan]), axis=0))
        avg_tfs.append(class_tfs)
    max_value = max(*[np.max(np.array(class_chan_avg_tf)) for class_chan_avg_tf in avg_tfs])
    tf_plots = []
    for class_idx in range(globals.get('n_classes')):
        tf_plots.append(tf_plot(avg_tfs[class_idx], f'average TF for {label_by_idx(class_idx)}', max_value))
    story = [get_image(tf) for tf in tf_plots]
    create_pdf_from_story(f'results/{date_time}/step5_tf_plots_avg_per_class.pdf', story)
    for tf in tf_plots:
        os.remove(tf)


if __name__ == '__main__':
    globals.set_dummy_config()
    globals.set('valid_set_fraction', 0.2)
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
    config_dict = config_to_dict('visualization_configurations/viz_config.ini')
    globals.set('dataset', config_dict['DEFAULT']['dataset'])
    globals.set('models_dir', config_dict['DEFAULT']['models_dir'])
    globals.set('model_name', config_dict['DEFAULT']['model_name'])
    set_params_by_dataset('../configurations/dataset_params.ini')
    model_selection = 'evolution'
    cnn_layer = {'evolution': 10, 'deep4': 25}
    filter_pos = {'evolution': 0, 'deep4': 0}
    model = {'evolution': torch.load(f'../models/{globals.get("models_dir")}/{globals.get("model_name")}'),
                        'deep4': target_model('deep')}
    subject_id = config_dict['DEFAULT']['subject_id']
    train_set = {}
    val_set = {}
    test_set = {}
    train_set[subject_id], val_set[subject_id], test_set[subject_id] = \
        get_train_val_test(globals.get('data_folder'), subject_id, globals.get('low_cut_hz'))

    orig_data = [get_tf_data_efficient(train_set[1].X[0:2], 0, 250)]
    tf_plot(orig_data, 'before subtraction')
    subtracted_data = subtract_frequency(train_set[1].X[0:2], 10, 250)
    tf_plot([get_tf_data_efficient(subtracted_data, 0, 250)], 'after subtraction')

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
    create_optimal_samples_per_filter(pretrained_model, date_time, steps=1)
    get_avg_class_tf(train_set, date_time)
