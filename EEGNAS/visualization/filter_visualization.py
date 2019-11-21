import configparser
import os
import sys
from copy import deepcopy
from sacred import Experiment
from sacred.observers import MongoObserver
sys.path.append("..")
sys.path.append("../..")
from EEGNAS.visualization.external_models import MultivariateLSTM
from EEGNAS.evolution.nn_training import NN_Trainer
from EEGNAS.visualization import viz_reports
from EEGNAS.utilities.config_utils import set_default_config, update_global_vars_from_config_dict, get_configurations
from EEGNAS.utilities.misc import concat_train_val_sets
import logging
from EEGNAS.visualization.dsp_functions import butter_bandstop_filter, butter_bandpass_filter
from EEGNAS.visualization.signal_plotting import plot_one_tensor
import torch
from braindecode.torch_ext.util import np_to_var
from EEGNAS import global_vars
from EEGNAS.data_preprocessing import get_dataset
from EEGNAS_experiment import set_params_by_dataset, get_normal_settings
import matplotlib.pyplot as plt
from EEGNAS.utilities.misc import create_folder
from EEGNAS.visualization.pdf_utils import create_pdf
import numpy as np
from EEGNAS.visualization.wavelet_functions import subtract_frequency
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet
from EEGNAS.utilities.misc import label_by_idx
styles = getSampleStyleSheet()
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
plt.interactive(False)
ex = Experiment()


def get_intermediate_act_map(data, select_layer, model):
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
      x = l(x)
    act_map = x.cpu().detach().numpy()
    act_map_avg = np.average(act_map, axis=0).swapaxes(0, 1).squeeze(axis=2)
    return act_map_avg


def plot_avg_activation_maps(pretrained_model, dataset, date_time):
    img_paths = []
    class_examples = []
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(dataset.X[np.where(dataset.y == class_idx)])
    for index, layer in enumerate(list(pretrained_model.children())[:-1]):
        act_maps = []
        for class_idx in range(global_vars.get('n_classes')):
            act_maps.append(plot_one_tensor(get_intermediate_act_map
                                            (class_examples[class_idx], index, pretrained_model),
                                            f'Layer {index}, {label_by_idx(class_idx)}'))
        img_paths.extend(act_maps)
    create_pdf(f'results/{date_time}_{global_vars.get("dataset")}/step2_avg_activation_maps.pdf', img_paths)
    for im in img_paths:
        os.remove(im)


def frequency_correlation_single_example(pretrained_model, data, discriminating_layer, low_freq, high_freq):
    # find the most prominent example in each class
    # for each freq:
    #   get probability of correct class after each perturbation
    # plot probabilities as a function of the frequency
    max_per_class = get_max_examples_per_channel(data, discriminating_layer, pretrained_model)
    for chan_idx, example_idx in enumerate(max_per_class):
        correct_class_probas = []
        for freq in range(low_freq, high_freq+1):
            data_to_perturb = deepcopy(data)
            perturbed_data = subtract_frequency(data_to_perturb, freq, global_vars.get('frequency'))
            pretrained_model.eval()
            probas = pretrained_model(data[example_idx])
            print
    pass


@ex.main
def main():
    getattr(viz_reports, f'{global_vars.get("report")}_report')(model, dataset, folder_name)


if __name__ == '__main__':
    configs = configparser.ConfigParser()
    configs.read('visualization_configurations/viz_config.ini')
    configurations = get_configurations(sys.argv[1], configs, set_exp_name=False)
    global_vars.init_config('configurations/config.ini')
    set_default_config('../configurations/config.ini')
    global_vars.set('cuda', True)

    prev_dataset = None
    for configuration in configurations:
        update_global_vars_from_config_dict(configuration)
        global_vars.set('band_filter', {'pass': butter_bandpass_filter,
                                        'stop': butter_bandstop_filter}[global_vars.get('band_filter')])

        set_params_by_dataset('../configurations/dataset_params.ini')
        subject_id = global_vars.get('subject_id')
        dataset = get_dataset(subject_id)
        prev_dataset = global_vars.get('dataset')

        if global_vars.get('model_name') == 'rnn':
            model = MultivariateLSTM(dataset['train'].X.shape[1], 100, global_vars.get('batch_size'),
                                     global_vars.get('input_height'), global_vars.get('n_classes'), eegnas=True)
        else:
            model = torch.load(f'../models/{global_vars.get("models_dir")}/{global_vars.get("model_name")}')
        model.cuda()

        if global_vars.get('finetune_model'):
            stop_criterion, iterator, loss_function, monitors = get_normal_settings()
            trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
            model = trainer.train_model(model, dataset, final_evaluation=True)

        concat_train_val_sets(dataset)

        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y-%H:%M")
        folder_name = f'results/{date_time}_{global_vars.get("dataset")}_{global_vars.get("report")}'
        create_folder(folder_name)
        print(f'generating {global_vars.get("report")} report for model:')
        print(model)
        if global_vars.get('to_eeglab'):
            create_folder(f'{folder_name}/{global_vars.get("report")}')

        exp_name = f"{global_vars.get('dataset')}_{global_vars.get('report')}"
        ex.config = {}
        ex.add_config(configuration)
        if len(ex.observers) == 0 and len(sys.argv) <= 2:
            ex.observers.append(MongoObserver.create(url=f'mongodb://132.72.80.67/{global_vars.get("mongodb_name")}',
                                                     db_name=global_vars.get("mongodb_name")))
        global_vars.set('sacred_ex', ex)
        ex.run(options={'--name': exp_name})


