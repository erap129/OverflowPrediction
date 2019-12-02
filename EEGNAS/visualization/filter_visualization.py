import configparser
import itertools
import os
import pickle
import sys
import traceback
from collections import defaultdict, OrderedDict
from copy import deepcopy
import gc
from sacred import Experiment
from sacred.observers import MongoObserver

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../nsga_net")
from EEGNAS.utilities.data_utils import write_config
from EEGNAS.utilities.report_generation import add_params_to_name
from nsga_net.models.micro_models import NetworkCIFAR
from nsga_net.search.micro_encoding import decode, convert
from EEGNAS.visualization.external_models import MultivariateLSTM
from EEGNAS.evolution.nn_training import NN_Trainer
from EEGNAS.visualization import viz_reports
from EEGNAS.utilities.config_utils import set_default_config, update_global_vars_from_config_dict, get_configurations, \
    get_multiple_values
from EEGNAS.utilities.misc import concat_train_val_sets, get_exp_id
import logging
from EEGNAS.visualization.dsp_functions import butter_bandstop_filter, butter_bandpass_filter
import torch
from braindecode.torch_ext.util import np_to_var
from EEGNAS import global_vars
from EEGNAS.data_preprocessing import get_dataset
from EEGNAS_experiment import set_params_by_dataset, get_normal_settings
import matplotlib.pyplot as plt
from EEGNAS.utilities.misc import create_folder
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
plt.interactive(False)
ex = Experiment()
FEATURE_VALUES = OrderedDict()
LAST_EXP_FOLDER = ''
MODEL_ALIASES = {'cnn': ('411_1_netflow_regression_daily_normalized', 'best_model_1.th'),
                 'rnn': ('rnn', 'rnn'),
                 'nsga': ('nsga_micro_netflow_normalized', 'best_genome_normalized.pkl')}

def get_intermediate_act_map(data, select_layer, model):
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
      x = l(x)
    act_map = x.cpu().detach().numpy()
    act_map_avg = np.average(act_map, axis=0).swapaxes(0, 1).squeeze(axis=2)
    return act_map_avg


def matrix_mse(a, b):
    return np.square(a - b).mean()


@ex.main
def main():
    global FEATURE_VALUES, LAST_EXP_FOLDER
    exp_folder = f"results/{exp_name}"
    create_folder(exp_folder)
    res = getattr(viz_reports, f'{global_vars.get("report")}_report')(model, dataset, exp_folder)
    if global_vars.get('report') == 'feature_importance':
        for segment in ['train', 'test', 'both']:
            FEATURE_VALUES[(global_vars.get('model_alias'), global_vars.get('explainer'), global_vars.get('iteration'), segment)] = res[segment]
    write_config(global_vars.config, f"{exp_folder}/config_{exp_name}.ini")
    LAST_EXP_FOLDER = exp_folder


if __name__ == '__main__':
    configs = configparser.ConfigParser()
    configs.read('visualization_configurations/viz_config.ini')
    configurations = get_configurations(sys.argv[1], configs, set_exp_name=False)
    global_vars.init_config('configurations/config.ini')
    set_default_config('../configurations/config.ini')
    global_vars.set('cuda', True)
    exp_id = get_exp_id('results')
    multiple_values = get_multiple_values(configurations)
    prev_dataset = None
    for index, configuration in enumerate(configurations):
        update_global_vars_from_config_dict(configuration)
        if index + 1 < global_vars.get('start_exp_idx'):
            continue
        if global_vars.get('model_alias') == 'nsga' and global_vars.get('explainer') == 'integrated_gradients':
            continue
        if global_vars.get('model_alias'):
            alias = global_vars.get('model_alias')
            global_vars.set('models_dir', MODEL_ALIASES[alias][0])
            global_vars.set('model_name', MODEL_ALIASES[alias][1])
        global_vars.set('band_filter', {'pass': butter_bandpass_filter,
                                        'stop': butter_bandstop_filter}[global_vars.get('band_filter')])

        set_params_by_dataset('../configurations/dataset_params.ini')
        subject_id = global_vars.get('subject_id')
        dataset = get_dataset(subject_id)
        prev_dataset = global_vars.get('dataset')
        exp_name = f"{exp_id}_{index+1}_{global_vars.get('report')}_{global_vars.get('dataset')}"
        exp_name = add_params_to_name(exp_name, multiple_values)

        if global_vars.get('model_name') == 'rnn':
            model = MultivariateLSTM(dataset['train'].X.shape[1], 100, global_vars.get('batch_size'),
                                     global_vars.get('input_height'), global_vars.get('n_classes'), eegnas=True)

        elif 'nsga' in global_vars.get("models_dir"):
            with open(f'../models/{global_vars.get("models_dir")}/{global_vars.get("model_name")}', 'rb') as f:
                genotype = pickle.load(f)
                genotype = decode(convert(genotype))
                model = NetworkCIFAR(24, global_vars.get('steps_ahead'), global_vars.get('max_handovers'), 11, False,
                                     genotype)
                model.droprate = 0.0
                model.single_output = True

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
        print(f'generating {global_vars.get("report")} report for model:')
        print(model)
        ex.config = {}
        ex.add_config(configuration)
        if len(ex.observers) == 0 and len(sys.argv) <= 2:
            ex.observers.append(MongoObserver.create(url=f'mongodb://{global_vars.get("mongodb_server")}/{global_vars.get("mongodb_name")}',
                                                     db_name=global_vars.get("mongodb_name")))
        global_vars.set('sacred_ex', ex)
        try:
            ex.run(options={'--name': exp_name})
            gc.collect()
        except (MemoryError, RuntimeError) as me:
            print(f'failed experiment {exp_id}_{index+1} because of memory error, trying with sampling rate/2...')
            global_vars.set('explainer_sampling_rate', global_vars.get('explainer_sampling_rate') / 2)
            ex.run(options={'--name': exp_name})
        except Exception as e:
            print('experiment failed. Exception message: %s' % (str(e)))
            print(traceback.format_exc())
            print(f'failed experiment {exp_id}_{index+1}, continuing...')


    if len(FEATURE_VALUES.keys()) > 0:
        # models = list(set([i for (i,j,k) in list(FEATURE_VALUES.keys())]))
        # iterations = list(set([j for (i,j,k) in list(FEATURE_VALUES.keys())]))
        # explainers = list(set([k for (i,j,k) in list(FEATURE_VALUES.keys())]))
        # for segment in ['train', 'test', 'both']:
        #     mse_avg_per_model = {}
        #     for model in models:
        #         for explainer in explainers:
        #             model_avg = 0
        #             mse_calc = []
        #             for iteration in iterations:
        #                 mse_calc.append(FEATURE_VALUES[(model, iteration, explainer)][segment])
        #             for (feature_val_1, feature_val_2) in itertools.combinations(mse_calc, 2):
        #                     model_avg += (np.square(feature_val_1 - feature_val_2)).mean()
        #             model_avg = model_avg / len(mse_calc)
        #             with open(f"{folder_name}/feature_mse.txt", "a") as f:
        #                 f.write(f'{segment}: average MSE between iterations for model {model} for explainer {explainer} is {model_avg}\n')
        #             mse_avg_per_model[model] = np.mean(mse_calc, axis=0)
        #
        #     for (model1, model2) in itertools.combinations(models, 2):
        #         with open(f"{folder_name}/feature_mse.txt", "a") as f:
        #             f.write(f'{segment}: MSE between model {model1} and model {model2} is '
        #                     f'{(np.square(mse_avg_per_model[model1] - mse_avg_per_model[model2])).mean()}\n')
        # df = pd.DataFrame.from_dict(FEATURE_VALUES)
        l = len(FEATURE_VALUES.keys())
        results = np.zeros((l, l))
        for i, ac in enumerate(FEATURE_VALUES.values()):
            for j, bc in enumerate(FEATURE_VALUES.values()):
                results[j, i] = matrix_mse(ac, bc)
        corr = pd.DataFrame(results, index=list(FEATURE_VALUES.keys()), columns=list(FEATURE_VALUES.keys()))
        plt.clf()
        ax = sns.heatmap(
            corr,
            vmin=corr.min().min(), vmax=corr.max().max(),
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90,
            horizontalalignment='right'
        )
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        plt.tight_layout()
        plt.savefig(f"{LAST_EXP_FOLDER}/feature_correlations.png", dpi=300)