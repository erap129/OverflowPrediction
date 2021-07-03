import ast
import atexit
import gc
import itertools
import operator
import pickle
import random
import shutil
import sys
import os
import traceback
from collections import defaultdict
from copy import deepcopy
from statistics import mean
# import autokeras as ak

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/../EEGNAS')
sys.path.append(f'{dir_path}/../EEGNAS/EEGNAS')
sys.path.append(f'{dir_path}/../nsga_net')
sys.path.append(f'{dir_path}/statistical_anomaly_detection/src')
from nn_training import MTS_train, Optim
from EEGNAS.data.MTS_benchmarks.MTS_utils import Data_utility
from EEGNAS.data.netflow.netflow_data_utils import get_netflow_minmax_scaler, get_handover_locations
from EEGNAS.model_generation.simple_model_generation import create_ensemble_from_population_file
from fold_utils import powerset, filter_eegnas_population_files, plot_classification_curves, \
    calc_tpr_fpr_precision_recall, augment_dataset, create_regression_report, predict_by_batch, MTS_evaluate
from MHA_Net import MHANetModel
from LSTNet import LSTNetModel
from results_aggregation import corr
# from nsga_net.models.micro_genotypes import NetflowMultiHandover
from EEGNAS.model_generation.custom_modules import AveragingEnsemble, BasicEnsemble
from statistical_anomaly_detection.src.pytorch_lstm import MultivariateLSTM, MultivariateMultistepLSTM, MultivariateParallelMultistepLSTM, LSTMMulticlassClassification
# from nsga_net.models.micro_models import NetworkCIFAR
# from nsga_net.search.micro_encoding import convert, decode
from fold_utils import get_fold_idxs, get_data_by_balanced_folds, create_classification_report, \
    get_start_point_by_start_hour, average_classification_reports, get_classification_curves, prepare_y_for_classification, split_train_test_by_date
from EEGNAS.utilities.config_utils import get_multiple_values
from EEGNAS.utilities.misc import get_exp_id, exit_handler
from EEGNAS.utilities.report_generation import add_params_to_name
from EEGNAS.utilities.data_utils import unison_shuffled_copies
from EEGNAS.utilities.data_utils import load_values_from_config
from datetime import datetime
import torch
from sacred.observers import MongoObserver
from sklearn.model_selection import train_test_split
from EEGNAS import global_vars
from EEGNAS_experiment import get_normal_settings
from EEGNAS.data.netflow.netflow_data_utils import get_whole_netflow_data, preprocess_netflow_data,\
    get_netflow_threshold, get_moving_threshold, get_netflow_handovers
from EEGNAS.data_preprocessing import get_dataset, makeDummySignalTargets
from EEGNAS.evolution.nn_training import NN_Trainer
from EEGNAS.utilities.config_utils import set_params_by_dataset, get_configurations, set_gpu
from EEGNAS.utilities.data_utils import calc_regression_accuracy, aggregate_accuracies
from EEGNAS.utilities.misc import concat_train_val_sets, create_folder, unify_dataset, reset_model_weights
import torch.nn.functional as F
from sacred import Experiment
import numpy as np
import logging
import sys
import pandas as pd

ex = Experiment()
AS_NAMES = {20940: 'akamai', 16509: 'amazon', 6185: 'apple', 15133: 'edgecast', 32934: 'facebook', 15169: 'google',
            3356: 'level3', 202818: 'level3_cdn', 22822: 'limelight_networks', 2906: 'netflix', 32590: 'valve'}
DF_COLS = ['exp_id', 'training_as', 'tested_as', 'handover', 'training_hours', 'start_date', 'end_date', 'prediction_time', 'predicted_hours', 'threshold(stds)',
           'moving_threshold', 'train/test', 'support_0', 'support_1', 'auc', 'auc_precision_recall', 'fpr', 'tpr', 'threshold', 'threshold_ratio', 'rrse', 'rae',
           'corr', 'omniboard_id', 'evaluator', 'model', 'weighted_avg', 'max_epochs', 'early_stopping', 'xgboost_weight', 'eegnas_iteration', 'inheritance_type', 'cnn_ensemble_size', 'iteration', 'permidx', 'per_handover']
ALL_CLASSIF_REPORTS = []
ALL_CLASSIF_REPORTS_BY_FOLD = []
NETFLOW_THRESHOLD_STDS = {32934: 2, 16509: 2, 6185: 2, 3356: 2, 15169: 0.5, 20940: 2,
                          15133: 1, 22822: 1, 32590: 2, 2906: 2, 46489: 2, 202818: 2, 65013: 2, 16276: 2}

def set_random_seed(permidx):
    if permidx:
        random.seed(permidx)


def get_model_filename_kfold(type, fold_idx):
    inheritance_str, skip_cnn_str, unique_test_str, drop_others_str, top_pni_str, test_handover_str, as_str, samelocs_str, evaluator_str, seed_str, handover_str, interpolation_str = '', '', '', '', '', '', '', '', '', '', '', ''
    if global_vars.get('interpolate_netflow'):
        interpolation_str = '_interp'
    if global_vars.get('top_handovers'):
        handover_str = f'_top{global_vars.get("top_handovers")}'
    if global_vars.get('random_ho_permutations') and global_vars.get('permidx'):
        seed_str = f'_permidx_{global_vars.get("permidx")}'
    if set(global_vars.get('evaluator')) != set(["cnn", "rnn", "LSTNet", "nsga"]):
        if isinstance(global_vars.get('evaluator'), list):
            evaluator_str = f'_{"_".join(global_vars.get("evaluator"))}'
        else:
            evaluator_str = global_vars.get('evaluator')
    if global_vars.get('same_handover_locations'):
        samelocs_str = '_samelocs'
        if global_vars.get('test_handover_locs'):
            test_handover_str = '_testlocs'
    if global_vars.get('netflow_subfolder') == 'top_99':
        as_str = f'top{len(global_vars.get("autonomous_systems"))}'
    else:
        as_str = global_vars.get('autonomous_systems')
    if global_vars.get('top_pni'):
        top_pni_str = '_top_pni'
    if not global_vars.get('netflow_drop_others'):
        drop_others_str = '_others'
    if global_vars.get('unique_test_model'):
        unique_test_str = f'_unq_{global_vars.get("as_to_test")}'
    if not global_vars.get('skip_cnn_training') and 'cnn' in list(global_vars.get('evaluator')):
        skip_cnn_str = '_noskipcnn'
    if global_vars.get('eegnas_inheritance_type'):
        inheritance_str = f'_{global_vars.get("eegnas_inheritance_type")}'

    return f"{type}/{global_vars.get('dataset')}_{global_vars.get('date_range')}_{as_str}" \
            f"_{global_vars.get('per_handover_prediction')}_" \
            f"{global_vars.get('final_max_epochs')}_{global_vars.get('data_augmentation')}_" \
            f"{global_vars.get('n_folds')}_{fold_idx}_{global_vars.get('iteration')}{interpolation_str}" \
            f"{handover_str}{seed_str}{evaluator_str}{samelocs_str}{test_handover_str}{top_pni_str}{drop_others_str}{unique_test_str}{skip_cnn_str}{inheritance_str}.th"


def get_actual_predicted_from_df(df, handover):
    predicted = df[f'{handover}_{global_vars.get("steps_ahead")}_steps_ahead_pred'].values
    predicted = predicted[~np.isnan(predicted)]
    predicted = np.max(predicted.reshape(-1, global_vars.get('steps_ahead')), axis=1)
    actual = df[f'{handover}_{global_vars.get("steps_ahead")}_steps_ahead_real'].values
    actual = actual[~np.isnan(actual)]
    actual = np.max(actual.reshape(-1, global_vars.get('steps_ahead')), axis=1)
    return actual, predicted


@ex.capture
def create_report(df, segment, plot_curves, folder_name, regression_report, scaler, columns, _run):
    classif_reports = []
    if scaler is not None:
        test_df_copy = deepcopy(df)
        test_df_copy = apply_scaler_to_df(scaler, columns, test_df_copy)
    all_handovers = global_vars.get('netflow_handovers')
    for handover in all_handovers:
        if global_vars.get('plotting_problem') == 'regression' and np.sum(df[f'{handover}_5_steps_ahead_real'].values[:5]) == 0:
            continue
        if global_vars.get('plotting_problem') == 'regression':
            actual, predicted = get_actual_predicted_from_df(df, handover)
            if scaler is not None:
                actual_real, predicted_real = get_actual_predicted_from_df(test_df_copy, handover)
            if 'corr' not in regression_report.keys():
                regression_report_handover = create_regression_report(actual, predicted)
            if global_vars.get('static_threshold'):
                actual_pos = actual >= global_vars.get(f'netflow_threshold_{handover}')
                actual_neg = actual < global_vars.get(f'netflow_threshold_{handover}')
                actual[actual_pos] = 1
                actual[actual_neg] = 0
        else:
            actual = df[f'{handover}_5_steps_ahead_real'].values
            predicted = df[f'{handover}_5_steps_ahead_pred'].values
        classif_report = {'exp_id': global_vars.get('prediction_exp_name').split('_')[1], 'training_as': global_vars.get('autonomous_systems'), 'tested_as': global_vars.get('as_to_test'), 'handover': handover, 'training_hours': global_vars.get('input_height'),
                          'start_date': global_vars.get('date_range').split('-')[0], 'end_date': global_vars.get('date_range').split('-')[1], 'prediction_time': global_vars.get('start_hour'),
                          'predicted_hours': f"{global_vars.get('start_hour') + global_vars.get('prediction_buffer')}-{global_vars.get('start_hour') + global_vars.get('prediction_buffer') + global_vars.get('steps_ahead') - 1}",
                          'threshold(stds)': global_vars.get(f'netflow_threshold_stds_{handover}'), 'moving_threshold': global_vars.get('moving_threshold'), 'train/test': segment, 'evaluator': global_vars.get('evaluator'),
                          'model': '', 'weighted_avg': '', 'max_epochs': global_vars.get('final_max_epochs'), 'early_stopping': global_vars.get('final_max_increase_epochs'), 'eegnas_iteration': global_vars.get('eegnas_iteration'), 'inheritance_type': global_vars.get('eegnas_inheritance_type'), 'cnn_ensemble_size': global_vars.get('cnn_ensemble_size'), 'iteration': global_vars.get('iteration'),
                          'permidx': global_vars.get('permidx'), 'per_handover': global_vars.get('per_handover_prediction')}
        if global_vars.get('evaluator') == 'cnn':
            classif_report['model'] = global_vars.get('models_dir').split('_')[0]
        if type(global_vars.get('evaluator')) == list:
            classif_report['weighted_avg'] = not(global_vars.get('true_ensemble_avg'))
        if global_vars.get('add_xgboost_to_ensemble'):
            classif_report['xgboost_weight'] = global_vars.get('xgboost_weight')
        else:
            classif_report['xgboost_weight'] = 0
        if global_vars.get('plotting_problem') == 'regression':
            classif_report = {**regression_report_handover, **create_classification_report(actual, predicted, global_vars.get(f'netflow_threshold_{handover}')), **classif_report}
        else:
            classif_report = {**create_classification_report(actual, predicted, global_vars.get(f'netflow_threshold_{handover}')), **classif_report}
        if global_vars.get('use_sacred'):
            classif_report['omniboard_id'] = _run._id
        else:
            classif_report['omniboard_id'] = -1
        try:
            if global_vars.get('static_threshold'):
                fpr_tpr, precision_recall = get_classification_curves(actual, predicted)
                stats = fpr_tpr[fpr_tpr.fpr <= 0.05].iloc[-1]
            else:
                curves = calc_tpr_fpr_precision_recall(actual, predicted)
                stats = curves[curves.fpr <= 0.05].iloc[-1]
            for stat_name, stat in stats[['fpr', 'tpr', 'threshold']].iteritems():
                classif_report[stat_name] = stat
            if global_vars.get('plotting_problem') == 'regression':
                classif_report['threshold_ratio'] = df[f'{handover}_threshold'].values[0] / classif_report['threshold']
        except IndexError as e:
            for stat_name in ['fpr', 'tpr', 'threshold']:
                classif_report[stat_name] = 'NaN'
            classif_report['threshold_ratio'] = 'NaN'

        if scaler is not None:
            real_overflows = actual_real - test_df_copy[f'{handover}_threshold'].values[0]
            predicted_overflows = predicted_real - test_df_copy[f'{handover}_threshold'].values[0]
            classif_report['real_overflow_avg'] = np.average(real_overflows[real_overflows > 0])
            classif_report['predicted_overflow_avg'] = np.average(predicted_overflows[predicted_overflows > 0])
            if handover == 'sum' and global_vars.get('per_handover_prediction'):
                all_real_avgs = np.argsort([clf['real_overflow_avg'] for clf in classif_reports])
                all_pred_avgs = np.argsort([clf['predicted_overflow_avg'] for clf in classif_reports])
                correlation = corr(all_real_avgs, all_pred_avgs)
                classif_report['ranking_correlation'] = correlation
            else:
                classif_report['ranking_correlation'] = 0

        if plot_curves:
            if global_vars.get('static_threshold'):
                curve_imgs = plot_classification_curves(fpr_tpr.fpr, fpr_tpr.tpr, precision_recall.precision, precision_recall.recall, folder_name, handover)
            else:
                curve_imgs = plot_classification_curves(curves.fpr, curves.tpr, curves.precision,
                                                        curves.recall, folder_name, handover)
            if global_vars.get('use_sacred'):
                for img in curve_imgs:
                    ex.add_artifact(img)
            if global_vars.get('static_threshold'):
                fpr_tpr.to_csv(f'{folder_name}/fpr_tpr_{handover}.csv')
                precision_recall.to_csv(f'{folder_name}/precision_recall_{handover}.csv')
            else:
                curves.to_csv(f'{folder_name}/curves_{handover}.csv')

        classif_reports.append(classif_report)
    return classif_reports


@ex.capture
def classification_report(df, fold_idx, threshold, aggregate, segment, _run):
    actual, predicted = df['real'].values, df['pred'].values
    classif_report = {'training_as': global_vars.get('autonomous_systems'), 'tested_as': global_vars.get('as_to_test'),
                      'fold': fold_idx, 'date_range': global_vars.get('date_range'), 'prediction_time': global_vars.get('start_hour'),
                      'predicted_hours': f"{global_vars.get('start_hour') + global_vars.get('prediction_buffer')}-{global_vars.get('start_hour') + global_vars.get('prediction_buffer') + global_vars.get('steps_ahead') - 1}",
                      'threshold(stds)': threshold, 'moving/static': global_vars.get('moving_threshold'), 'train/test': segment, 'model': global_vars.get('models_dir').split('_')[0]}
    classif_report = {**create_classification_report(actual, predicted), **classif_report}
    classif_report['omniboard_id'] = _run._id
    return classif_report


def df_to_sacred(df, segment, fold=''):
    if fold != '':
        fold = f'_fold_{fold}'
    for idx in range(len(df)):
        for col in df.columns:
            ex.log_scalar(f'{col}_{segment}{fold}', df.iloc[idx][col], idx+1)


def write_dict(dict, filename):
    with open(filename, 'w') as f:
        all_keys = []
        for _, inner_dict in sorted(dict.items()):
            for K, _ in sorted(inner_dict.items()):
                all_keys.append(K)
        for K in all_keys:
            f.write(f"{K}\t{global_vars.get(K)}\n")


def get_netflow_test_data_by_indices(train_index, test_index, problem):
    prev_steps_ahead = global_vars.get('steps_ahead')
    prev_problem = global_vars.get('problem')
    global_vars.set('problem', problem)
    global_vars.set('steps_ahead', 24)
    dataset = get_dataset('all')
    if train_index is not None:
        data = unify_dataset(dataset)
        X_test = data.X[test_index]
        y_test = data.y[test_index]
    else:
        X_test = dataset['test'].X
        y_test = dataset['test'].y
    global_vars.set('problem', prev_problem)
    global_vars.set('steps_ahead', prev_steps_ahead)
    return X_test, y_test


def regression_to_accuracy(predicted, real, aggregate, threshold, moving_threshold):
    actual, predicted = calc_regression_accuracy(predicted, real, threshold, moving_threshold)
    if aggregate:
        actual, predicted = aggregate_accuracies((actual, predicted), global_vars.get('steps_ahead'))
    return actual, predicted


def get_dataset_from_folds(fold_samples):
    X_train, X_test = fold_samples['X_train'], fold_samples['X_test']
    y_train, y_test = fold_samples['y_train'], fold_samples['y_test']
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'),
                                                      shuffle=False)
    # X_train, y_train = unison_shuffled_copies(X_train, y_train)
    dataset = {}
    dataset['train'], dataset['valid'], dataset['test'] = \
        makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return dataset


def get_pretrained_model(filename):
    if global_vars.get('overwrite_models'):
        return None
    if os.path.exists(filename):
        print(f'loaded {filename}')
        return torch.load(filename)
    try:
        filename_split = filename.split('_')
        ass = ast.literal_eval(filename_split[4])
        del filename_split[4]
        for file in [f'kfold_models/{f}' for f in os.listdir('kfold_models')]:
            file_split = file.split('_')
            curr_ass = ast.literal_eval(file_split[4])
            del file_split[4]
            if set(ass) == set(curr_ass) and file_split == filename_split:
                model = torch.load(file)
                print(f'loaded {file}')
                return model
    except Exception:
        return None
    return None


def tsfresh_features(X_train):
    feature_templates = ['__large_standard_deviation__r_0.25',
                         '__symmetry_looking__r_0.1',
                         '__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
                         '__median',
                         '__quantile__q_0.4']
    all_features = [f'{i}{st}' for i in range(40) for st in feature_templates]
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(all_features)
    reshaped_examples = np.vstack([np.swapaxes(X_train[i], 0, 1) for i in range(len(X_train))])
    time = np.hstack([list(range(240)) for i in range(len(X_train))])
    id = np.hstack([[j + 1 for i in range(240)] for j in range(len(X_train))])
    df = pd.DataFrame(data=reshaped_examples)
    df['id'] = id
    df['time'] = time
    extracted_features = extract_features(df, column_id="id", column_sort="time",
                                          kind_to_fc_parameters=kind_to_fc_parameters)
    X = extracted_features.to_numpy()
    return X


def train_sklearn_for_netflow(model, dataset):
    # X = tsfresh_features(dataset['train'].X)
    X = dataset['train'].X.reshape(dataset['train'].X.shape[0], -1)
    model.fit(X, dataset['train'].y)


def train_autokeras_for_netflow(model, dataset):
    X = dataset['train'].X.reshape(dataset['train'].X.shape[0], dataset['train'].X.shape[2], dataset['train'].X.shape[1])
    model.fit(X, dataset['train'].y)


def train_model_for_netflow(model, dataset):
    print(f'start training model: {type(model).__name__}')
    if type(model).__name__ in ['RandomForestRegressor', 'MultiOutputRegressor']:
        train_sklearn_for_netflow(model, dataset)
    elif 'ImageRegressor' == type(model).__name__:
        train_autokeras_for_netflow(model, dataset)
    elif 'Ensemble' in type(model).__name__:
        for mod in model.models:
            mod.cuda()
            mod.train()
            train_model_for_netflow(mod, dataset)
        if type(model) == BasicEnsemble:
            model.freeze_child_models(False)
            global_vars.get('evaluator_obj').train_model(model, dataset, final_evaluation=True)
            model.freeze_child_models(True)
    else:
        if global_vars.get('dataset') in ['solar', 'electricity', 'exchange_rate'] and global_vars.get('training_method') == 'LSTNet':
            data = Data_utility(f'../EEGNAS/EEGNAS/data/MTS_benchmarks/'
                                f'{global_vars.get("dataset")}.txt', 0.6, 0.2, device='cuda', window=24 * 7, horizon=12)
            optim = Optim(model.parameters(), 'adam', 0.001, 10.)
            criterion = torch.nn.MSELoss(size_average=False)
            MTS_train(model, data, criterion, optim, 100, 128)
        else:
            if type(model).__name__ == 'Sequential' and global_vars.get('skip_cnn_training'):
                print('skipping CNN training')
                return
            global_vars.get('evaluator_obj').train_model(model, dataset, final_evaluation=True)


def apply_scaler(x, min, scale):
    x -= min
    x /= scale
    return x


def apply_scaler_to_df(scaler, columns, df):
    if global_vars.get('per_handover_prediction'):
        for col in columns:
            for df_col in df.columns:
                if str(col) in df_col:
                    df[df_col] = df[df_col].apply(apply_scaler, args=(scaler.min_[columns.index(col)], scaler.scale_[columns.index(col)]))
    else:
        df = df.apply(apply_scaler, args=(scaler.min_[-1], scaler.scale_[-1]))
    return df


def apply_scaler_to_numpy(scaler, columns, y_all):
    y_all_copy = y_all.copy()
    for col in range(y_all_copy.shape[1]):
        # if global_vars.get('netflow_handovers')[col] in columns and\
        #         global_vars.get('netflow_handovers')[col] != global_vars.get('as_to_test') and\
        #         global_vars.get('netflow_handovers')[col] != 'sum':
        if global_vars.get('netflow_handovers')[col] in columns and \
                    global_vars.get('netflow_handovers')[col] != global_vars.get('as_to_test'):
            min = scaler.min_[columns.index(global_vars.get('netflow_handovers')[col])]
            scale = scaler.scale_[columns.index(global_vars.get('netflow_handovers')[col])]
            for sample in range(y_all_copy.shape[0]):
                y_all_copy[sample, col] -= min
                y_all_copy[sample, col] /= scale
        else:
            y_all_copy[:, col, :] = 0
    return y_all_copy


def apply_scaler_to_vector(scaler, columns, col_name, y):
    scale = scaler.scale_[columns.index(col_name)]
    min = scaler.min_[columns.index(col_name)]
    for idx in range(y.shape[0]):
        y[idx] *= scale
        y[idx] += min
    return y


def export_kfold_data(folder_name):
    fold_idxs = get_fold_idxs(global_vars.get('as_to_test'))
    folds_target = get_data_by_balanced_folds \
        ([global_vars.get('as_to_test')], fold_idxs)
    folds = get_data_by_balanced_folds(global_vars.get('autonomous_systems'), fold_idxs)
    if global_vars.get('highest_handover_overflow'):
        folds, folds_target = prepare_y_for_classification(folds, folds_target)
    for fold_idx, fold_samples in folds.items():
        dataset = get_dataset_from_folds(fold_samples)
        concat_train_val_sets(dataset)
        per_handover_str = ''
        samelocs_str = ''
        classification_str = ''
        if global_vars.get('per_handover_prediction'):
            per_handover_str = '_per_handover'
        if global_vars.get('same_handover_locations'):
            samelocs_str = '_samelocs'
        if global_vars.get('problem') == 'classification':
            classification_str = '_classification'
        full_folder_name = f'{folder_name}/{global_vars.get("dataset")}{per_handover_str}{samelocs_str}{classification_str}_fold_{fold_idx}'
        create_folder(full_folder_name)
        np.save(f'{full_folder_name}/X_train.npy', dataset['train'].X)
        np.save(f'{full_folder_name}/y_train.npy', dataset['train'].y)
        np.save(f'{full_folder_name}/X_test.npy', dataset['test'].X)
        np.save(f'{full_folder_name}/y_test.npy', dataset['test'].y)


def export_train_test_data(folder_name):
    dataset = get_dataset('all')
    if global_vars.get('train_test_split_date'):
        data = unify_dataset(dataset)
        _, _, datetimes_X, datetimes_Y = preprocess_netflow_data(
            [f'../EEGNAS/EEGNAS/data/netflow/top_10_corona/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv'],
            global_vars.get('input_height'), global_vars.get('steps_ahead'),
            global_vars.get('jumps'), global_vars.get('prediction_buffer'))
        dataset, datetimes = split_train_test_by_date(data, datetimes_X, datetimes_Y)
    per_handover_str = ''
    if global_vars.get('per_handover_prediction'):
        per_handover_str = '_per_handover'
    np.save(f'{folder_name}/X_train{per_handover_str}_corona.npy', dataset['train'].X)
    np.save(f'{folder_name}/y_train{per_handover_str}_corona.npy', dataset['train'].y)
    np.save(f'{folder_name}/X_test{per_handover_str}_corona.npy', dataset['test'].X)
    np.save(f'{folder_name}/y_test{per_handover_str}_corona.npy', dataset['test'].y)


def kfold_exp(folder_name):
    subfolder_str = ''
    if global_vars.get('netflow_subfolder'):
        subfolder_str = f"/{global_vars.get('netflow_subfolder')}"
    set_random_seed(global_vars.get('permidx'))
    fold_idxs = get_fold_idxs(global_vars.get('as_to_test'))
    required_num_samples = len(fold_idxs[global_vars.get('n_folds')-1]['train_idxs']) + len(fold_idxs[global_vars.get('n_folds')-1]['test_idxs'])
    folds_target = get_data_by_balanced_folds \
        ([global_vars.get('as_to_test')], fold_idxs, required_num_samples)
    if global_vars.get('netflow_subfolder') == 'top_99':
        if os.path.exists(f'top_{len(global_vars.get("autonomous_systems"))}_data.pkl'):
            with open(f'top_{len(global_vars.get("autonomous_systems"))}_data.pkl', "rb") as f:
                folds = pickle.load(f)
        else:
            folds = get_data_by_balanced_folds(global_vars.get('autonomous_systems'), fold_idxs, required_num_samples)
    else:
        folds = get_data_by_balanced_folds(global_vars.get('autonomous_systems'), fold_idxs, required_num_samples)
    if global_vars.get('highest_handover_overflow'):
        folds, folds_target = prepare_y_for_classification(folds, folds_target)

    test_dfs = []
    test_reports = []
    for fold_idx, fold_samples in folds.items():
        print(f'Fold {fold_idx} started\n----------------------------------------')
        dataset = get_dataset_from_folds(fold_samples)
        dataset_target = get_dataset_from_folds(folds_target[fold_idx])

        if global_vars.get('k_fold_time'):
            filename = get_model_filename_kfold('kfold_models', fold_idx)
            model = get_pretrained_model(filename)
            if model is None:
                model = get_evaluator(global_vars.get('evaluator'), fold_idx)
                if global_vars.get('data_augmentation'):
                    dataset = augment_dataset(dataset)
                train_model_for_netflow(model, dataset)
                torch.save(model, filename)
        else:
            model = get_evaluator(global_vars.get('evaluator'))
            train_model_for_netflow(model, dataset)
        if global_vars.get('add_xgboost_to_ensemble'):
            xgboost_filename = get_model_filename_kfold('xgboost_models', fold_idx)
            try:
                xgboost = joblib.load(xgboost_filename)
            except FileNotFoundError:
                xgboost = get_evaluator('xgboost')
                train_model_for_netflow(xgboost, dataset)
                joblib.dump(xgboost, xgboost_filename)
        else:
            xgboost = None
        concat_train_val_sets(dataset_target)
        if global_vars.get('only_train_models'):
            continue
        for segment in dataset_target.keys():
            df = pd.DataFrame()
            df, reports = globals()[f'export_{global_vars.get("plotting_problem")}_results'](df, dataset_target[segment], segment,
                                                                      model, folder_name, fold_idxs=fold_idxs[fold_idx], xgboost=xgboost)
            if segment == 'test':
                test_dfs.append(df)
                for rep in reports:
                    rep['fold'] = fold_idx
                ALL_CLASSIF_REPORTS_BY_FOLD.append(pd.DataFrame(reports))
    if global_vars.get('only_train_models'):
        return
    all_test_df = pd.concat(test_dfs)
    if 'time' in all_test_df.columns:
        all_test_df.sort_values('time', inplace=True)
    scaler, columns = get_netflow_minmax_scaler(f'../EEGNAS/data/{global_vars.get("dataset")}{subfolder_str}'
                                       f'/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv')
    if not global_vars.get('static_threshold'):
        all_test_df = apply_scaler_to_df(scaler, columns, all_test_df)
    all_filename = f'{folder_name}/{global_vars.get("input_height")}_' \
               f'{global_vars.get("steps_ahead")}_ahead_{segment}_all.csv'
    if global_vars.get('plotting_problem') == 'regression':
        classif_reports = create_report(all_test_df, 'test', True, folder_name, regression_report={}, scaler=scaler, columns=columns)
    else:
        classif_reports = create_report(all_test_df, 'test', True, folder_name, regression_report={}, scaler=None, columns=None)
    auc_result = mean([clf['auc'] for clf in classif_reports])
    classification_filename = f'{folder_name}/{global_vars.get("prediction_exp_name")}_complete_test_report.csv'
    if global_vars.get('plotting_problem') == 'classification':
        for key in DF_COLS + ['real_overflow_avg', 'predicted_overflow_avg', 'ranking_correlation']:
            for classif_report in classif_reports:
                if key not in classif_report.keys():
                    classif_report[key] = ''
    classif_df = pd.DataFrame(classif_reports)[DF_COLS + ['real_overflow_avg', 'predicted_overflow_avg']]
    classif_df.to_csv(classification_filename, index=False)
    ALL_CLASSIF_REPORTS.extend([pd.DataFrame([classif_report])[DF_COLS + ['real_overflow_avg', 'predicted_overflow_avg', 'ranking_correlation']] for classif_report in classif_reports])
    if global_vars.get('use_sacred'):
        ex.add_artifact(classification_filename)
    if global_vars.get('moving_threshold'):
        all_test_df[f'threshold_{threshold}'] = global_vars.get(f'moving_threshold_{threshold}')[
                               len(global_vars.get(f'moving_threshold_{threshold}')) - len(all_test_df):]
    if global_vars.get('plotting_problem') == 'regression':
        if global_vars.get('static_threshold'):
            all_test_df = apply_scaler_to_df(scaler, columns, all_test_df)
        all_test_df.to_csv(all_filename)
        all_test_df_binned = all_test_df.resample('D').max()
        all_test_df_binned.to_csv(f'{all_filename[:-4]}_daily_bins.csv')
    if global_vars.get('use_sacred') and global_vars.get('plot_to_sacred'):
        df_to_sacred(all_test_df, 'test', 'all')
    return auc_result


def no_kfold_exp(folder_name):
    subfolder_str = ''
    if global_vars.get('netflow_subfolder'):
        subfolder_str = f"/{global_vars.get('netflow_subfolder')}"
    scaler, columns = get_netflow_minmax_scaler(f'../EEGNAS/EEGNAS/data/netflow{subfolder_str}'
                                       f'/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv')
    dataset = get_dataset('all')
    autonomous_systems = global_vars.get('autonomous_systems')
    if global_vars.get('dataset') != 'netflow_asflow':
        dataset_target = dataset
    else:
        global_vars.set("autonomous_systems", [global_vars.get('as_to_test')])
        dataset_target = get_dataset('all')
        global_vars.set("autonomous_systems", autonomous_systems)
    datetimes = {'train': None, 'test': None}
    if global_vars.get('train_test_split_date'):
        data = unify_dataset(dataset)
        data_target = unify_dataset(dataset_target)
        _, _, datetimes_X, datetimes_Y = preprocess_netflow_data(
            [f'../EEGNAS/EEGNAS/data/netflow{subfolder_str}/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv'],
            global_vars.get('input_height'), global_vars.get('steps_ahead'),
            global_vars.get('jumps'), global_vars.get('prediction_buffer'))
        dataset, datetimes = split_train_test_by_date(data, datetimes_X, datetimes_Y)
        dataset_target, _ = split_train_test_by_date(data_target, datetimes_X, datetimes_Y)
    filename = f"train_test_models/{global_vars.get('train_test_split_date')}_{global_vars.get('autonomous_systems')}" \
                                    f"_{global_vars.get('per_handover_prediction')}_{global_vars.get('iteration')}" \
                                    f"_{global_vars.get('final_max_epochs')}_{global_vars.get('data_augmentation')}.th"
    model = get_pretrained_model(filename)
    if model is None:
        model = get_evaluator(global_vars.get('evaluator'))
        if global_vars.get('data_augmentation'):
            dataset = augment_dataset(dataset)
        train_model_for_netflow(model, dataset)
        torch.save(model, filename)
    concat_train_val_sets(dataset_target)
    test_df = None
    for segment in dataset_target.keys():
        df = pd.DataFrame()
        if segment == 'test':
            df, report = globals()[f'export_{global_vars.get("plotting_problem")}_results'](df, dataset_target[segment],
                                                                    segment, model, folder_name, datetimes[segment],
                                                                    scaler=scaler, columns=columns, plot_curves=True)
        else:
            df, report = globals()[f'export_{global_vars.get("plotting_problem")}_results'](df, dataset_target[segment],
                                                                    segment, model, folder_name, datetimes[segment])
        if segment == 'test':
            classif_reports = report
            test_df = df.copy(deep=True)
    if global_vars.get('dataset') == 'netflow_asflow':
        test_df.sort_values('time', inplace=True)
        scaler, columns = get_netflow_minmax_scaler(f'../EEGNAS/EEGNAS/data/netflow{subfolder_str}'
                                       f'/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv')
        if not global_vars.get('static_threshold'):
            test_df = apply_scaler_to_df(scaler, columns, test_df)
    all_filename = f'{folder_name}/{global_vars.get("input_height")}_' \
                   f'{global_vars.get("steps_ahead")}_ahead_{segment}_all.csv'
    auc_result = mean([clf['auc'] for clf in classif_reports])
    classification_filename = f'{folder_name}/{global_vars.get("prediction_exp_name")}_complete_test_report.csv'
    pd.DataFrame(classif_reports)[DF_COLS].to_csv(classification_filename, index=False)
    ALL_CLASSIF_REPORTS.extend([pd.DataFrame([classif_report])[DF_COLS] for classif_report in classif_reports])
    if global_vars.get('use_sacred'):
        ex.add_artifact(classification_filename)
    if global_vars.get('moving_threshold'):
        test_df[f'threshold_{threshold}'] = global_vars.get(f'moving_threshold_{threshold}')[
                                                len(global_vars.get(f'moving_threshold_{threshold}')) - len(
                                                    test_df):]
    if global_vars.get('static_threshold'):
        test_df = apply_scaler_to_df(scaler, columns, test_df)
    test_df.to_csv(all_filename)
    test_df_binned = test_df.resample('D').max()
    test_df_binned.to_csv(f'{all_filename[:-4]}_daily_bins.csv')
    if global_vars.get('dataset') == 'netflow_asflow' and global_vars.get('static_threshold'):
        test_df = apply_scaler_to_df(scaler, columns, test_df)
    if global_vars.get('plot_to_sacred'):
        df_to_sacred(test_df, 'test', 'all')
    return auc_result


def get_datetimes_by_fold_and_segment(fold_idxs, segment):
    netflow_subfolder_str = ''
    if global_vars.get('netflow_subfolder'):
        netflow_subfolder_str = f'{global_vars.get("netflow_subfolder")}/'
    _, _, datetimes_X, datetimes_Y = preprocess_netflow_data(
        [f'../EEGNAS/data/{global_vars.get("dataset")}/{netflow_subfolder_str}{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv'],
        global_vars.get('input_height'), global_vars.get('steps_ahead'),
        global_vars.get('jumps'), global_vars.get('prediction_buffer'))

    datetimes = {}
    if fold_idxs is None:
        _, _, datetimes['train'], datetimes['test'] = train_test_split(datetimes_X, datetimes_Y,
                                                                       test_size=global_vars.get('valid_set_fraction'),
                                                                       shuffle=False)
    else:
        datetimes['train'], datetimes['test'] = datetimes_Y[fold_idxs['train_idxs']], datetimes_Y[fold_idxs['test_idxs']]
    return datetimes[segment]


def export_regression_results(df, data, segment, model, folder_name, datetimes=None, fold_idxs=None, xgboost=None, scaler=None, columns=None, plot_curves=False):
    netflow_subfolder_str = ''
    if global_vars.get('netflow_subfolder'):
        netflow_subfolder_str = f'{global_vars.get("netflow_subfolder")}/'
    if datetimes is None:
        datetimes = get_datetimes_by_fold_and_segment(fold_idxs, segment)
    if data.X.ndim == 3:
        data.X = data.X[:, :, :, None]
    if global_vars.get('evaluator') in ['randomforest', 'xgboost']:
        y_pred_all = model.predict(torch.tensor(data.X).reshape(data.X.shape[0], -1))
        # y_pred_all = model.predict(torch.tensor(tsfresh_features(data.X.squeeze())))
    elif global_vars.get('evaluator') == 'autokeras':
        y_pred_all = model.predict(data.X.reshape(data.X.shape[0], data.X.shape[2], data.X.shape[1]))
    else:
        model.eval()
        model.cpu()
        if 'Ensemble' in type(model).__name__:
            for mod in model.models:
                if 'Ensemble' in type(mod).__name__:
                    for inner_mod in mod.models:
                        inner_mod.cpu()
                        inner_mod.eval()
                mod.cpu()
                mod.eval()
        gc.collect()
        y_pred_all = model(torch.tensor(data.X).float()).cpu().detach().numpy()
    if xgboost:
        y_pred_all = (1-global_vars.get('xgboost_weight')) * y_pred_all + global_vars.get('xgboost_weight')\
                     * xgboost.predict(torch.tensor(data.X).reshape(data.X.shape[0], -1))
    if len(global_vars.get('netflow_handovers')) < global_vars.get('max_handovers'):
        y_pred_all = y_pred_all[:, -len(global_vars.get('netflow_handovers')) * global_vars.get('steps_ahead'):]
        y_real_all = data.y[:, -len(global_vars.get('netflow_handovers')) * global_vars.get('steps_ahead'):]
        current_handovers = len(global_vars.get('netflow_handovers'))
        y_real_all = y_real_all.reshape(data.y.shape[0], current_handovers, -1)
    else:
        current_handovers = global_vars.get('max_handovers')
        y_real_all = data.y.reshape(data.y.shape[0], current_handovers, -1)
    y_pred_all = y_pred_all.reshape(y_pred_all.shape[0], current_handovers, -1)

    for handover_idx, handover in enumerate(global_vars.get('netflow_handovers')[:current_handovers]):
        y_pred = y_pred_all[:, handover_idx, :]
        y_real = y_real_all[:, handover_idx, :]
        y_pred = np.array([np.concatenate([y, np.array([np.nan for i in range(int(global_vars.get('jumps') -
                                                    global_vars.get('steps_ahead')))])], axis=0) for y in y_pred])
        y_real = np.array([np.concatenate([y, np.array([np.nan for i in range(int(global_vars.get('jumps') -
                                                    global_vars.get('steps_ahead')))])], axis=0) for y in y_real])

        y_pred = np.concatenate([yi for yi in y_pred], axis=0).clip(min=0)
        y_real = np.concatenate([yi for yi in y_real], axis=0)
        df[f'{handover}_{global_vars.get("steps_ahead")}_steps_ahead_real'] = y_real
        df[f'{handover}_{global_vars.get("steps_ahead")}_steps_ahead_pred'] = y_pred
        df[f'{handover}_{global_vars.get("steps_ahead")}_steps_ahead_error'] = y_pred - y_real
        df[f'{handover}_threshold'] = [global_vars.get(f'netflow_threshold_{handover}') for i in range(len(df))]

    if len(global_vars.get('netflow_handovers')) > 1:
        scaler, columns = get_netflow_minmax_scaler(f'../EEGNAS/data/{global_vars.get("dataset")}/{netflow_subfolder_str}'
                                                    f'{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv')
        sum_columns = columns
        y_pred_all_scaled = apply_scaler_to_numpy(scaler, sum_columns, y_pred_all)
        y_real_all_scaled = apply_scaler_to_numpy(scaler, sum_columns, y_real_all)
        y_pred = np.sum(y_pred_all_scaled, axis=1)
        y_real = np.sum(y_real_all_scaled, axis=1)
        y_pred = apply_scaler_to_vector(scaler, columns, 'sum', y_pred)
        y_real = apply_scaler_to_vector(scaler, columns, 'sum', y_real)
        y_pred = np.array([np.concatenate([y, np.array([np.nan for i in range(int(global_vars.get('jumps') -
                                                    global_vars.get('steps_ahead')))])],axis=0) for y in y_pred])
        y_real = np.array([np.concatenate([y, np.array([np.nan for i in range(int(global_vars.get('jumps') -
                                                    global_vars.get('steps_ahead')))])],axis=0) for y in y_real])
        y_pred = np.concatenate([yi for yi in y_pred], axis=0).clip(min=0)
        y_real = np.concatenate([yi for yi in y_real], axis=0)
        df[f'sum_{global_vars.get("steps_ahead")}_steps_ahead_real'] = y_real
        df[f'sum_{global_vars.get("steps_ahead")}_steps_ahead_pred'] = y_pred
        df[f'sum_{global_vars.get("steps_ahead")}_steps_ahead_error'] = y_pred - y_real
        df[f'sum_threshold'] = [global_vars.get(f'netflow_threshold_sum') for i in range(len(df))]

    if global_vars.get('dataset') in ['netflow_asflow', 'overflow_prediction']:
        y_datetimes = np.array([np.concatenate([dt, pd.date_range(start=dt[-1] + np.timedelta64(1, 'h'),
                             periods=global_vars.get('jumps') - global_vars.get('steps_ahead'), freq='h')], axis=0)
                            for dt in datetimes])
        y_datetimes = np.concatenate([yi for yi in y_datetimes], axis=0)
        df['time'] = y_datetimes
        df.index = pd.to_datetime(df['time'])
        df = df.drop(columns=['time'])
        df.sort_values('time', inplace=True)
    regression_report = {}
    if global_vars.get('dataset') in ['exchange_rate', 'solar', 'electricity']:
        data = Data_utility(f'../EEGNAS/EEGNAS/data/MTS_benchmarks/'
                            f'{global_vars.get("dataset")}.txt', 0.6, 0.2, device='cpu', window=24 * 7, horizon=12)
        test_rse, test_rae, test_corr = MTS_evaluate(data, data.test[0], data.test[1], model, global_vars.get('batch_size'))
        regression_report['rrse'] = test_rse
        regression_report['rae'] = test_rae
        regression_report['corr'] = test_corr
    classif_reports = create_report(df, segment, plot_curves, folder_name, regression_report=regression_report, scaler=scaler, columns=columns)
    global_vars.get('classification_reports')[segment].extend(classif_reports)
    return df, classif_reports


def export_classification_results(df, data, segment, model, folder_name, fold_idx=None, fold_idxs=None, **kwargs):
    datetimes = get_datetimes_by_fold_and_segment(fold_idxs, segment)
    model.eval()
    model.cpu()
    if 'Ensemble' in type(model).__name__:
        for mod in model.models:
            if 'Ensemble' in type(mod).__name__:
                for inner_mod in mod.models:
                    inner_mod.cpu()
                    inner_mod.eval()
            mod.cpu()
            mod.eval()
    gc.collect()
    if data.X.ndim == 3:
        data.X = data.X[:, :, :, None]
    y_pred = model(torch.tensor(data.X).float()).cpu().detach().numpy()
    # create_classification_report(data.y, y_pred[:, 1], global_vars.get(f'netflow_threshold_sum'))
    df[f'sum_{global_vars.get("steps_ahead")}_steps_ahead_real'] = data.y
    df[f'sum_{global_vars.get("steps_ahead")}_steps_ahead_pred'] = y_pred[:, 1]
    df['time'] = datetimes[:, 0]
    classif_reports = create_report(df, segment, False, folder_name, regression_report={},
                                    scaler=None, columns=None)
    fold_str = ''
    if fold_idx is not None:
        fold_str = f'_fold_{fold_idx}'
    df.to_csv(f'{folder_name}/{global_vars.get("input_height")}_{fold_str}_classification.csv')
    # classification_filename = f'{folder_name}/{global_vars.get("input_height")}_' \
    #                           f'{global_vars.get("steps_ahead")}_ahead_{segment}{fold_str}_accuracy_threshold_{global_vars.get("num_std")}.csv'
    # classif_report[DF_COLS].to_csv(classification_filename)
    # global_vars.get('classification_reports')[segment].append(classif_reports)
    return df, classif_reports


def get_evaluator(evaluator_type, fold_idx=None):
    if global_vars.get('top_handovers'):
        channels = global_vars.get('top_handovers')
    elif global_vars.get('max_handovers'):
        channels = global_vars.get('max_handovers')
    elif global_vars.get('handovers'):
        channels = len(global_vars.get('handovers'))
    if global_vars.get('per_handover_prediction'):
        output_size = global_vars.get('steps_ahead') * channels
    else:
        output_size = global_vars.get('steps_ahead')
    if type(evaluator_type) == list:
        models = [get_evaluator(ev, fold_idx) for ev in global_vars.get('evaluator')]
        if not global_vars.get('true_ensemble_avg'):
            model = BasicEnsemble(models, output_size)
        else:
            model = AveragingEnsemble(models, global_vars.get('true_ensemble_avg'))
        model.cpu()
        return model
    if evaluator_type == 'cnn':
        if global_vars.get('cnn_ensemble'):
            all_population_files = os.listdir('eegnas_models')
            global_vars.set('current_fold_idx', fold_idx)
            pop_files = list(filter(filter_eegnas_population_files, all_population_files))
            assert len(pop_files) == 1
            pop_file = pop_files[0]
            model = create_ensemble_from_population_file(f'eegnas_models/{pop_file}', global_vars.get('cnn_ensemble_size'))
            if not global_vars.get('skip_cnn_training'):
                for mod in model.models:
                    reset_model_weights(mod)
            model.cpu()
        else:
            model = torch.load(
                f'../EEGNAS/EEGNAS/models/{global_vars.get("models_dir")}/{global_vars.get("model_file_name")}')
            reset_model_weights(model)
            model.cpu()
            load_values_from_config(f'../EEGNAS/EEGNAS/models/{global_vars.get("models_dir")}'
                                    f'/config_{global_vars.get("models_dir")}.ini',
                                    ['input_height', 'start_hour', 'start_point', 'date_range', 'prediction_buffer',
                                     'steps_ahead', 'jumps', 'normalize_netflow_data', 'per_handover_prediction'])
        return model
    elif evaluator_type == 'nsga':
        nsga_file = 'nsga_models/best_genome_normalized.pkl'
        if global_vars.get('top_handovers'):
            nsga_file = f'{nsga_file[:-4]}_top{global_vars.get("top_handovers")}.pkl'
        if global_vars.get('per_handover_prediction'):
            nsga_file = f'{nsga_file[:-4]}_per_handover.pkl'
        if global_vars.get('same_handover_locations'):
            nsga_file = f'{nsga_file[:-4]}_samelocs.pkl'
        if global_vars.get('handovers'):
            nsga_file = f'{nsga_file[:-4]}_handovers.pkl'
        if global_vars.get('netflow_subfolder') and 'corona' in global_vars.get('netflow_subfolder'):
            nsga_file = f'{nsga_file[:-4]}_corona.pkl'
        if fold_idx is not None:
            nsga_file = f'{nsga_file[:-4]}_fold{fold_idx}.pkl'
        with open(nsga_file, 'rb') as f:
            genotype = pickle.load(f)
            genotype = decode(convert(genotype))
            model = NetworkCIFAR(24, output_size, channels, 11, False, genotype)
        model.droprate = 0.0
        model.single_output = True
        return model
    elif evaluator_type == 'rnn':
        if global_vars.get('highest_handover_overflow'):
            model = LSTMMulticlassClassification(channels, 100, global_vars.get('batch_size'),
                                              global_vars.get('input_height'), num_layers=global_vars.get('lstm_layers'), eegnas=True)
        elif global_vars.get('per_handover_prediction'):
            model = MultivariateParallelMultistepLSTM(channels, 100, global_vars.get('batch_size'),
                                              global_vars.get('input_height'), num_layers=global_vars.get('lstm_layers'),
                                                      eegnas=True)

        else:
            model = MultivariateLSTM(channels, 100, global_vars.get('batch_size'),
                                 global_vars.get('input_height'), global_vars.get('steps_ahead'), num_layers=global_vars.get('lstm_layers'), eegnas=True)
        model.cpu()
        return model
    elif evaluator_type == 'LSTNet':
        model = LSTNetModel(channels, output_size, window=global_vars.get('input_height'))
        model.cpu()
        return model
    elif evaluator_type == 'MHANet':
        model = MHANetModel(channels, output_size)
        model.cpu()
        return model
    elif evaluator_type == 'WaveNet':
        model = WaveNet(input_channels=channels, output_channels=1, horizon=global_vars.get('steps_ahead'))
        model.cpu()
        return model
    elif evaluator_type == 'xgboost':
        model = MultiOutputRegressor(XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
            max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
            n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
            silent=True, subsample=1, tree_method='gpu_hist', gpu_id=0))
        return model
    elif evaluator_type == 'randomforest':
        model = RandomForestRegressor()
        return model
    elif evaluator_type == 'autokeras':
        return ak.ImageRegressor(max_trials=3)


@ex.main
def main():
    if global_vars.get('netflow_threshold_std') == 'best':
        global_vars.set('netflow_threshold_std', NETFLOW_THRESHOLD_STDS[global_vars.get('as_to_test')])
    if global_vars.get('per_handover_prediction'):
        if global_vars.get('handovers'):
            global_vars.set('netflow_handovers', global_vars.get('handovers') + ['sum'])
        else:
            global_vars.set('netflow_handovers', get_netflow_handovers(f'../EEGNAS/EEGNAS/data/netflow{subfolder_str}'
                             f'/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv'))
        netflow_threshs = {}
        for handover in global_vars.get('netflow_handovers'):
            try:
                thresh, stds = get_netflow_threshold(f'../EEGNAS/data/{global_vars.get("dataset")}'
                                                f'/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv',
                                                global_vars.get('netflow_threshold_std'), handover)
                global_vars.set(f'netflow_threshold_{handover}', thresh)
                global_vars.set(f'netflow_threshold_stds_{handover}', stds)
                netflow_threshs[str(handover)] = thresh
            except Exception:
                print(f'ignoring handover {handover} while fetching thresholds, not in the test AS')
        print(f'Thresholds for AS {global_vars.get("as_to_test")}: {netflow_threshs}')
    else:
        global_vars.set('netflow_handovers', ['sum'])
        thresh, stds = get_netflow_threshold(f'../EEGNAS/EEGNAS/data/netflow{subfolder_str}'
                         f'/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv', global_vars.get('netflow_threshold_std'))
        global_vars.set(f'netflow_threshold_sum', thresh)
        global_vars.set(f'netflow_threshold_stds_sum', stds)
    if global_vars.get('moving_threshold'):
        all_data = get_whole_netflow_data(
            f'../EEGNAS/EEGNAS/data/netflow/{global_vars.get("as_to_test")}_{global_vars.get("date_range")}.csv')
        global_vars.set(f'moving_threshold', get_moving_threshold(all_data['sum'].values, global_vars.get('netflow_threshold_std'))
        [global_vars.get('input_height') - global_vars.get('start_hour') - 3 + 1:])
    if global_vars.get('handovers'):
        global_vars.set('netflow_handover_locations', [str(h) for h in global_vars.get('handovers')])
        global_vars.set('max_handovers', len(global_vars.get('handovers')))
    elif global_vars.get('same_handover_locations'):
        if global_vars.get('unique_test_model'):
            global_vars.set('test_handover_locs', [global_vars.get('as_to_test')])
        if global_vars.get('test_handover_locs'):
            as_for_locs = global_vars.get('test_handover_locs')
        else:
            as_for_locs = global_vars.get('autonomous_systems')
        file_paths = [f"../EEGNAS/EEGNAS/data/netflow{subfolder_str}/{ats}_" \
                      f"{global_vars.get('date_range')}.csv" for ats in as_for_locs]
        handover_locations = get_handover_locations(file_paths)
        global_vars.set('netflow_handover_locations', handover_locations)
        global_vars.set('max_handovers', len(handover_locations))

    write_dict(global_vars.config, f'{global_vars.get("config_folder_name")}/config.txt')
    if global_vars.get('use_sacred'):
        ex.add_artifact(f'{global_vars.get("config_folder_name")}/config.txt')
    if global_vars.get('export_data_folder'):
        if global_vars.get('k_fold'):
            export_kfold_data(global_vars.get('export_data_folder'))
        else:
            export_train_test_data(global_vars.get('export_data_folder'))
        sys.exit(0)
    if global_vars.get('k_fold'):
        res = kfold_exp(global_vars.get('config_folder_name'))
    else:
        res = no_kfold_exp(global_vars.get('config_folder_name'))
    if global_vars.get('only_train_models'):
        return
    for segment in global_vars.get('classification_reports').keys():
        classification_filename = f'{global_vars.get("config_folder_name")}/{global_vars.get("prediction_exp_name")}_{segment}_avg_by_fold_report.csv'
        segment_classif_report = average_classification_reports(global_vars.get('classification_reports')[segment])
        pd.DataFrame([segment_classif_report])[DF_COLS].to_csv(classification_filename, index=False)
        if global_vars.get('use_sacred'):
            ex.add_artifact(classification_filename)
    pd.concat(ALL_CLASSIF_REPORTS).to_csv(f'{folder_name}/all_classification_reports.csv')
    pd.concat(ALL_CLASSIF_REPORTS_BY_FOLD).to_csv(f'{folder_name}/all_classification_reports_by_fold.csv')
    return res


if __name__ == '__main__':
    exp_id = get_exp_id('prediction_results')
    folder_name = f'prediction_results/{exp_id}'
    create_folder(folder_name)
    atexit.register(exit_handler, folder_name, len(sys.argv) > 2)
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    global_vars.init_config('netflow_config.ini')
    shutil.copyfile('netflow_config.ini', f'{folder_name}/{sys.argv[1]}.ini')
    configurations = get_configurations(sys.argv[1], global_vars.configs)
    if 'all_as_combinations' in configurations[0][list(configurations[0].keys())[1]].keys():
        if configurations[0][list(configurations[0].keys())[1]]['all_as_combinations']:
            new_configurations = []
            ass = configurations[0][list(configurations[0].keys())[1]]['autonomous_systems']
            for config in configurations:
                as_to_test = config[list(config.keys())[1]]['as_to_test']
                ass.remove(as_to_test)
                ass.insert(0, as_to_test)
                for i in range(1, len(ass) + 1):
                    new_config = deepcopy(config)
                    new_config[list(new_config.keys())[1]]['autonomous_systems'] = ass[:i]
                    new_configurations.append(new_config)
            configurations = new_configurations
    elif 'all_ensemble_permutations' in configurations[0][list(configurations[0].keys())[1]].keys():
        if configurations[0][list(configurations[0].keys())[1]]['all_ensemble_permutations']:
            new_configurations = []
            model_names = configurations[0][list(configurations[0].keys())[1]]['evaluator']
            for config in configurations:
                for model_set in powerset(model_names):
                    if len(model_set) == 0:
                        continue
                    if 'only_ensembles' in configurations[0][list(configurations[0].keys())[1]]:
                        if len(model_set) == 1 and configurations[0][list(configurations[0].keys())[1]]['only_ensembles']:
                            continue
                    new_config = deepcopy(config)
                    if len(model_set) == 1:
                        new_config[list(new_config.keys())[1]]['evaluator'] = model_set[0]
                    else:
                        new_config[list(new_config.keys())[1]]['evaluator'] = list(model_set)
                    new_configurations.append(new_config)
            configurations = new_configurations
    multiple_values = deepcopy(get_multiple_values(configurations))
    for idx, configuration in enumerate(configurations):
        config_folder_name = f'{folder_name}/{idx+1}'
        create_folder(config_folder_name)
        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y")
        global_vars.set_config(configuration)
        global_vars.set('classification_reports', defaultdict(list))
        set_params_by_dataset('../EEGNAS/configurations/dataset_params.ini')
        global_vars.set('folder_name', folder_name)
        global_vars.set('config_folder_name', config_folder_name)
        configuration['DEFAULT']['as_to_test'] = global_vars.get('as_to_test')
        configuration['DEFAULT']['k_fold'] = global_vars.get('k_fold')
        if global_vars.get('start_idx') > idx + 1:
            continue
        set_gpu()
        if len(ex.observers) == 0 and len(sys.argv) <= 2:
            ex.observers.append(
                MongoObserver.create(url=f'mongodb://{global_vars.get("mongodb_server")}/{global_vars.get("mongodb_db")}',
                                     db_name=global_vars.get("mongodb_db")))
        global_vars.set('no_shuffle', True)
        stop_criterion, iterator, loss_function, monitors = get_normal_settings()
        if global_vars.get('highest_handover_overflow'):
            loss_function = F.nll_loss
        global_vars.set('evaluator_obj', NN_Trainer(iterator, loss_function, stop_criterion, monitors))
        global_vars.set('prediction_exp_name', f'{exp_id}_{idx+1}')
        global_vars.set('prediction_exp_name', add_params_to_name(global_vars.get('prediction_exp_name'), multiple_values))
        if len(global_vars.get('prediction_exp_name')) > 200:
            global_vars.set('prediction_exp_name', f'{global_vars.get("prediction_exp_name")[:200]}...')
        if global_vars.get('plotting_problem') == 'regression':
            global_vars.set('threshold_str', 'netflow')
        elif global_vars.get('plotting_problem') == 'classification':
            global_vars.set('threshold_str', 'aggregate')
        if global_vars.get('top_handovers'):
            global_vars.set('eeg_chans', global_vars.get('top_handovers'))
        else:
            global_vars.set('eeg_chans', global_vars.get('max_handovers'))
        try:
            if global_vars.get('use_sacred'):
                ex.add_config(configuration)
                ex.run(options={'--name': global_vars.get('prediction_exp_name')})
            else:
                main()
        except Exception as e:
            print('experiment failed. Exception message: %s' % (str(e)))
            print(traceback.format_exc())
            print(f'failed experiment {exp_id}_{idx+1}, continuing...')