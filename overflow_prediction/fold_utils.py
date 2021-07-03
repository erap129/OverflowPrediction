import math
import sys
import os
from copy import deepcopy

import torch
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, average_precision_score, \
    confusion_matrix, auc, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/../eegnas')
sys.path.append(f'{dir_path}/../EEGNAS')
from results_aggregation import rrse, rae, corr
from sklearn.model_selection import KFold, train_test_split, TimeSeriesSplit
from EEGNAS import global_vars
from EEGNAS.data_preprocessing import get_dataset, makeDummySignalTargets
from EEGNAS.utilities.misc import concat_train_val_sets, unify_dataset
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations
import pandas as pd
# from tsaug import RandomTimeWarp, RandomMagnify, RandomJitter, RandomTrend, Reverse, RandomCrossSum, RandomSidetrack


def get_fold_idxs(AS):
    if global_vars.get('k_fold_time'):
        kf = TimeSeriesSplit(n_splits=global_vars.get('n_folds'))
    else:
        kf = KFold(n_splits=global_vars.get('n_folds'), shuffle=True)
    prev_autonomous_systems = global_vars.get('autonomous_systems')
    global_vars.set('autonomous_systems', [AS])
    dataset = get_dataset('all')
    concat_train_val_sets(dataset)
    dataset = unify_dataset(dataset)
    fold_idxs = {i: {} for i in range(global_vars.get('n_folds'))}
    for fold_num, (train_index, test_index) in enumerate(kf.split(list(range(len(dataset.X))))):
        fold_idxs[fold_num]['train_idxs'] = train_index
        fold_idxs[fold_num]['test_idxs'] = test_index
    global_vars.set('autonomous_systems', prev_autonomous_systems)
    return fold_idxs


def get_data_by_balanced_folds(ASs, fold_idxs, required_num_samples=None):
    prev_autonomous_systems = global_vars.get('autonomous_systems')
    folds = {i: {'X_train': [], 'X_test': [], 'y_train': [], 'y_test': []} for i in range(global_vars.get('n_folds'))}
    for AS in ASs:
        if AS == 202818 and global_vars.get('problem') == 'classification':
            continue
        global_vars.set('autonomous_systems', [AS])
        dataset = get_dataset('all')
        concat_train_val_sets(dataset)
        dataset = unify_dataset(dataset)
        if np.count_nonzero(dataset.X) == 0:
            print(f'dropped AS {AS} - no common handovers')
            continue
        try:
            if required_num_samples is not None:
                assert len(dataset.X) == required_num_samples
            for fold_idx in range(global_vars.get('n_folds')):
                folds[fold_idx]['X_train'].extend(dataset.X[fold_idxs[fold_idx]['train_idxs']])
                folds[fold_idx]['X_test'].extend(dataset.X[fold_idxs[fold_idx]['test_idxs']])
                folds[fold_idx]['y_train'].extend(dataset.y[fold_idxs[fold_idx]['train_idxs']])
                folds[fold_idx]['y_test'].extend(dataset.y[fold_idxs[fold_idx]['test_idxs']])
        except IndexError:
            print(f'dropped AS {AS}')
        except AssertionError:
            print(f'dropped AS {AS}')
    for key in folds.keys():
        for inner_key in folds[key].keys():
            folds[key][inner_key] = np.stack(folds[key][inner_key], axis=0)
    global_vars.set('autonomous_systems', prev_autonomous_systems)
    return folds


def create_classification_report(actual, predicted, threshold):
    actual = np.array(actual).astype('int')
    pred_rounded = deepcopy(predicted)
    pred_pos = predicted >= threshold
    pred_neg = predicted < threshold
    pred_rounded[pred_pos] = 1
    pred_rounded[pred_neg] = 0

    res = {'support_0': np.bincount(actual)[0]}
    try:
        res['auc'] = roc_auc_score(actual, predicted)
    except ValueError:
        res['auc'] = 0
    try:
        res['auc_precision_recall'] = average_precision_score(actual, predicted)
    except ValueError:
        res['auc_precision_recall'] = 0
    try:
        res['support_1'] = np.bincount(actual)[1]
    except IndexError:
        res['support_1'] = 0
    return res


def create_regression_report(actual, predicted):
    res = {}
    res['rrse'] = rrse(actual, predicted)
    res['rae'] = rae(actual, predicted)
    res['corr'] = corr(actual, predicted)
    return res


def MTS_evaluate(data, X, Y, model, batch_size):
    evaluateL2 = nn.MSELoss(size_average=False);
    evaluateL1 = nn.L1Loss(size_average=False)
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = X.permute(0, 2, 1)
        X = X[:, :, :, None]
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict, output));
            test = torch.cat((test, Y));

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).data
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data
        n_samples += (output.size(0) * data.m);
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis=0);
    sigma_g = (Ytest).std(axis=0);
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return float(rse), float(rae), correlation;


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calc_tpr_fpr_precision_recall(actual, predicted):
    all_thresholds = np.linspace(actual.min(), actual.max(), 1000)
    fprs, tprs, precisions, recalls, thresholds = [], [], [], [], []
    for threshold in all_thresholds:
        tp, p, fp, n, fn = 0, 0, 0, 0, 0
        for real, pred in zip(actual, predicted):
            if real >= threshold:
                p += 1
                if pred >= threshold:
                    tp += 1
                else:
                    fn += 1
            else:
                n += 1
                if pred >= threshold:
                    fp += 1
        if p == 0:
            tpr = 0
        else:
            tpr = tp / p
        if n == 0:
            continue
        else:
            fpr = fp / n
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(threshold)
    zipped = list(zip(fprs, tprs, precisions, recalls, thresholds))
    zipped.sort()
    fprs, tprs, precisions, recalls, thresholds = zip(*zipped)
    return pd.DataFrame({'fpr': fprs, 'tpr': tprs, 'precision': precisions, 'recall': recalls, 'threshold': thresholds})


def get_classification_curves(actual, predicted):
    fpr, tpr, thresholds_roc = roc_curve(actual, predicted)
    fpr_tpr = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds_roc})
    precision, recall, thresholds_pr = precision_recall_curve(actual, predicted)
    precision_recall = pd.DataFrame({'precision': precision[1:], 'recall': recall[1:], 'threshold': thresholds_pr})
    return fpr_tpr, precision_recall


def plot_classification_curves(fpr, tpr, precision, recall, folder_name, handover):
    roc_filename = f'{folder_name}/roc_curves_{handover}.png'
    precision_recall_filename = f'{folder_name}/precision_recall_{handover}.png'

    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(roc_filename)

    plt.clf()
    plt.plot(recall, precision, lw=2)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig(precision_recall_filename)

    return roc_filename, precision_recall_filename


def average_classification_reports(report_dicts):
    avg_dict = {}
    for key in report_dicts[0].keys():
        try:
            avg_dict[key] = np.average([rprt[key] for rprt in report_dicts])
        except Exception as e:
            avg_dict[key] = report_dicts[0][key]
    return avg_dict


def get_start_point_by_start_hour(start_hour, df):
    return np.flatnonzero(df.index.hour == start_hour)[0]


def prepare_y_for_classification(folds, folds_target):
    for fold in folds:
        folds[fold] = find_highest_overflow_handover(folds[fold], 'y_train')
        folds[fold] = find_highest_overflow_handover(folds[fold], 'y_test')
    for fold in folds_target:
        folds_target[fold] = find_highest_overflow_handover(folds_target[fold], 'y_train')
        folds_target[fold] = find_highest_overflow_handover(folds_target[fold], 'y_test')
    return folds, folds_target


def find_highest_overflow_handover(fold, key):
    tmp_y_train = []
    res_y_train = []
    y_train = [x.reshape(11, 5) for x in fold[key]]
    y_train = [[max(y) for y in x] for x in y_train]
    for handover, handover_max_vols in zip(global_vars.get('netflow_handovers'), np.array(y_train).T):
        tmp_y_train.append([vol / global_vars.get(f'netflow_threshold_{handover}') if vol > global_vars.get(
            f'netflow_threshold_{handover}') else 0 for vol in handover_max_vols])
    tmp_y_train = np.array(tmp_y_train).T
    for i in range(len(tmp_y_train)):
        if sum(tmp_y_train[i]) > 0:
            max_idx = np.argmax(tmp_y_train[i])
            res_y_train.append([0] * (len(tmp_y_train[i]) + 1))
            res_y_train[i][max_idx] = 1
        else:
            res_y_train.append([0] * (len(tmp_y_train[i])))
            res_y_train[i].append(1)
    fold[key] = np.array(list(map(lambda x: np.argmax(x), np.array(res_y_train))))
    return fold


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def filter_eegnas_population_files(filename):
    res = True
    if global_vars.get('per_handover_prediction'):
        res = res and 'per_handover' in filename
    else:
        res = res and 'per_handover' not in filename
    if global_vars.get('top_handovers'):
        res = res and f'top{global_vars.get("top_handovers")}' in filename
    else:
        res = res and 'top' not in filename
    if global_vars.get('same_handover_locations'):
        res = res and 'samelocs' in filename
    else:
        res = res and 'samelocs' not in filename
    if (global_vars.get('current_fold_idx') or str(global_vars.get('current_fold_idx')) == '0') and\
            global_vars.get('current_fold_idx') is not None:
        res = res and f'fold_{global_vars.get("current_fold_idx")}' in filename
    if global_vars.get('netflow_subfolder') and 'corona' in global_vars.get('netflow_subfolder'):
        res = res and 'corona' in filename
    else:
        res = res and 'corona' not in filename
    if global_vars.get('eegnas_iteration'):
        res = res and f'iteration_{global_vars.get("eegnas_iteration")}' in filename
    else:
        res = res and f'iteration_{global_vars.get("eegnas_iteration")}' not in filename
    if global_vars.get('plotting_problem') == 'classification':
        res = res and 'classification' in filename
    else:
        res = res and 'classification' not in filename
    return res


def split_train_test_by_date(data, datetimes_X, datetimes_Y):
    date = pd.to_datetime(global_vars.get('train_test_split_date'), dayfirst=True).date()
    X = data.X
    y = data.y
    datetimes_X = [[pd.to_datetime(x).date() for x in dates] for dates in datetimes_X]
    idx = None
    for i, dates in enumerate(datetimes_X):
        if date in dates:
            idx = i
            break

    if len(datetimes_X) < len(X):
        all_X_train, all_X_valid, all_X_test, all_y_train, all_y_valid, all_y_test = [], [], [], [], [], []
        X = X.reshape(len(global_vars.get('autonomous_systems')), len(datetimes_X), X.shape[1], X.shape[2])
        y = y.reshape(len(global_vars.get('autonomous_systems')), len(datetimes_X), y.shape[1])
        for as_x, as_y in zip(X, y):
            X_train, X_valid, y_train, y_valid = train_test_split(as_x[:idx], as_y[:idx],
                                                                  test_size=global_vars.get('valid_set_fraction'))
            all_X_train.extend(X_train)
            all_X_valid.extend(X_valid)
            all_X_test.extend(as_x[idx:])
            all_y_train.extend(y_train)
            all_y_valid.extend(y_valid)
            all_y_test.extend(as_y[idx:])
    else:
        all_X_train, all_X_valid, all_y_train, all_y_valid = train_test_split(X[:idx], y[:idx],
                                                                              test_size=global_vars.get(
                                                                                  'valid_set_fraction'))
        all_X_test, all_y_test = X[idx:], y[idx:]
    train_set, valid_set, test_set = makeDummySignalTargets(all_X_train, all_y_train, all_X_valid, all_y_valid,
                                                            all_X_test, all_y_test)
    dataset = {'train': train_set, 'valid': valid_set, 'test': test_set}
    datetimes = {'train': datetimes_Y[:idx], 'test': datetimes_Y[idx:]}
    return dataset, datetimes


def augment_regression_data(X, y, factor=4):
    my_aug = (
            RandomMagnify(max_zoom=4.0, min_zoom=2.0) @ 0.5
            + RandomTimeWarp() @ 0.5
            + RandomJitter(strength=0.1) @ 0.5
            + RandomTrend(min_anchor=-0.5, max_anchor=0.5) @ 0.5
            + Reverse() @ 0.1
            + RandomCrossSum() @ 0.5
            + RandomSidetrack() @ 0.5
    )
    X_all = []
    y_all = []
    for i in range(factor):
        X_aug = my_aug.run(X)
        X_all.append(X_aug)
        y_all.append(y)
    return np.concatenate(X_all).astype(np.float32), np.concatenate(y_all)


def augment_dataset(dataset):
    for section in dataset.keys():
        if section != 'test':
            dataset[section].X, dataset[section].y = augment_regression_data(dataset[section].X, dataset[section].y)
    return dataset


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def predict_by_batch(model, data):
    dataloader = DataLoader(data, batch_size=global_vars.get('batch_size'))
    all_preds = []
    # for batch in chunks(data, global_vars.get('batch_size')):
    for batch in dataloader:
        batch = batch.cuda()
        pred = model(batch).cpu()
        torch.cuda.empty_cache()
        all_preds.append(pred)
    return torch.stack(all_preds)


# def bin_by_day(df):




