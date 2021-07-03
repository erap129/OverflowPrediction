import math

import pandas as pd
import numpy as np
import os

EPSILON = 1e-10


def aggregate_fpr_tpr_results(folder):
    results = {'AS': [], 'avg_fpr': [], 'avg_tpr': [], 'max_fpr': [], 'max_tpr': []}
    AS_folders = [d for d in os.listdir(folder) if os.path.isdir(f'{folder}/{d}')]
    for inner_folder in AS_folders:
        fpr_avg = 0
        tpr_avg = 0
        fpr_max = 0
        tpr_max = 0
        for file in os.listdir(f'{folder}/{inner_folder}'):
            df = pd.read_csv(f'{folder}/{inner_folder}/{file}')
            stats = df[df.fpr <= 0.05].iloc[-1]
            fpr_avg += stats['fpr']
            tpr_avg += stats['tpr']
            if stats['tpr'] > tpr_max:
                fpr_max = stats['fpr']
                tpr_max = stats['tpr']
        fpr_avg /= len(os.listdir(f'{folder}/{inner_folder}'))
        tpr_avg /= len(os.listdir(f'{folder}/{inner_folder}'))
        results['AS'].append(inner_folder)
        results['avg_fpr'].append(fpr_avg)
        results['avg_tpr'].append(tpr_avg)
        results['max_fpr'].append(fpr_max)
        results['max_tpr'].append(tpr_max)
    pd.DataFrame.from_dict(results, orient='index').transpose().to_csv(f'{folder}/aggregated_results.csv')


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))

def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)

def corr(actual, predicted):
    return np.corrcoef(actual, predicted)[0,1]


def get_regression_results(folder):
    AS_folders = [d for d in os.listdir(folder) if os.path.isdir(f'{folder}/{d}')]
    res_list = []
    for inner_folder in AS_folders:
        file = [f for f in os.listdir(f'{folder}/{inner_folder}') if '240_5' in f][0]
        AS_name = [f for f in os.listdir(f'{folder}/{inner_folder}') if 'as_to_test' in f][0]
        AS_name = AS_name[AS_name.index('as_to_test')+11:].split('_')[0]
        df = pd.read_csv(f'{folder}/{inner_folder}/{file}')
        df = df.dropna()
        real_cols = [col for col in df.columns if 'real' in col]
        pred_cols = [col for col in df.columns if 'pred' in col]
        for real, pred in zip(real_cols, pred_cols):
            for measure in [rrse, rae, corr]:
                actual = df[real].values
                predicted = df[pred].values
                measure_val = measure(actual, predicted)
                res = {'AS': AS_name,
                        'handover': real.split('_')[0],
                       'measure': measure.__name__,
                        'value': measure_val}
                res_list.append(res)
    result = pd.DataFrame(res_list)
    res_piv = pd.pivot_table(result, values='value', index=['AS'], columns=['measure'])
    res_piv.to_csv(f'{folder}/regression_results.csv')



if __name__ == '__main__':
    # aggregate_fpr_tpr_results('results_aggregation/fpr_tpr/89')
    get_regression_results("results_aggregation/regression_measures/87")