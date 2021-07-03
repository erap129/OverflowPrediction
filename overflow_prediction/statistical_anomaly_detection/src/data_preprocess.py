from numpy import array
import numpy as np
import json
from configparser import ConfigParser
from datetime import date
from datetime import datetime
import holidays
import pandas as pd
import os
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
conf = ConfigParser()
conf.read('config.ini')


# split a univariate sequence into samples
def split_univariate_sequence(sequence, n_steps, slide_len):
    X, y = list(), list()
    for i in range(0, len(sequence), slide_len):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# # split a multivariate sequence into samples
# def split_parallel_sequences(sequences, n_steps, slide_len):
#     X, y = list(), list()
#     for i in range(0, len(sequences), slide_len):
#         # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the dataset
#         if end_ix > len(sequences) - 1:
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)


# split a multivariate sequence into samples
def split_multuple_sequences(sequences, n_steps, slide_len):
    X, y = list(), list()
    for i in range(0, len(sequences), slide_len):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def split_sequence(sequence, n_steps, n_steps_ahead, jumps, buffer):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix % jumps != 0:
            continue
        if end_ix + n_steps_ahead + buffer - 1 > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix+buffer:end_ix+buffer+n_steps_ahead]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_parallel_sequences(sequences, n_steps, n_steps_ahead, jumps, buffer):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix % jumps != 0:
            continue
        if end_ix + n_steps_ahead + buffer - 1 > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix+buffer:end_ix+buffer+n_steps_ahead, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# split a multivariate sequence into samples
def split_multuple_sequences_multi_step(sequences, n_steps_in, slide_len, n_steps_out):
    X, y = list(), list()
    for i in range(0, len(sequences), slide_len):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def prepare_multiple_data_sample(data_sample, n_steps_out):
    # the last column should be handovers sum or overflow array
    pred_array = data_sample[len(data_sample) - 1][n_steps_out:]
    del data_sample[len(data_sample) - 1]
    data_sample = [x[:len(x) - n_steps_out] for x in data_sample]
    data_sample.append(pred_array)
    return data_sample


def prepare_multiple_data_sample_classification(data_sample, n_steps_out):
    # the last column should be handovers sum or overflow array
    original_data = get_overflow_array(data_sample[len(data_sample) - 1][n_steps_out:])
    del data_sample[len(data_sample) - 1]
    data_sample = [x[:len(x) - n_steps_out] for x in data_sample]
    data_sample.append(original_data)
    return data_sample


def scale_data_sample(data_sample):
    result = []
    for sample in data_sample:
        min_vol = min(sample)
        max_vol = max(sample)
        sample = list(map(lambda y: (y - min_vol)/(max_vol-min_vol), sample))
        result.append(sample)
    return result


def multivariate_data_preprocess(vols):
    data_sample = []
    timestamps = None
    source_as = json.loads(conf.get('LSTM', 'source_as'))
    source_as_idx = None
    for i, items in enumerate(vols.items()):
        key, value = items
        if key == source_as:
            source_as_idx = i
        df = pd.DataFrame([json.loads(value[0]), json.loads(value[1])], index=['ts', 'vol']).T
        df = df.sort_values(by='ts')
        data_sample.append(array(df['vol']))
        if timestamps is None:
            timestamps = array(df['ts'])
    # weekends = get_weekends_feature(timestamps)
    # eu_holidays = get_holidays_feature(timestamps)
    # data_sample.append(weekends)
    # data_sample.append(eu_holidays)
    # normalizing the data
    data_sample = array(scale_data_sample(data_sample))
    original_data = [sum(x) for x in zip(*data_sample)]
    original_data = [a - b for a, b in zip(original_data, data_sample[source_as_idx])]
    data_sample = list(data_sample)
    data_sample.append(array(original_data))

    return data_sample, source_as, original_data, timestamps


def get_overflow_array(timeseries, overflow_thresh=1):
    timeseries = array(timeseries)
    nans = np.argwhere(np.isnan(timeseries))
    if len(nans) > 0:
        mean = np.mean(timeseries[nans[len(nans) - 1][0] + 1:])
        std = np.std(timeseries[nans[len(nans) - 1][0] + 1:])
    else:
        mean = np.mean(timeseries)
        std = np.std(timeseries)
    res = []
    for vol in timeseries:
        if vol >= mean + overflow_thresh * std:
            res.append(1)
        else:
            res.append(0)
    return res


def normalize(l):
    arr = array(l)
    min_val = arr.min()
    max_val = arr.max()
    return list(map(lambda x: (x - min_val) / (max_val - min_val), l))


def ep_to_day(ep):
    return datetime.fromtimestamp(ep).strftime("%A")


def get_weekends_feature(timestamps):
    weekends = []
    for ts in timestamps:
        day = ep_to_day(ts)
        if day == 'Sunday' or day == 'Saturday':
            weekends.append(1)
        else:
            weekends.append(0)
    return array(weekends)


def get_holidays_feature(timestamps):
    holidays_feature = []
    de_holidays = holidays.Germany()
    for ts in timestamps:
        h_date = datetime.fromtimestamp(ts)
        if date(h_date.year, h_date.month, h_date.day) in de_holidays:
            holidays_feature.append(1)
        else:
            holidays_feature.append(0)
    return array(holidays_feature)


def link_utilization_preprocess(timeseries_df):
    timeseries_df.sort_values(by=['group', 'timestamp'], inplace=True)
    timeseries_df.set_index(keys=['group'], drop=False, inplace=True)
    interfaces_ids = timeseries_df['group'].unique().tolist()
    interfaces = []
    for interface_id in interfaces_ids:
        interfaces.append(timeseries_df.loc[timeseries_df.group == interface_id])
    return interfaces, interfaces_ids


def find_anomaly_by_threshold(data_path, result_path, name):
    df = pd.read_csv(data_path)
    error = df['normalized_error']
    anomaly = get_overflow_array(error)
    pd.DataFrame.from_dict({'ts': df['ts'], 'anomaly': anomaly}).to_csv(result_path + name)


def correlate_anomalies(data_path_netflow, data_path_snmp, result_path, name):
    df1 = pd.read_csv(data_path_netflow)
    df2 = pd.read_csv(data_path_snmp)
    df_inner = pd.merge(df1, df2, on='ts', how='inner')
    df_inner = df_inner.loc[df_inner['anomaly_x'] == 1]
    df_inner.to_csv(result_path + name, index=False)


def calculate_performance_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def preprocess_netflow_data(files, n_before, n_ahead, jumps, buffer, overflow_thresh):
    all_X, all_y, all_datetimes_X, all_datetimes_Y, all_handovers = [], [], [], [], []
    for file in files:
        all_data = get_whole_netflow_data(file)
        overflow_arr = get_overflow_array(all_data['sum'], overflow_thresh)
        all_data['overflow'] = overflow_arr
        # del all_data['sum']
        all_data.fillna(method='ffill', inplace=True)
        all_data.fillna(method='bfill', inplace=True)
        sample_list, y = split_parallel_sequences(all_data.values, n_before, n_ahead, jumps, buffer)
        datetimes_X, datetimes_Y = split_sequence(all_data.index, n_before, n_ahead, jumps, buffer)
        all_datetimes_X.extend(datetimes_X)
        all_datetimes_Y.extend(datetimes_Y)
        num_handovers = sample_list.shape[2] - 1
        all_X.extend(sample_list.swapaxes(1, 2)[:, :num_handovers])
        all_y.extend(y.swapaxes(1, 2)[:, num_handovers])
        all_handovers.append(all_data)
    all_y = list(map(lambda x: 1 if sum(x) > 0 else 0, all_y))
    prepare_output(all_X)
    return np.stack(all_X, axis=0), np.stack(all_y, axis=0), \
           np.stack(all_datetimes_X, axis=0), np.stack(all_datetimes_Y, axis=0), all_handovers


def preprocess_multistep_netflow_data(files, n_before, n_ahead, jumps, buffer, overflow_thresh):
    all_X, all_y, all_datetimes_X, all_datetimes_Y, all_handovers = [], [], [], [], []
    for file in files:
        all_data = get_whole_netflow_data(file)
        all_data.fillna(method='ffill', inplace=True)
        all_data.fillna(method='bfill', inplace=True)
        overflow_thresh = np.mean(all_data['sum']) + overflow_thresh * np.std(all_data['sum'])
        sample_list, y = split_parallel_sequences(all_data.values, n_before, n_ahead, jumps, buffer)
        datetimes_X, datetimes_Y = split_sequence(all_data.index, n_before, n_ahead, jumps, buffer)
        all_datetimes_X.extend(datetimes_X)
        all_datetimes_Y.extend(datetimes_Y)
        num_handovers = sample_list.shape[2] - 1
        all_X.extend(sample_list.swapaxes(1, 2)[:, :num_handovers])
        all_y.extend(y.swapaxes(1, 2)[:, num_handovers])
        all_handovers.append(all_data)
    prepare_output(all_X)
    return np.stack(all_X, axis=0), np.stack(all_y, axis=0), \
           np.stack(all_datetimes_X, axis=0), np.stack(all_datetimes_Y, axis=0), overflow_thresh, all_handovers


def preprocess_handover_regression_data(files, n_before, n_ahead, jumps, buffer, overflow_thresh):
    all_X, all_y, all_datetimes_X, all_datetimes_Y, all_handovers, overflow_thresholds = [], [], [], [], [], []
    for file in files:
        all_data = get_whole_netflow_data(file)
        del all_data['sum']
        all_data.fillna(method='ffill', inplace=True)
        all_data.fillna(method='bfill', inplace=True)
        for col in all_data.columns:
            overflow_thresholds.append(np.mean(all_data[col]) + overflow_thresh * np.std(all_data[col]))
        sample_list, y = split_parallel_sequences(all_data.values, n_before, n_ahead, jumps, buffer)
        datetimes_X, datetimes_Y = split_sequence(all_data.index, n_before, n_ahead, jumps, buffer)
        all_datetimes_X.extend(datetimes_X)
        all_datetimes_Y.extend(datetimes_Y)
        # num_handovers = sample_list.shape[2]
        all_X.extend(sample_list)
        all_y.extend(y)
        all_handovers.append(all_data)
    prepare_output(all_X)
    return np.stack(all_X, axis=0), np.stack(all_y, axis=0), \
           np.stack(all_datetimes_X, axis=0), np.stack(all_datetimes_Y, axis=0), overflow_thresholds, all_handovers


def prepare_output(all_X):
    max_handovers = max(x.shape[0] for x in all_X)
    for idx in range(len(all_X)):
        if all_X[idx].shape[0] < max_handovers:
            all_X[idx] = np.pad(all_X[idx], pad_width=((max_handovers - all_X[idx].shape[0], 0), (0, 0)),
                                mode='constant')
        elif all_X[idx].shape[0] > max_handovers:
            all_X[idx] = all_X[idx][:max_handovers]


def get_whole_netflow_data(file):
    orig_df = pd.read_csv(file)
    own_as_num = os.path.basename(file).split('_')[0]
    vols = {}
    dfs = []
    for index, row in orig_df.iterrows():
        vols[row['id']] = [row['ts'], row['vol']]
    idx = 0
    for key, value in vols.items():
        datetimes = [datetime.utcfromtimestamp(int(tm)) for tm in json.loads(value[0])]
        df = pd.DataFrame(list(zip(datetimes, json.loads(value[1]))), columns=['ts', orig_df.iloc[idx]['id']])
        df = df.sort_values(by='ts')
        df.index = pd.to_datetime(df['ts'])
        df = df.drop(columns=['ts'])
        df = df.apply(lambda x: (x - df[key].min()) / (df[key].max() - df[key].min()))
        df = df.resample('H').pad()
        dfs.append(df)
        idx += 1
    all_data = pd.concat(dfs, axis=1)
    all_data = all_data.dropna(axis=1, how='any')
    all_data['sum'] = all_data.drop(labels=int(own_as_num), axis=1, errors='ignore').sum(axis=1)
    all_data = all_data[np.flatnonzero(df.index.hour == 15)[0]:]
    return all_data


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    # find_anomaly_by_threshold(
    #      conf.get('Paths', 'data_path') + conf.get('Paths', 'data_file'),
    #      conf.get('Paths', 'output_path') + conf.get('Paths', 'folder_name'), 'netflow_anomaly_2.csv')
    correlate_anomalies(
         r'..\..\data\results-anomaly-detection\anomaly_with_threshold\netflow_anomaly_2.csv',
         r'..\..\data\results-anomaly-detection\anomaly_with_threshold\snmp_anomaly_2.csv',
         conf.get('Paths', 'output_path') + conf.get('Paths', 'folder_name'), 'correlation_2.csv')
