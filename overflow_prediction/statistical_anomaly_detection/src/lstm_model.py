import json
import os
from configparser import ConfigParser
from datetime import datetime
from itertools import product
from os import listdir
from os.path import isfile, join
import pytorch_lstm as pt
import data_preprocess as dp
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from keras.models import Sequential
from numpy import array
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def univariate_lstm_model_keras(data_sample, n_steps, slide_len, n_features, lstm_units, activation, optimizer, loss,
                                epochs,
                                batch_size, csv_logger, l_rate):
    sample_list, y = dp.split_univariate_sequence(data_sample, n_steps, slide_len)
    try:
        X = sample_list.reshape((sample_list.shape[0], sample_list.shape[1], n_features))
    except:
        raise Exception('n_steps={} should be < len(data_sample)={}'.format(n_steps, len(data_sample)))
    model = build_model(X, n_features, n_steps, y, lstm_units, activation, optimizer, loss, epochs,
                        batch_size, csv_logger, l_rate)

    predictions = get_predictions(model, n_features, n_steps, sample_list)
    error = array(predictions) - data_sample[n_steps::slide_len]

    print("Maximum reconstruction error was %.1f" % error.max())

    return predictions, array(error)


def multivariate_lstm_model_keras(data_sample, n_steps, slide_len, n_features, lstm_units, activation, optimizer, loss,
                                  epochs, batch_size, original_data, csv_logger, l_rate, n_steps_out):
    # n_features = sample_list.shape[2] according to lstm guide
    data_sample = array(list(map(lambda x: x.reshape((len(x), 1)), data_sample)))
    dataset = np.hstack(data_sample)
    softmax = 'true' if json.loads(conf.get('LSTM', 'predict_overflows')) else None
    if conf.get('LSTM', 'multivariate_type') == 'multiple':
        sample_list, y = dp.split_multuple_sequences(dataset, n_steps, slide_len)
        model = build_model(sample_list, sample_list.shape[2], n_steps, y, lstm_units, activation,
                            optimizer, loss, epochs, batch_size, csv_logger, l_rate, softmax)
        predictions = get_predictions(model, sample_list.shape[2], n_steps, sample_list)
        error = array(predictions)[:len(predictions) - 1] - original_data[
                                                            n_steps + n_steps_out - 1:len(original_data) - 1:slide_len]
    else:
        sample_list, y = dp.split_parallel_sequences(dataset, n_steps, slide_len)
        model = build_multivariate_parallel_model(sample_list, sample_list.shape[2], n_steps, y, lstm_units, activation,
                                                  optimizer, loss, epochs, batch_size, csv_logger, l_rate)
        predictions = get_multivariate_parallel_predictions(model, sample_list.shape[2], n_steps, sample_list)
        error = array(predictions) - original_data[n_steps::slide_len]
    return predictions, error


def multivariate_lstm(batch_size, epochs, folder_name, loss, lstm_units, n_steps, optimizer,
                      output_path, rep, slide_len, timestamp, l_rate, n_steps_out, X, y, dates_X, dates_y, ex,
                      lstm_layers, all_data, threshold):
    # data_sample, source_as, original_data, timestamps = dp.multivariate_data_preprocess(vols)
    # data_sample = array(dp.prepare_multiple_data_sample(data_sample, n_steps_out))
    # timestamps = timestamps[n_steps_out - 1:]
    # data_sample = array(list(map(lambda x: x.reshape((len(x), 1)), data_sample)))
    # dataset = np.hstack(data_sample)
    X = array(list(map(lambda x: x.T, X)))
    y = array([x[0] for x in y])
    predictions, errors, losses, original_data, dates = pt.multivariate_model(X, y, n_steps, slide_len, lstm_units,
                                                                              epochs, loss,
                                                                              optimizer, l_rate, dates_X, dates_y,
                                                                              pt.MultivariateLSTM, ex, lstm_layers,
                                                                              batch_size)
    # export_results(timestamps, original_data[:len(original_data) - n_steps_out], predictions, slide_len,
    #                error, n_steps, output_path, folder_name, timestamp, n_steps_out, rep, loss_arr)
    flatten_predictions = [x for fold in predictions for x in fold]
    dates = [x for fold in dates for date in fold for x in date]
    df = pd.DataFrame(flatten_predictions)
    df.index = dates
    path = output_path + folder_name + '/{}/_results.csv'.format(str(int(timestamp)))
    df = pd.merge(all_data[0], df, left_index=True, right_index=True, how='outer')
    df.to_csv(path)
    ex.add_artifact(path)
    filtered_df = df.loc[(df.index.hour < 22) & (df.index.hour > 16)].dropna()
    overflow_predictions = [1 if x > threshold else 0 for x in filtered_df[0]]
    overflow_original = [1 if x > threshold else 0 for x in filtered_df['sum']]
    handle_results(ex, filtered_df[0], overflow_predictions, folder_name, output_path, timestamp, overflow_original)


def lstm_classification(batch_size, epochs, folder_name, loss, lstm_units, n_steps,
                        optimizer, output_path, rep, slide_len, timestamp, l_rate,
                        n_steps_out, X, y, dates_X, dates_y, ex, lstm_layers, all_data, use_mini_batches):
    X = array(list(map(lambda x: x.T, X)))
    predictions, errors, losses, original_data, dates = pt.multivariate_model(X, y, n_steps, slide_len, lstm_units,
                                                                              epochs, loss,
                                                                              optimizer, l_rate, dates_X, dates_y,
                                                                              pt.LSTMClassifier, ex, lstm_layers,
                                                                              batch_size, use_mini_batches)
    flatten_predictions = [x for fold in predictions for x in fold[0]]
    flatten_overflow_pct = [x for fold in predictions for x in fold[1]]
    dates = [x for fold in dates for date in fold for x in date]
    path = output_path + folder_name + '/{}/_results.csv'.format(str(int(timestamp)))
    if n_steps_out == 1:
        df = pd.DataFrame({'predictions': flatten_predictions, 'pct': flatten_overflow_pct, 'original': y})
        df.index = dates
        df = pd.merge(all_data[0], df, left_index=True, right_index=True, how='outer')
        df = df.loc[(df.index.hour < 22) & (df.index.hour > 16)].dropna()
        df.to_csv(path)
        handle_results(ex, df['pct'], df['predictions'], folder_name, output_path, timestamp, df['original'])
    else:
        predictions = [x for y in [[x] * n_steps_out for x in flatten_predictions] for x in y]
        df = pd.DataFrame(predictions)
        df.index = dates
        pd.merge(all_data[0], df, left_index=True, right_index=True, how='outer').to_csv(path)
        # export_results_test(date, y_test, prediction, slide_len, error, n_steps, output_path, folder_name, timestamp,
        #                     n_steps_out, rep, loss, ex, 'fold_' + str(i))
        handle_results(ex, flatten_overflow_pct, flatten_predictions, folder_name, output_path, timestamp, y)
    ex.add_artifact(path)


def handle_results(ex, flatten_overflow_pct, flatten_predictions, folder_name, output_path, timestamp, y):
    accuracy, auc_pr, auc_roc, curve_precision, curve_recall, f1, fn, fp, fpr, precision, recall, tn, tp, tpr = \
        calculate_metrics(flatten_overflow_pct, flatten_predictions, y)
    export_curves(curve_precision, curve_recall, ex, fpr, tpr, output_path, folder_name, timestamp)
    export_metrics_results(output_path, folder_name, timestamp, accuracy, precision, recall, f1, tn, fp, fn,
                           tp, ex, sum(y), len(y), auc_roc, auc_pr)


def calculate_metrics(flatten_overflow_pct, flatten_predictions, y):
    fpr, tpr, thresholds = roc_curve(y, flatten_overflow_pct)
    curve_precision, curve_recall, thresholds = precision_recall_curve(y, flatten_overflow_pct)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(curve_recall, curve_precision)
    accuracy, precision, recall, f1 = dp.calculate_performance_metrics(y, flatten_predictions)
    tn, fp, fn, tp = confusion_matrix(y, flatten_predictions).ravel()
    return accuracy, auc_pr, auc_roc, curve_precision, curve_recall, f1, fn, fp, fpr, precision, recall, tn, tp, tpr


def multistep_multivariate_lstm(batch_size, epochs, folder_name, loss, lstm_units, n_features, n_steps,
                                optimizer, output_path, rep, slide_len, timestamp, vols, csv_logger, l_rate,
                                n_steps_out, X, y, dates_X, dates_y, ex, threshold, lstm_layers, use_mb):
    predictions, errors, losses, original_data, dates = pt.multivariate_model(X, y, n_steps, slide_len, lstm_units,
                                                                              epochs, loss,
                                                                              optimizer, l_rate, dates_X, dates_y,
                                                                              pt.MultivariateMultistepLSTM, ex,
                                                                              lstm_layers, batch_size, use_mb)
    flatten_predictions = [x for fold in predictions for x in fold]
    flatten_dates = [pd.to_datetime(date) for fold in dates for date in fold]
    if slide_len == 1:
        relevant_indexes = [index for index, date in enumerate(flatten_dates) if date[0].hour == 17]
        flatten_predictions = [flatten_predictions[i] for i in relevant_indexes]
        y = [y[i] for i in relevant_indexes]
    max_5_hour_values = [max(x) for x in flatten_predictions]
    max_vol, min_vol = max(max_5_hour_values), min(max_5_hour_values)
    normalized_vol_for_auc = list(map(lambda x: (x - min_vol) / (max_vol - min_vol), max_5_hour_values))
    flatten_predictions = [1 if any(a > threshold for a in l) else 0 for l in flatten_predictions]
    y = [1 if any(a > threshold for a in l) else 0 for l in y]
    handle_results(ex, normalized_vol_for_auc, flatten_predictions, folder_name, output_path, timestamp, y)


def multistep_parallel_multivariate_lstm(batch_size, epochs, folder_name, loss, lstm_units, n_features, n_steps,
                                         optimizer, output_path, rep, slide_len, timestamp, vols, csv_logger, l_rate,
                                         n_steps_out, X, y, dates_X, dates_y, ex, threshold, lstm_layers, use_mb,
                                         all_handovers):
    predictions, errors, losses, original_data, dates = pt.multivariate_model(X, y, n_steps, slide_len, lstm_units,
                                                                              epochs, loss,
                                                                              optimizer, l_rate, dates_X, dates_y,
                                                                              pt.MultivariateParallelMultistepLSTM, ex,
                                                                              lstm_layers, batch_size, use_mb)
    flatten_predictions = [x for fold in predictions for x in fold]
    flatten_dates = [pd.to_datetime(date) for fold in dates for date in fold]
    if slide_len == 1:
        relevant_indexes = [index for index, date in enumerate(flatten_dates) if date[0].hour == 17]
        flatten_predictions = [flatten_predictions[i] for i in relevant_indexes]
        y = [y[i] for i in relevant_indexes]
    max_5_hour_values = [[max(x) for x in pd.DataFrame(val).T.to_numpy()] for val in flatten_predictions]
    max_5_hour_values = [[x[i] for x in max_5_hour_values] for i in range(X.shape[2])]
    y = [[max(x) for x in pd.DataFrame(val).T.to_numpy()] for val in y]
    y = [[x[i] for x in y] for i in range(X.shape[2])]
    accuracy_l, auc_pr_l, auc_roc_l, f1_l, fn_l, precision_l, recall_l,\
    tn_l, tp_l, fp_l, pos_vals, all_vals = [], [], [], [], [], [], [], [], [], [], [], []
    for handover_predicted_vol, handover_real_vol, thresh, handover_id in \
            zip(max_5_hour_values, y, threshold, list(all_handovers[0].columns)):
        flatten_predictions = [1 if a > thresh else 0 for a in handover_predicted_vol]
        handover_real_vol = [1 if a > thresh else 0 for a in handover_real_vol]
        accuracy, auc_pr, auc_roc, curve_precision, curve_recall, f1, fn, fp, fpr, precision, recall, tn, tp, tpr = \
            calculate_metrics(handover_predicted_vol, flatten_predictions, handover_real_vol)
        export_curves(curve_precision, curve_recall, ex, fpr, tpr, output_path, folder_name, timestamp, str(handover_id))
        accuracy_l.append(accuracy)
        auc_pr_l.append(auc_pr)
        auc_roc_l.append(auc_roc)
        f1_l.append(f1)
        fn_l.append(fn)
        precision_l.append(precision)
        recall_l.append(recall)
        tn_l.append(tn)
        tp_l.append(tp)
        fp_l.append(fp)
        pos_vals.append(str(sum(handover_real_vol)))
        all_vals.append(str(len(handover_real_vol)))
    export_metrics_results(output_path, folder_name, timestamp, accuracy_l, precision_l, recall_l, f1_l, tn_l, fp_l,
                           fn_l, tp_l, ex, pos_vals, all_vals, auc_roc_l, auc_pr_l, list(all_handovers[0].columns))


# using instances having no overflow in the last hour
def get_overflow_predictions(model, n_features, n_steps, sample_list):
    prediction = []
    for sample in sample_list:
        x_input = array(sample)
        if x_input[len(x_input) - 1][len(x_input[len(x_input) - 1]) - 1] == 1:
            prediction.append(0)
            continue
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        prediction.extend(yhat[0])
    return prediction


def get_predictions(model, n_features, n_steps, sample_list):
    prediction = []
    for sample in sample_list:
        x_input = array(sample)
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        prediction.append(yhat[0][0])
        # prediction.extend(yhat[0])
    return prediction


def get_multivariate_parallel_predictions(model, n_features, n_steps, sample_list):
    prediction = []
    for sample in sample_list:
        x_input = array(sample)
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        prediction.append(yhat[0][len(yhat[0]) - 1])
    return prediction


def build_model(X, n_features, n_steps, y, lstm_units, activation, optimizer, loss, epochs, batch_size,
                csv_logger, l_rate, softmax=None):
    model = Sequential()
    model.add(LSTM(lstm_units, activation=activation, input_shape=(n_steps, n_features)))
    # model.add(LSTM(lstm_units, activation=activation, return_sequences=True, input_shape=(n_steps, n_features)))
    # model.add(LSTM(lstm_units, activation=activation))
    # model.add(Dropout(0.2))
    # model.add(Dense(30))
    # model.add(Dropout(0.2))
    if softmax is None:
        model.add(Dense(1))
    else:
        model.add(Dense(2, activation='softmax'))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    if l_rate != "adaptive":
        model.optimizer.lr = l_rate
    model.fit(X, y, epochs=epochs, verbose=2, callbacks=[csv_logger])
    return model


def build_model_multistep_out(X, n_features, n_steps, y, lstm_units, activation, optimizer, loss, epochs, batch_size,
                              csv_logger, l_rate, n_steps_out, softmax=None):
    model = Sequential()
    model.add(LSTM(lstm_units, activation=activation, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(lstm_units, activation='relu'))
    if softmax is None:
        model.add(Dense(n_steps_out))
    else:
        model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss)
    if l_rate != "adaptive":
        model.optimizer.lr = l_rate
    model.fit(X, y, epochs=epochs, verbose=2, callbacks=[csv_logger])
    return model


def build_multivariate_parallel_model(X, n_features, n_steps, y, lstm_units, activation, optimizer, loss, epochs,
                                      batch_size, csv_logger, l_rate):
    model = Sequential()
    model.add(LSTM(lstm_units, activation=activation, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(n_features))
    model.compile(optimizer=optimizer, loss=loss)
    if l_rate != "adaptive":
        model.optimizer.lr = l_rate
    model.fit(X, y, epochs=epochs, verbose=2, callbacks=[csv_logger])
    return model


def config_keras():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
                            allow_soft_placement=True, device_count={'CPU': 4})
    session = tf.Session(config=config)
    K.set_session(session)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


def univariate_lstm(batch_size, epochs, folder_name, loss, lstm_units, n_features, n_steps, optimizer,
                    output_path, rep, slide_len, timestamp, vols, csv_logger, l_rate, n_steps_out):
    if n_steps_out > 1:
        raise Exception('Not implemented yet')
    for key, value in vols.items():
        df = pd.DataFrame([json.loads(value[0]), json.loads(value[1])], index=['ts', 'vol']).T
        df = df.sort_values(by='ts')
        data_sample = dp.normalize(array(df['vol']))
        predictions, error, loss_arr, y_test, timestamps = pt.univariate_model(data_sample, n_steps, slide_len,
                                                                               lstm_units, epochs, loss, optimizer,
                                                                               l_rate, batch_size, df['ts'],
                                                                               n_features=1)
        export_results_test(timestamps, y_test, predictions, slide_len, error, n_steps, output_path, folder_name,
                            timestamp, n_steps_out, rep, loss_arr, str(key))


def handle_snmp(data_path, _, folder_name, output_path, lstm_units, activation, optimizer, loss, epochs, batch_size,
                n_steps, n_features, slide_len, timestamp, rep, csv_logger, l_rate):
    if not os.path.exists(output_path + folder_name + '//' + str(int(timestamp))):
        os.mkdir(output_path + folder_name + '//' + str(int(timestamp)))
    links = [f for f in listdir(data_path) if not isfile(join(data_path, f))]
    for link in links:
        if not os.path.exists(output_path + folder_name + '/{}/{}'.format(str(int(timestamp)), link)):
            os.mkdir(output_path + folder_name + '//' + str(int(timestamp)) + '//' + link)
        interfaces = [i for i in listdir(data_path + '/{}/'.format(link)) if
                      isfile(join(data_path + '/{}/'.format(link), i))]
        for interface in interfaces:
            df = pd.read_csv(data_path + '/{}/{}'.format(link, interface))
            data_sample = df[conf.get('LSTM', 'direction')].apply(lambda x: x / (2 ** 20))
            error_data = {}
            predictions, error = univariate_lstm_model_keras(data_sample, n_steps, slide_len, n_features, lstm_units
                                                             , activation, optimizer, loss, epochs, batch_size,
                                                             csv_logger,
                                                             l_rate)
            export_results(df['timestamp'], data_sample, predictions, slide_len, error, n_steps, output_path,
                           folder_name, timestamp, rep, [], '/{}/{}'.format(link, interface.split('.csv')[0]))


def handle_snmp_utilization(data_path, data_file, folder_name, output_path, lstm_units, activation, optimizer, loss,
                            epochs, batch_size, n_steps, n_features, slide_len, timestamp, rep, csv_logger, l_rate,
                            n_steps_out):
    if not os.path.exists(output_path + folder_name + '//' + str(int(timestamp))):
        os.mkdir(output_path + folder_name + '//' + str(int(timestamp)))
    df = pd.read_csv(data_path + data_file)
    interfaces, ids = dp.link_utilization_preprocess(df)
    for interface, i_id in zip(interfaces, ids):
        # converting volume from byte to GB as AS flow data
        data_sample = dp.normalize(list(interface[conf.get('LSTM', 'direction')].apply(lambda x: x / (2 ** 30))))
        error_data = {}
        predictions, error = univariate_lstm_model_keras(data_sample, n_steps, slide_len, n_features, lstm_units
                                                         , activation, optimizer, loss, epochs, batch_size, csv_logger,
                                                         l_rate)
        export_results(interface['timestamp'], data_sample, predictions, slide_len, error, n_steps, output_path,
                       folder_name, timestamp, n_steps_out, rep, [], '/{}'.format(i_id))


def export_results(timestamps, data_sample, predictions, slide_len, error, n_steps, output_path, folder_name,
                   timestamp, n_steps_out, rep, loss_arr, link=''):
    path = output_path + folder_name + '/{}/{}_results.csv'.format(str(int(timestamp)), link)
    loss_path = output_path + folder_name + '/{}/{}_results_loss.csv'.format(str(int(timestamp)), link)
    pred = [None for _ in range(len(data_sample))]
    err = [None for _ in range(len(data_sample))]
    normalized_err = [None for _ in range(len(data_sample))]
    repetition = [rep + 1 for _ in range(len(data_sample))]
    normalized_error = dp.normalize(np.absolute(error))
    idx = 0
    # for i in range(n_steps, len(data_sample), slide_len):  # for slide_len without multi-step output
    for i in range(n_steps + n_steps_out - 1, len(data_sample)):
        pred[i] = predictions[idx]
        err[i] = error[idx]
        normalized_err[i] = normalized_error[idx]
        idx = idx + 1
    data = {'ts': timestamps, 'original': data_sample, 'prediction': pred, 'error': err,
            'normalized_error': normalized_err, 'repetition': repetition}
    df = pd.DataFrame(data)
    loss = pd.DataFrame({'loss': loss_arr})
    if not os.path.isfile(path):
        df.to_csv(path)
        loss.to_csv(loss_path)
    else:
        df.to_csv(path, mode='a', header=False)
        loss.to_csv(loss_path, mode='a', header=False)


def export_results_test(timestamps, data_sample, predictions, slide_len, error, n_steps, output_path, folder_name,
                        timestamp, n_steps_out, rep, loss_arr, ex, link=''):
    path = output_path + folder_name + '/{}/{}_results.csv'.format(str(int(timestamp)), link)
    loss_path = output_path + folder_name + '/{}/{}_results_loss.csv'.format(str(int(timestamp)), link)
    repetition = [rep + 1 for _ in range(len(data_sample))]
    normalized_error = dp.normalize(np.absolute(error))
    data = {'ts': timestamps, 'original': data_sample, 'prediction': predictions, 'error': error,
            'normalized_error': normalized_error, 'repetition': repetition}
    try:
        df = pd.DataFrame(data)
    except:
        data = {'ts': [str(x) for x in timestamps], 'original': data_sample, 'prediction': predictions, 'error': error,
                'normalized_error': normalized_error, 'repetition': repetition}
        df = pd.DataFrame(data)
    loss = pd.DataFrame({'loss': loss_arr})
    if not os.path.isfile(path):
        df.to_csv(path)
        ex.add_artifact(path)
        loss.to_csv(loss_path)
    else:
        df.to_csv(path, mode='a', header=False)
        loss.to_csv(loss_path, mode='a', header=False)


def export_metrics_results(output_path, folder_name, timestamp, accuracy, precision, recall, f1, tn, fp, fn, tp, ex,
                           positive_vals, all_vals, auc_roc, auc_pr, ids=None):
    path = output_path + folder_name + '/{}/_metrics_results.csv'.format(str(int(timestamp)))
    data = {'accuracy': [accuracy], 'precision': precision, 'recall': recall, 'f1_score': f1, 'tp': tp, 'fp': fp,
            'fn': fn, 'tn': tn, 'positive_values': positive_vals, 'all': all_vals, 'auc_roc': auc_roc, 'auc_pr': auc_pr}
    try:
        df = pd.DataFrame(data)
    except:
        data['accuracy'] = accuracy
        data['id'] = ids
        df = pd.DataFrame(data)
    if not os.path.isfile(path):
        df.to_csv(path, index=False)
        ex.add_artifact(path)
    else:
        df.to_csv(path, mode='a', header=False, index=False)


def export_curves(curve_precision, curve_recall, ex, fpr, tpr, output_path, folder_name, timestamp, name=''):
    pr_curve_path = output_path + folder_name + '/{}/pr_curve_{}.csv'.format(str(int(timestamp)), name)
    roc_curve_path = output_path + folder_name + '/{}/roc_curve_{}.csv'.format(str(int(timestamp)), name)
    pr_curve = {'recall': curve_recall, 'precision': curve_precision}
    pr_df = pd.DataFrame(pr_curve)
    pr_df.to_csv(pr_curve_path, index=False)
    ex.add_artifact(pr_curve_path)
    roc_cur = {'tpr': tpr, 'fpr': fpr}
    roc_df = pd.DataFrame(roc_cur)
    roc_df.to_csv(roc_curve_path, index=False)
    ex.add_artifact(roc_curve_path)


def handle_errors(error_data, folder_name, output_path, timestamp, rep, link=''):
    errors = pd.DataFrame(columns=['ID', 'avg error', 'max error'])
    idx = 0
    for key, value in error_data.items():
        errors.loc[idx] = [key, value['avg error'], value['max error']]
        idx = idx + 1
    path = output_path + folder_name + '/{}/{}errors_rep_{}.csv'.format(str(int(timestamp)), link, rep + 1)
    if not os.path.isfile(path):
        errors.to_csv(path)
    else:
        errors.to_csv(path, mode='a', header=False)


def main(folder_name, data_file, data_path, output_path, lstm_units_list, activations, optimizers, losses, epochs_list,
         batch_sizes, n_steps_list, n_features_list, slide_lens, repetitions, n_steps_out, l_rates):
    config_keras()
    if not os.path.exists(output_path + folder_name):
        os.mkdir(output_path + folder_name)
    func = None
    if conf.get('Data', 'type') == 'snmp':
        func = handle_snmp
    elif conf.get('Data', 'type') == 'snmp_utilization':
        func = handle_snmp_utilization
    prod = product(lstm_units_list, activations, optimizers, losses, epochs_list, batch_sizes, n_steps_list,
                   n_features_list, slide_lens, l_rates, n_steps_out)
    for lstm_units, activation, optimizer, loss, epochs, batch_size, n_steps, n_features, slide_len, l_rate, \
        n_steps_out in prod:
        timestamp = datetime.timestamp(datetime.now())
        for rep in range(repetitions):
            csv_logger = CSVLogger(
                output_path + folder_name + '//' + str(int(timestamp)) + '/log_rep' + str(rep + 1) + '.csv',
                append=True, separator=';')
            func(data_path, data_file, folder_name, output_path, lstm_units, activation, optimizer, loss, epochs,
                 batch_size, n_steps, n_features, slide_len, timestamp, rep, csv_logger, l_rate, n_steps_out)
        update_results_file(activation, batch_size, epochs, folder_name, l_rate, loss, lstm_units, n_features, n_steps,
                            optimizer, output_path, repetitions, slide_len, timestamp, n_steps_out)


def update_results_file(batch_size, epochs, folder_name, l_rate, loss, lstm_units, n_features, n_steps,
                        optimizer, output_path, repetitions, slide_len, timestamp, n_steps_out):
    data = {'exp_ts': timestamp, 'repetitions': repetitions, 'lstm_units': lstm_units,
            'optimizer': optimizer, 'loss': loss, 'epochs': epochs, 'batch_size': batch_size, 'n_steps': n_steps,
            'n_features': n_features, 'slide_len': slide_len, 'learning_rate': l_rate, 'look_forward': n_steps_out}
    column_order = ['exp_ts', 'repetitions', 'lstm_units', 'optimizer', 'loss', 'epochs',
                    'batch_size', 'n_steps', 'look_forward', 'n_features', 'slide_len', 'learning_rate']
    res = pd.DataFrame([data])
    if not os.path.isfile(output_path + folder_name + '/results.csv'):
        res[column_order].to_csv(output_path + folder_name + '/results.csv')
    else:
        with open(output_path + folder_name + '/results.csv', 'a') as f:
            res[column_order].to_csv(f, header=False)


conf = ConfigParser()
conf.read('config.ini')
if __name__ == '__main__':
    main(conf.get('Paths', 'output_folder_name'),
         conf.get('Paths', 'data_file'),
         conf.get('Paths', 'data_path'),
         conf.get('Paths', 'output_path'),
         json.loads(conf.get('LSTM', 'lstm_units')),
         json.loads(conf.get('LSTM', 'activation')),
         json.loads(conf.get('LSTM', 'optimizer')),
         json.loads(conf.get('LSTM', 'loss')),
         json.loads(conf.get('LSTM', 'epochs')),
         json.loads(conf.get('LSTM', 'batch_size')),
         json.loads(conf.get('LSTM', 'n_steps')),
         json.loads(conf.get('LSTM', 'n_features')),
         json.loads(conf.get('LSTM', 'slide_len')),
         json.loads(conf.get('LSTM', 'repetitions')),
         json.loads(conf.get('LSTM', 'look_forward')),
         json.loads(conf.get('LSTM', 'l_rates')))
