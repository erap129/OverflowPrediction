import json
import os

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
from configparser import ConfigParser
import lstm_model as lm
from itertools import product
from datetime import datetime
import data_preprocess as dp
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver(url='mongodb://132.72.81.248/netflow_roman', db_name='netflow_roman'))
conf = ConfigParser()
conf.read('config.ini')


@ex.config
def my_config():
    folder_name = conf.get('Paths', 'output_folder_name')
    data_file = conf.get('Paths', 'data_file')
    data_path = conf.get('Paths', 'data_path')
    output_path = conf.get('Paths', 'output_path')
    lstm_units = None
    optimizer = None
    loss = None
    epochs = None
    batch_size = None
    n_steps = None
    slide_len = None
    repetitions = None
    n_features = None
    n_steps_out = None
    l_rate = None
    timestamp = None
    rep = None
    csv_logger = None
    overflow_thresh = None
    lstm_layers = None
    use_mini_batches = None


@ex.main
def handle_netflow(data_path, data_file, folder_name, output_path, lstm_units, optimizer, loss, epochs,
                   batch_size, n_steps, n_features, slide_len, timestamp, rep, csv_logger, l_rate, n_steps_out,
                   overflow_thresh, lstm_layers, use_mini_batches):
    if not os.path.exists(output_path + folder_name + '//' + str(int(timestamp))):
        os.mkdir(output_path + folder_name + '//' + str(int(timestamp)))
    X, y, dates_X, dates_y, threshold, all_handovers = dp.preprocess_handover_regression_data([data_path + data_file], n_steps,
                                                                                  n_steps_out, slide_len, 2,
                                                                                  overflow_thresh)
    vols = {}
    lm.multistep_parallel_multivariate_lstm(batch_size, epochs, folder_name, loss, lstm_units, n_features, n_steps,
                                            optimizer, output_path, rep, slide_len, timestamp, vols, csv_logger, l_rate,
                                            n_steps_out, X, y, dates_X, dates_y, ex, threshold, lstm_layers,
                                            use_mini_batches, all_handovers)


def main(folder_name, data_file, data_path, output_path, lstm_units_list, optimizers, losses, epochs_list,
         batch_sizes, n_steps_list, n_features_list, slide_lens, repetitions, n_steps_out, l_rates,
         overflow_thresholds, lstm_layers, use_mini_batches):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not os.path.exists(output_path + folder_name):
        os.mkdir(output_path + folder_name)
    prod = product(lstm_units_list, optimizers, losses, epochs_list, batch_sizes, n_steps_list,
                   n_features_list, slide_lens, l_rates, n_steps_out, overflow_thresholds, lstm_layers)
    for lstm_units, optimizer, loss, epochs, batch_size, n_steps, n_features, slide_len, l_rate, \
        n_steps_out, overflow_thresh, n_lstm_layers in prod:
        timestamp = datetime.timestamp(datetime.now())
        for rep in range(repetitions):
            ex.run(config_updates={'lstm_units': lstm_units, 'optimizer': optimizer, 'loss': loss, 'epochs': epochs,
                                   'batch_size': batch_size, 'n_steps': n_steps, 'n_features': n_features,
                                   'slide_len': slide_len, 'l_rate': l_rate, 'n_steps_out': n_steps_out,
                                   'timestamp': timestamp, 'rep': rep, 'overflow_thresh': overflow_thresh,
                                   'lstm_layers': n_lstm_layers, 'use_mini_batches': use_mini_batches})
        lm.update_results_file(batch_size, epochs, folder_name, l_rate, loss, lstm_units, n_features,
                               n_steps, optimizer, output_path, repetitions, slide_len, timestamp, n_steps_out)


if __name__ == '__main__':
    main(conf.get('Paths', 'output_folder_name'),
         conf.get('Paths', 'data_file'),
         conf.get('Paths', 'data_path'),
         conf.get('Paths', 'output_path'),
         json.loads(conf.get('LSTM', 'lstm_units')),
         json.loads(conf.get('LSTM', 'optimizer')),
         json.loads(conf.get('LSTM', 'loss')),
         json.loads(conf.get('LSTM', 'epochs')),
         json.loads(conf.get('LSTM', 'batch_size')),
         json.loads(conf.get('LSTM', 'n_steps')),
         json.loads(conf.get('LSTM', 'n_features')),
         json.loads(conf.get('LSTM', 'slide_len')),
         json.loads(conf.get('LSTM', 'repetitions')),
         json.loads(conf.get('LSTM', 'look_forward')),
         json.loads(conf.get('LSTM', 'l_rates')),
         json.loads(conf.get('LSTM', 'overflow_threshold')),
         json.loads(conf.get('LSTM', 'lstm_layers')),
         conf.get('LSTM', 'use_mini_batches'))
