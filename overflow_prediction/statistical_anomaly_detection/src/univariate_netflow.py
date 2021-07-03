import json
import os
from configparser import ConfigParser
import lstm_model as lm
from itertools import product
from datetime import datetime
import pandas as pd
from sacred import Experiment


def handle_netflow(data_path, data_file, folder_name, output_path, lstm_units, optimizer, loss, epochs,
                   batch_size, n_steps, n_features, slide_len, timestamp, rep, csv_logger, l_rate, n_steps_out):
    if not os.path.exists(output_path + folder_name + '//' + str(int(timestamp))):
        os.mkdir(output_path + folder_name + '//' + str(int(timestamp)))
    df = pd.read_csv(data_path + data_file)
    vols = {}
    for index, row in df.iterrows():
        vols[row['id']] = [row['ts'], row['vol']]
    lm.univariate_lstm(batch_size, epochs, folder_name, loss, lstm_units, n_features, n_steps, optimizer,
                       output_path, rep, slide_len, timestamp, vols, csv_logger, l_rate, n_steps_out)


def main(folder_name, data_file, data_path, output_path, lstm_units_list, optimizers, losses, epochs_list,
         batch_sizes, n_steps_list, n_features_list, slide_lens, repetitions, n_steps_out, l_rates):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not os.path.exists(output_path + folder_name):
        os.mkdir(output_path + folder_name)
    prod = product(lstm_units_list, optimizers, losses, epochs_list, batch_sizes, n_steps_list,
                   n_features_list, slide_lens, l_rates, n_steps_out)
    for lstm_units, optimizer, loss, epochs, batch_size, n_steps, n_features, slide_len, l_rate, \
        n_steps_out in prod:
        timestamp = datetime.timestamp(datetime.now())
        for rep in range(repetitions):
            # csv_logger = CSVLogger(
            #     output_path + folder_name + '//' + str(int(timestamp)) + '/log_rep' + str(rep + 1) + '.csv',
            #     append=True, separator=';')
            handle_netflow(data_path, data_file, folder_name, output_path, lstm_units, optimizer, loss,
                           epochs,
                           batch_size, n_steps, n_features, slide_len, timestamp, rep, None, l_rate, n_steps_out)
        lm.update_results_file(batch_size, epochs, folder_name, l_rate, loss, lstm_units, n_features,
                               n_steps, optimizer, output_path, repetitions, slide_len, timestamp, n_steps_out)


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
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
         json.loads(conf.get('LSTM', 'l_rates')))
