import torch
import pandas as pd
from numpy import array, math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from tqdm import tqdm
import os
from sacred import Experiment
ex = Experiment('Detection')


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
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


class Autoencoder(nn.Module):
    """
    The number of channels in the input and filter should be the same.
    The first element of the output is the sum  of element-wise product of the input*filter in each channel.
    num of out channels = num of filters.
    """
    def __init__(self, num_of_devices):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_of_devices, 5, 2),  #
            nn.ReLU(True),                    #
            nn.Conv1d(5, 10, 2),  #
            nn.ReLU(True),
            # nn.MaxPool1d(2, padding="same"),  #
            nn.Conv1d(10, 5, 2),  #
            nn.ReLU(True),
            # nn.MaxPool2d(2)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(5, 10, 2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose1d(10, 5, 2),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose1d(5, num_of_devices, 2),  # b, 1, 28, 28
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

@ex.config
def config():
    NOISE_RATIO=0.1
    dataset_netflow_paths_pr = './data/snmp/NEWper_router_interface_no_leading_zeros_gt0.9zeros/'
    routers_models_path = './results/snmp_results/routers_models'
    # dataset_netflow_path_pr = './data/netflow/all_routers_flow.csv'


@ex.capture
def find_model_path(r_file, routers_models_path):
    for model_file in os.listdir(routers_models_path):
        if r_file in model_file:
            return os.path.join(routers_models_path, model_file)
    return None


def find_rw_anomalies(ad_df):
    # ad_path = 'C:\\Users\\Administrator\\PycharmProjects\\netflowinsights\\statistical_anomaly_detection\\' \
    #           'cnn_denoising_autoencoders\\netflow_results\\v1_routers\\tableau\AD_router_VIE-SB5_95_gt0.75zeros.csv_v1.csv'
    # ad_df = pd.read_csv(ad_path)

    devices = ad_df['device'].unique()
    df_new_anomalies_list = []
    for device in devices:
        print(device)
        try:
            dev_df = ad_df[ad_df['device'] == device].copy()
            r = dev_df['abs_diff'].rolling(window=24)  # Create a rolling object (no computation yet)
            mps = r.mean() + 2 * r.std()
            dev_df['pd_anomaly'] = dev_df['abs_diff'] > mps
            # dev_df['abs_diff'] > dev_df['input_vol'].mean() + 2 * dev_df['input_vol'].std()
            df_new_anomalies_list.append(dev_df)
        except:
            print ('rw exception')
            continue
    df_rw_ad = pd.concat(df_new_anomalies_list)
    return df_rw_ad


def find_general_anomalies(df):
    devices = df['device'].unique()
    df_new_anomalies_list = []
    for device in devices:
        dev_df = df[df['device'] == device]
        diff_mean = np.mean(dev_df['abs_diff'])
        diff_std = np.std(dev_df['abs_diff'])
        dev_df['gen_pd_anomaly'] = \
            dev_df['abs_diff'] > diff_mean + 2 * diff_std
        df_new_anomalies_list.append(dev_df.copy())
    df_gen_ad = pd.concat(df_new_anomalies_list)
    return df_gen_ad

@ex.automain
def main(dataset_netflow_paths_pr):
    all_results = []
    for r_file in os.listdir(dataset_netflow_paths_pr):
        print(r_file)
        dataset_netflow_path_pr = os.path.join(dataset_netflow_paths_pr, r_file)
        model_path = find_model_path(r_file)
        print(model_path)
        if not model_path:
            break
        # dataset_netflow_path_pr = './data/netflow/as_flow_vol_final.csv'
        pd_df = pd.read_csv(dataset_netflow_path_pr, index_col=0).fillna(0)
        time_steps = pd_df.columns
        devs_id = pd_df.index
        my_data = pd_df.values.transpose()
        my_data = my_data / 2 ** 30

        steps = pd_df.shape[1]-1
        batch_size = pd_df.shape[1]
        ds = split_sequence(my_data, steps)[0]
        ds = ds.swapaxes(1, 2)
        num_of_devices = ds.shape[1]

        # Load the data to tensor
        trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = Autoencoder(num_of_devices).cpu()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # model = torch.load(f'./conv_autoencoder.pth')

        for i, data in enumerate(tqdm(trainloader, 0)):
            # data = data.reshape(data.shape[0], 1, data.shape[1])
            input = data
            input = Variable(input)
            output = model(input.float())
            abs_diff = abs(input.float() - output.float())
            ts = time_steps[i:i+steps]
            k=0
            for i_id, i in enumerate(devs_id):
            # for i in range(num_of_devices):
            #     for k in range(batch_size):
                for j in range(steps):
                    epoch_res = {'device': i, 'ts_date': ts[j],
                                 'input_vol': input.data[k][i_id][j].item(),
                                 'output_vol': output.data[k][i_id][j].item(),
                                 'abs_diff': abs_diff.data[k][i_id][j].item()}
                    all_results.append(epoch_res.copy())
                    # df_results = pd.DataFrame(all_results)
                    # df_results.to_csv(f'./netflow_results/v1/anomaly_det_tableau/as_flow_res_bs{batch_size}_'
                                      # f'noise{NOISE_RATIO}_ts_{ts[j]}.csv', index=False)

    df_results = pd.DataFrame(all_results)
    df_results = find_rw_anomalies(df_results)
    df_results = find_general_anomalies(df_results)

    df_results.to_csv(f'./results/snmp_results/AD/AD_snmp_rw_gen_router_gt0.9_no_stend_zeros.csv', index=False)



