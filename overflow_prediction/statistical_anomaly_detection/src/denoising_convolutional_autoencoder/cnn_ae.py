
import torch
import torchvision
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from numpy import genfromtxt
from sacred import Experiment
ex = Experiment('Learning')


import time

from tqdm import tqdm


class Autoencoder(nn.Module):
    """
    The number of channels in the input and filter should be the same.
    The first element of the output is the sum  of element-wise product of the input*filter in each channel.
    num of out channels = num of filters.
    """
    def __init__(self, num_of_devices):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_of_devices, 5, 2),  # 5 - # of filters
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


@ex.capture
def noise_input(data, devs_id, NOISE_RATIO, steps):
    noise_data = data.clone()
    for id_in_batch in range(data.shape[0]):
        noise_steps = np.random.choice(range(steps), size=int(steps * NOISE_RATIO), replace=False)
        b_mean = np.mean(data[id_in_batch].numpy())
        b_std = np.std(data[id_in_batch].numpy())
        for dev_id in range(len(devs_id)):
            noise_data[id_in_batch][dev_id][noise_steps] = np.random.normal(b_mean, b_std)
    return noise_data


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


@ex.capture
def is_done(r_file, results_path):
    for done_f in os.listdir(results_path):
        if r_file in done_f:
            return True


@ex.config
def config():
    start_time = time.time()
    scaler = MinMaxScaler()
    dataset_netflow_paths_pr = './data/snmp/NEWper_router_interface_no_leading_zeros_gt0.9zeros/'
    results_path = './results/snmp_results/training/'
    # dataset_netflow_path_pr = './data/netflow/all_routers_flow.csv'
    # parameters
    NOISE_RATIO = 0
    num_epochs = 30
    batch_sizes = [8]
    learning_rates = [1e-4]
    steps = 25
    loss_type = 'MSE'
    optimizer_type = 'Adam'
    normalize_opts = [False]
    multivariate = 'False'


@ex.named_config
def variant1():
    normalize_opts = [True]


@ex.named_config
def variant2():
    NOISE_RATIO = 0.4


@ex.named_config
def variant3():
    normalize_opts = [True]
    NOISE_RATIO = 0.4


@ex.automain
def main(start_time, scaler, dataset_netflow_paths_pr, NOISE_RATIO, num_epochs, batch_sizes, learning_rates, steps, loss_type, optimizer_type, normalize_opts, multivariate):
    for r_file in  os.listdir(dataset_netflow_paths_pr):
        if is_done(r_file):
            continue
        print(r_file)
        dataset_netflow_path_pr = os.path.join(dataset_netflow_paths_pr, r_file)
        pd_df = pd.read_csv(dataset_netflow_path_pr, index_col=0).fillna(0)
        time_steps = pd_df.columns
        devs_id = pd_df.index
        my_data = pd_df.values.transpose()
        my_data = my_data / 2 ** 30
        #
        # dataset_netflow_path_pr = './data/netflow/as_flow_sorted_vol.csv'
        # my_data = genfromtxt(dataset_netflow_path_pr, delimiter=',').transpose()

        my_data_scaled = scaler.fit_transform(my_data)

        # START WITH THE EXPERIMENTS
        all_results = []
        for batch_size in batch_sizes:
            print(f'batch size: {batch_size}')
            for learning_rate in learning_rates:
                print(f'learning rate: {learning_rate}')
                for normalize in normalize_opts:
                    print(f'normalize: {normalize}')

                    # Prepare data
                    if normalize:
                        ds = split_sequence(my_data_scaled, steps)[0]
                    else:
                        ds = split_sequence(my_data, steps)[0]
                    ds = ds.swapaxes(1, 2)
                    num_of_devices = len(ds[1])

                    # Load the data to tensor
                    trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

                    # Define Autoencoder model
                    model = Autoencoder(num_of_devices).cpu()

                    if loss_type is 'MSE':
                        criterion = nn.MSELoss()
                    if optimizer_type is 'Adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

                    for epoch in range(num_epochs):
                        # epoch_res = {}
                        current_loss = 0
                        for i, data in enumerate(trainloader, 0):
                            # data = data.reshape(data.shape[0], 1, data.shape[1])
                            input = data
                            noise_in = noise_input(input, devs_id)
                            input = Variable(input)
                            noise_in = Variable(noise_in)
                            model.zero_grad()

                            output = model(noise_in.float())
                            # loss = criterion(output, input.float())
                            loss = criterion(output, input.float())

                            loss.backward()
                            optimizer.step()

                            loss = loss.data
                            current_loss += loss

                            if i == 0:
                                for i_id, i in enumerate(devs_id):
                                # for i in range(num_of_devices):
                                    for k in range(batch_size):
                                        for j in range(steps):
                                            epoch_res = {'epoch': epoch, 'loss': loss.item(), 'batch_size': batch_size,
                                                         'learning_rate': learning_rate, 'multivariate': multivariate,
                                                         'time_window': steps, 'loss_type': loss_type, 'optimizer_type': optimizer_type,
                                                         'normalize': normalize, 'device': i, 'id_in_batch': k, 'ts': j,
                                                         'input_vol': input.data[k][i_id][j].item(),
                                                         'input_noise': noise_in.data[k][i_id][j].item(),
                                                         'output_vol': output.data[k][i_id][j].item(),
                                                         'noise': NOISE_RATIO}
                                            all_results.append(epoch_res.copy())

                        current_loss = 0
                        # ===================log========================
                        print('epoch [{}/{}], loss:{:.4f}'
                            .format(epoch+1, num_epochs, loss.data))
                    # df_results = pd.DataFrame(all_results)
                    # df_results.to_csv(f'./netflow_results/tableau/noise_as_flow_res_bs{batch_size}_norm-{normalize}_'
                    #                   f'lr{learning_rate}_steps{steps}_loss{loss_type}_multi{multivariate}.csv',
                    #                   index=False)
                    # all_results = []

        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f'./results/snmp_results/training/{r_file}_routers_res_bs{batch_size}_norm-{normalize}_'
                                                f'lr{learning_rate}_steps{steps}_loss{loss_type}_'
                        f'noise{NOISE_RATIO}_multi{multivariate}_gt0.9zeros.csv', index=False)
        torch.save(model.state_dict(), f'./results/snmp_results/routers_models/{r_file}_router_cnn_dn_ae_{NOISE_RATIO}'
        f'_gt0.9zeros.pth')
        print("--- %s seconds ---" % (time.time() - start_time))
