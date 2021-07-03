import torch
from torch import nn
import numpy as np
import data_preprocess as dp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
import torch.nn.functional as F


class UnivariateLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, seq_len, output_dim=1, num_layers=2):
        super(UnivariateLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.input_emd_dim = 20

        self.linear2 = nn.Linear(32, self.input_emd_dim)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self, b_size=None):
        # This is what we'll initialise our hidden state as
        if b_size is None:
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda())

    def forward(self, input, b_size=None):
        # input_emd = self.linear2(input)

        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, _ = self.lstm(input.view(len(input), self.batch_size, -1))
        # lstm_out, self.hidden = self.lstm(input_emd, self.hidden)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if b_size is None:
            y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        else:
            y_pred = self.linear(lstm_out[-1].view(b_size, -1))
        return y_pred.view(-1)

    def predict(self, X_test):
        y_hat = self.forward(X_test)
        return y_hat.tolist()


# Here we define our model as a class
class MultivariateLSTM(nn.Module):

    def __init__(self, n_features, hidden_dim, batch_size, n_steps, output_dim=1,
                 num_layers=2, eegnas=False):
        super(MultivariateLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim  # number of hidden states
        self.batch_size = batch_size
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.n_steps = n_steps
        self.eegnas = eegnas

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.n_steps, output_dim)

    def init_hidden(self, b_size=None):
        # This is what we'll initialise our hidden state as
        if b_size is None:
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda())

    def forward(self, input, b_size=None):
        if self.eegnas:
            input = input.squeeze(dim=3)
            input = input.permute(0, 2, 1)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if b_size is None:
            y_pred = self.linear(lstm_out.contiguous().view(input.shape[0], -1))
        else:
            y_pred = self.linear(lstm_out.contiguous().view(b_size, -1))
        return y_pred

    def predict(self, X_test):
        y_hat = self.forward(X_test)
        return y_hat.tolist()

# Here we define our model as a class
class LSTMMulticlassClassification(nn.Module):

    def __init__(self, n_features, hidden_dim, batch_size, n_steps, output_dim=1,
                 num_layers=2, eegnas=False):
        super(LSTMMulticlassClassification, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim  # number of hidden states
        self.batch_size = batch_size
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.n_steps = n_steps
        self.eegnas = eegnas

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.n_steps, output_dim * (self.n_features + 1))

    def init_hidden(self, b_size=None):
        # This is what we'll initialise our hidden state as
        if b_size is None:
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda())

    def forward(self, input, b_size=None):
        if self.eegnas:
            input = input.squeeze(dim=3)
            input = input.permute(0, 2, 1)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.cuda())

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if self.eegnas:
            y_pred = self.linear(lstm_out.contiguous().view(input.shape[0], -1))
        elif b_size is None:
            y_pred = self.linear(lstm_out.contiguous().view(self.batch_size, -1))
        else:
            y_pred = self.linear(lstm_out.contiguous().view(b_size, -1))
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred

    def predict(self, X_test):
        y_hat = self.forward(X_test)
        return y_hat.tolist()


class MultivariateMultistepLSTM(nn.Module):

    def __init__(self, n_features, hidden_dim, batch_size, n_steps, num_layers=3, output_dim=5, eegnas=False):
        super(MultivariateMultistepLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim  # number of hidden states
        self.batch_size = batch_size
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.n_steps = n_steps
        self.eegnas = eegnas

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.n_steps, output_dim)

    def init_hidden(self, b_size=None):
        # This is what we'll initialise our hidden state as
        if b_size is None:
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda())

    def forward(self, input, b_size=None):
        if self.eegnas:
            input = input.squeeze(dim=3)
            input = input.permute(0, 2, 1)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if self.eegnas:
            y_pred = self.linear(lstm_out.contiguous())
        elif b_size is None:
            y_pred = self.linear(lstm_out.contiguous().view(self.batch_size, -1))
        else:
            y_pred = self.linear(lstm_out.contiguous().view(b_size, -1))
        return y_pred

    def predict(self, X_test):
        y_hat = self.forward(X_test)
        return y_hat.tolist()


class MultivariateParallelMultistepLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim, batch_size, n_steps, num_layers=3, output_dim=5, eegnas=False):
        super(MultivariateParallelMultistepLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim  # number of hidden states
        self.batch_size = batch_size
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.n_steps = n_steps
        self.output_dim = output_dim
        self.eegnas = eegnas

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.n_steps, output_dim * self.n_features)

    def init_hidden(self, b_size=None):
        # This is what we'll initialise our hidden state as
        if b_size is None:
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda())

    def forward(self, input, b_size=None):
        if self.eegnas:
            input = input.squeeze(dim=3)
            input = input.permute(0, 2, 1)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if self.eegnas:
            y_pred = self.linear(lstm_out.contiguous().view(input.shape[0], -1))
        elif b_size is None:
            y_pred = self.linear(lstm_out.contiguous().view(self.batch_size, -1)).view(self.batch_size, self.output_dim,
                                                                                       self.n_features)
        else:
            y_pred = self.linear(lstm_out.contiguous().view(b_size, -1)).view(b_size, self.output_dim,
                                                                              self.n_features)
        return y_pred

    def predict(self, X_test):
        y_hat = self.forward(X_test)
        return y_hat.tolist()


class LSTMClassifier(nn.Module):

    def __init__(self, n_features, hidden_dim, batch_size, n_steps, num_layers, output_dim=2):
        super(LSTMClassifier, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim  # number of hidden states
        self.batch_size = batch_size
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.n_steps = n_steps

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.n_steps, output_dim)

    def init_hidden(self, b_size=None):
        # This is what we'll initialise our hidden state as
        if b_size is None:
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda())

    def forward(self, input, b_size=None):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if b_size is None:
            y_pred = self.linear(lstm_out.contiguous().view(self.batch_size, -1))
        else:
            y_pred = self.linear(lstm_out.contiguous().view(b_size, -1))
        return y_pred

    def predict(self, X_test):
        pred = F.softmax(self.forward(X_test))
        ans = []
        # Pick the class with maximum weight
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return ans, [a[1].item() for a in pred]


def get_loss_fun(loss_fn):
    return {
        'mse': torch.nn.MSELoss(size_average=False),
        'crossentropy': torch.nn.CrossEntropyLoss(size_average=False)
    }[loss_fn]


def train_model(X_train, y_train, model, num_epochs, loss_fn, optimizer, learning_rate, batch_size, n_steps, ex, idx):
    loss_fn = get_loss_fun(loss_fn)
    model.cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    #####################
    # Train model
    #####################

    hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()

        # Forward pass
        y_pred = model(X_train)
        try:
            loss = loss_fn(y_pred, y_train)
        except:
            loss = loss_fn(y_pred, y_train.long())

        print("Epoch ", t, "Loss: ", loss.item())
        ex.log_scalar('loss_fold_{}'.format(idx), loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
    return hist


# model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
def train_model_with_batches(X_train, y_train, model, num_epochs, loss_fn, optimizer, learning_rate, batch_size,
                             n_steps, ex, idx):
    loss_fn = get_loss_fun(loss_fn)
    model.cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #####################
    # Train model
    #####################
    X_train = DataLoader(X_train, batch_size=batch_size, shuffle=False)
    y_train = DataLoader(y_train, batch_size=batch_size, shuffle=False)
    hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        y_pred_list = []
        loss_list = []

        for batch_x, batch_y in zip(X_train, y_train):
            # Clear stored gradient
            model.zero_grad()

            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            model.hidden = model.init_hidden(len(batch_y))
            # batch_x = batch_x.view([n_steps, -1, 1]).cuda()
            batch_x = batch_x.cuda()
            # batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            # Forward pass
            y_pred = model(batch_x, len(batch_y))

            try:
                loss = loss_fn(y_pred, batch_y)
            except:
                loss = loss_fn(y_pred, batch_y.long())
            y_pred_list.extend(y_pred.tolist())
            loss_list.append(loss.item())

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()
        hist[t] = np.mean(loss_list)
        ex.log_scalar('loss_fold_{}'.format(idx), np.mean(loss_list))
        print(f"Loss: {np.mean(loss_list)}, Epoch: {t}")
    return hist


def univariate_model(data_sample, n_steps, slide_len, lstm_units, num_epochs, loss_fn, optimizer, learning_rate,
                     batch_size, timestamps, n_features):
    sample_list, y = dp.split_univariate_sequence(data_sample, n_steps, slide_len)
    X_train, X_test, y_train, y_test = train_test_split(sample_list, y, test_size=0.33, shuffle=False)
    len_X_train = len(X_train)
    len_X_test = len(X_test)
    _, test_timestamps = train_test_split(timestamps[n_steps:], test_size=0.33, shuffle=False)
    X_train = torch.from_numpy(X_train.transpose()).type(torch.Tensor).view([n_steps, -1, 1])
    X_test = torch.from_numpy(X_test.transpose()).type(torch.Tensor).view([n_steps, -1, 1])
    y_train = torch.from_numpy(y_train).type(torch.Tensor).view(-1)

    hidden_dimension = lstm_units
    model = UnivariateLSTM(n_features, hidden_dimension, len_X_train, n_steps)
    loss_arr = train_model(X_train, y_train, model, num_epochs, loss_fn, optimizer, learning_rate,
                           len_X_train, n_steps)
    model.batch_size = len_X_test
    predictions = model.predict(X_test.cuda())
    error = np.array(predictions) - y_test
    # error = np.array(predictions) - y_test[n_steps::slide_len]

    print("Maximum reconstruction error was %.1f" % error.max())

    return predictions, np.array(error), loss_arr, y_test, test_timestamps


def multivariate_model(X, y, n_steps, slide_len, lstm_units, num_epochs, loss_fn, optimizer, learning_rate,
                       dates_X, dates_y, LSTM_Model, ex, lstm_layers, batch_size, use_mb):
    # sample_list, y = dp.split_multuple_sequences(X, n_steps, slide_len)
    kf = KFold(n_splits=5, shuffle=False)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
    # _, test_timestamps = train_test_split(dates_y[n_steps:], test_size=0.33, shuffle=False)
    # _, test_timestamps = train_test_split(dates_y, test_size=0.33, shuffle=False)
    hidden_dimension = lstm_units
    n_features = X.shape[2]
    predictions, errors, losses, original_data, dates = [], [], [], [], []
    i = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = torch.from_numpy(X_train.astype(float)).float()
        X_test = torch.from_numpy(X_test.astype(float)).float()
        y_train = torch.from_numpy(y_train.astype(float)).float()

        model = LSTM_Model(n_features, hidden_dimension, len(X_train), n_steps, lstm_layers)

        if use_mb.lower() == 'true':
            loss_arr = train_model_with_batches(X_train, y_train, model, num_epochs, loss_fn, optimizer, learning_rate,
                                                batch_size, n_steps, ex, i)
        else:
            loss_arr = train_model(X_train, y_train, model, num_epochs, loss_fn, optimizer, learning_rate,
                                   len(X_train), n_steps, ex, i)
        i = i + 1
        losses.append(loss_arr)
        model.batch_size = len(X_test)
        y_hat = model_predict(X_test, model, batch_size, use_mb)
        predictions.append(y_hat)
        if len(y_hat) < len(y_test):
            errors.append(y_hat[0] - y_test)
        else:
            errors.append(y_hat - y_test)
        original_data.append(y_test)
        dates.append(dates_y[test_index])
    # error = np.array(predictions)[:len(predictions) - 1] - original_data[
    #                                                        n_steps + n_steps_out - 1:len(original_data) - 1:slide_len]

    return predictions, errors, losses, original_data, dates


def model_predict(X_test, model, batch_size, use_mb):
    if use_mb.lower() != 'true':
        return model.predict(X_test.cuda())
    result = []
    X_test = DataLoader(X_test, batch_size=batch_size, shuffle=False)
    for batch_x in X_test:
        model.batch_size = len(batch_x)
        result.extend(model.predict(batch_x.cuda()))
    return result
