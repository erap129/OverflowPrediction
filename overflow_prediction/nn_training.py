import time
import torch
from torch.autograd import Variable
from fold_utils import MTS_evaluate
import math
import torch.optim as optim

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()


def MTS_train(model, data, criterion, optim, epochs, batch_size):
    print('begin training');
    model.cuda()
    best_val = math.inf
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(data, data.train[0], data.train[1], model, criterion, optim, batch_size)
        val_loss, val_rae, val_corr = MTS_evaluate(data, data.valid[0], data.valid[1], model, batch_size);
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            torch.save(model.state_dict(), 'best_model.th')
            best_val = val_loss
        if epoch % 5 == 0:
            test_acc, test_rae, test_corr = MTS_evaluate(data, data.test[0], data.test[1], model, batch_size);
            print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    model.load_state_dict(torch.load('best_model.th'))


def train_epoch(data, X, Y, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        X = X.permute(0, 2, 1)
        X = X[:, :, :, None]
        model.zero_grad();
        output = model(X);
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.data;
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples


def get_batches(inputs, targets, batch_size, device, shuffle=True):
    length = len(inputs)
    if shuffle:
        index = torch.randperm(length,device=device)
    else:
        index = torch.as_tensor(range(length),device=device,dtype=torch.long)
    start_idx = 0
    while (start_idx < length):
        end_idx = min(length, start_idx + batch_size)
        excerpt = index[start_idx:end_idx]
        X = inputs[excerpt]
        Y = targets[excerpt]
        yield Variable(X), Variable(Y)
        start_idx += batch_size