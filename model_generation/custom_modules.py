import torch

from torch import nn


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


class AveragingModule(nn.Module):
    def forward(self, inputs):
        return torch.mean(inputs, dim=0)


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_neurons, n_networks, true_avg=False):
        super(LinearWeightedAvg, self).__init__()
        self.weight_inputs = []
        for network_idx in range(n_networks):
            self.weight_inputs.append(nn.Parameter(torch.randn(1, n_neurons, 1, 1).cuda()))
            init.xavier_uniform_(self.weight_inputs[-1], gain=1)
        if true_avg:
            for network_idx in range(n_networks):
                self.weight_inputs[network_idx].data = torch.tensor([[0.5 for i in range(n_neurons)]]).view((1, n_neurons, 1, 1)).cuda()
        self.weight_inputs = nn.ParameterList(self.weight_inputs)

    def forward(self, *inputs):
        res = 0
        for inp_idx, input in enumerate(inputs):
            res += input * self.weight_inputs[inp_idx]
        return res


class _squeeze_final_output(nn.Module):
    def __init__(self):
        super(_squeeze_final_output, self).__init__()

    def forward(self, x):
        assert x.size()[3] == 1
        x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
        return x


class _transpose(nn.Module):
    def __init__(self, shape):
        super(_transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)


class AveragingEnsemble(nn.Module):
    def __init__(self, models):
        super(AveragingEnsemble, self).__init__()
        self.avg_layer = LinearWeightedAvg(globals.get('n_classes'), globals.get('ensemble_size'), true_avg=True)
        self.models = models
        self.softmax = nn.Softmax()
        self.flatten = _squeeze_final_output()

    def forward(self, input):
        outputs = []
        for model in self.models:
            outputs.append(model(input))
        avg_output = self.avg_layer(*outputs)
        softmaxed = self.softmax(avg_output)
        return self.flatten(softmaxed)