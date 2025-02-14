"""
mnist1d from
https://github.com/greydanus/mnist1d
"""
import pickle
from collections import abc
from functools import partial
from urllib.request import urlopen

import torch
from torch import nn
from mnist1d.data import make_dataset
from myai import nn as mynn
from ..benchmark import Benchmark, make_dataset_from_tensor, sig
from ..utils import CUDA_IF_AVAILABLE


class ObjectView:
    """this is taken from mnist1d.utils (i added with)"""
    def __init__(self, d): self.__dict__ = d

def _load_frozen(path = None):
    """loads frozen mnist1d. if path is None downloads it."""
    if path is None:
        url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
        with urlopen(url) as f:
            data = pickle.load(f)

    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
    return data

def _process(data:dict, dtype, device):
    # data.keys()
    # >>> dict_keys(['x', 'x_test', 'y', 'y_test', 't', 'templates'])  # these are NumPy arrays
    # x is (4000, 40), x_test is (1000, 40)
    # y and y_test are (4000, ) and (1000, )
    x = torch.tensor(data['x'], dtype = dtype, device = device)
    x_test = torch.tensor(data['x_test'], dtype = dtype, device = device)
    y = torch.tensor(data['y'], dtype = torch.int64, device = device)
    y_test = torch.tensor(data['y_test'], dtype = torch.int64, device = device)
    return (x, y), (x_test, y_test)

def get_frozen_mnist1d(path = None, dtype = torch.float32, device = 'cuda'):
    data = _load_frozen(path)
    return _process(data, dtype, device)

def get_mnist1d( # pylint:disable = dangerous-default-value
    num_samples=5000,
    train_split=0.8,
    template_len=12,
    padding=[36, 60],
    scale_coeff=4,
    max_translation=48,
    corr_noise_scale=0.25,
    iid_noise_scale=2e-2,
    shear_scale=0.75,
    shuffle_seq=False,
    final_seq_length=40,
    seed=42,
    url="https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl",
    template=None,
    dtype=torch.float32,
    device: torch.types.Device = CUDA_IF_AVAILABLE,
):
    """loads frozen mnist1d. if path is None downloads it."""
    kwargs = locals().copy()
    template = kwargs.pop('template')
    dtype = kwargs.pop('dtype')
    device = kwargs.pop('device')
    data = make_dataset(ObjectView(kwargs), template)

    return _process(data, dtype, device)

class MNIST1D(Benchmark):
    def __init__( # pylint:disable = dangerous-default-value
        self,
        model: torch.nn.Module | sig,
        batch_size: int | None,
        test_batch_size: int | None = None,
        loss: abc.Callable | sig = torch.nn.CrossEntropyLoss(),
        num_samples = 5000,
        train_split = 0.8,
        template_len = 12,
        padding = [36,60],
        scale_coeff = 4,
        max_translation = 48,
        corr_noise_scale = 0.25,
        iid_noise_scale = 2e-2,
        shear_scale = 0.75,
        shuffle_seq = False,
        final_seq_length = 40,
        seed = 42,
        url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl',
        template = None,
        autoencoder = False,
        train_batch_tfms = None,
        test_batch_tfms = None,
        log_projections = True,
        dtype = torch.float32,
        device = CUDA_IF_AVAILABLE,
    ):
        train, test = get_mnist1d(
            num_samples=num_samples,
            train_split=train_split,
            template_len=template_len,
            padding=padding,
            scale_coeff=scale_coeff,
            max_translation=max_translation,
            corr_noise_scale=corr_noise_scale,
            iid_noise_scale=iid_noise_scale,
            shear_scale=shear_scale,
            shuffle_seq=shuffle_seq,
            final_seq_length=final_seq_length,
            seed=seed,
            url=url,
            template=template,
            dtype=dtype,
            device=device,
        )
        train_data, test_data = make_dataset_from_tensor(
            dataset = train,
            batch_size = batch_size,
            test_batch_size = test_batch_size,
            test_dataset = test,
            seed = seed,
        )

        super().__init__(
            train_data = train_data,
            test_data = test_data,
            train_batch_tfms = train_batch_tfms,
            test_batch_tfms = test_batch_tfms,
            log_projections = log_projections,
            seed = seed
        )

        self.model = self._save_signature(model, 'model')
        self.loss_fn = self._save_signature(loss, 'loss_fn')
        self.autoencoder = autoencoder
        self.to(device)
        self.model.train()

    def get_loss(self):
        inputs, targets = self.batch
        preds: torch.Tensor = self.model(inputs)
        if self.autoencoder: return self.loss_fn(preds, inputs)

        loss = self.loss_fn(preds, targets)
        accuracy = preds.argmax(1).eq_(targets).float().mean()
        return loss, {"accuracy": accuracy}

    def reset(self, model):
        super().reset()
        self.model = self._save_signature(model, 'model')
        self.model.train()


class SparseMNIST1D(MNIST1D):
    def __init__(
        self,
        model: torch.nn.Module | sig,
        batch_size: int | None,
        reconstruction_loss = torch.nn.functional.mse_loss,
        sparse_loss = torch.mean,
        sparse_weight = 1e-3,
        log_projections = True,
        dtype = torch.float32,
        device = CUDA_IF_AVAILABLE,
    ):
        super().__init__(model=model,batch_size=batch_size,loss=reconstruction_loss,log_projections=log_projections,dtype=dtype,device=device,autoencoder=True)
        self.sparse_loss_fn = sparse_loss
        self.sparse_weight = sparse_weight

    def get_loss(self):
        inputs, _ = self.batch
        preds, features = self.model(inputs)
        rec_loss = self.loss_fn(preds, inputs)
        sparse_loss = self.sparse_loss_fn(features)
        loss = rec_loss + sparse_loss*self.sparse_weight
        return loss, {"reconstruction loss": rec_loss, "sparse loss": sparse_loss}



def mnist1d_mlp(channels = (40, 80, 160, 80, 10)):
    torch.manual_seed(0)
    layers = []

    for i, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(torch.nn.Linear(cin, cout))
        if i != len(channels) - 2:
            layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)

def mnist1d_resnet(step = 8):
    from myai.nn.nets.resnet import ResNet18
    return ResNet18(1, step, 10, 1)



class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(2) # from (batch_size, 40) to (batch_size, 40, 1) otherwise known as (batch_size, seq_length, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)# hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # cell state

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0)) # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :] # last timestep's output (batch_size, hidden_size)
        out = self.fc(out) # (batch_size, num_classes)
        return out

class CNN(torch.nn.Module):
    def __init__(self, channels = (1, 32, 64, 128, 256), fixed_bn = False):
        super().__init__()
        self.enc = torch.nn.Sequential(
            *[mynn.ConvBlock(i, o, 2, 2, act='relu', norm='bn', ndim=1) for i, o in zip(channels[:-1], channels[1:])]
        )
        self.head = mynn.LinearBlock(channels[-1]*2, 10, flatten=True)
        if fixed_bn: torch.func.replace_all_batch_norm_modules_(self)


    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.enc(x)
        return self.head(x)


class Autoencoder(torch.nn.Module):
    def __init__(self, channels = (1, 16, 32, 64, 128, 256), act='relu',norm='bn', fixed_bn = False):
        super().__init__()
        channels = list(channels)
        self.enc = torch.nn.Sequential(
            *[mynn.ConvBlock(i, o, 2, 2, act=act, norm=norm, ndim=1) for i, o in zip(channels[:-1], channels[1:])]
        )

        channels.reverse()
        self.dec = torch.nn.Sequential(
            *[mynn.ConvTransposeBlock(i, o, 3, 2, act=act if n!=len(channels)-2 else None, norm=norm, ndim=1) for n, (i, o) in enumerate(zip(channels[:-1], channels[1:]))]
        )

        if fixed_bn: torch.func.replace_all_batch_norm_modules_(self)


    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
            sq = True
        else: sq = False
        l = x.shape[2]
        feats = self.enc(x)
        x = self.dec(feats)
        if sq: x = x[:,0]
        return x[:,:l], feats

MNIST1D_LogisticRegression = lambda device=CUDA_IF_AVAILABLE: MNIST1D(mnist1d_mlp([40, 10]), None, device=device)
"""fullbatch single layer perceptron i.e. logistic regression"""

MNIST1D_MLP_Minibatch = lambda layers = (40, 80, 160, 80, 10), device=CUDA_IF_AVAILABLE: MNIST1D(mnist1d_mlp(layers), 64, device=device)
"""minibatch multilayer perceptron"""

MNIST1D_ResNet_Minibatch = lambda channel_step = 8, device=CUDA_IF_AVAILABLE: MNIST1D(mnist1d_resnet(channel_step), 64, device=device)
"""minibatch resnet"""

MNIST1D_ResNet_Fullbatch = lambda channel_step = 8, device=CUDA_IF_AVAILABLE: MNIST1D(mnist1d_resnet(channel_step), None, device=device)
"""fullbatch resnet"""

MNIST1D_ResNet_Online = lambda channel_step = 8, device=CUDA_IF_AVAILABLE: MNIST1D(mnist1d_resnet(channel_step), 1, device=device)
"""online resnet"""

MNIST1D_LSTM_Minibatch = lambda hidden_size = 64, num_layers = 5, device=CUDA_IF_AVAILABLE: MNIST1D(
    LSTMClassifier(1, hidden_size, num_layers, 10), 64, device=device)
"""minibatch LSTM"""

MNIST1D_ConvNet_Minibatch = lambda channels = (1, 32, 64, 128, 256), fixed_bn = False, device = CUDA_IF_AVAILABLE: MNIST1D(
    CNN(channels, fixed_bn), 64, device=device)

MNIST1D_ConvNet_Fullbatch = lambda channels = (1, 32, 64, 128, 256), fixed_bn = False, device = CUDA_IF_AVAILABLE: MNIST1D(
    CNN(channels, fixed_bn), None, device=device)

MNIST1D_SparseAutoencoder = lambda channels = (1, 16, 32, 64, 128, 256), act='relu',norm='bn', fixed_bn = False, sparse_weight = 1e-3, device=CUDA_IF_AVAILABLE: SparseMNIST1D(
    Autoencoder(channels, act, norm, fixed_bn), 64, sparse_weight = sparse_weight, device = device)