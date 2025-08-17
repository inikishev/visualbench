# pylint: disable = not-callable
import random

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceFocalLoss
from torch.utils.data import Dataset

from ...utils import CUDA_IF_AVAILABLE
from .dataset import DatasetBenchmark


class _SyntheticSegmentation1D(Dataset):
    def __init__(self, num_samples=4_000, seq_length=32, num_classes=5, seed=0, device = CUDA_IF_AVAILABLE):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.seed = seed
        np.random.seed(seed)

        self.device = device
        self._generate_dataset()

    def _generate_dataset(self):
        data = []; labels = []
        for _ in range(self.num_samples):
            x, y = self._create_sample()
            data.append(torch.as_tensor(x, dtype = torch.float32))
            labels.append(torch.as_tensor(y, dtype = torch.int64))

        self.data = torch.stack(data).unsqueeze(1).to(self.device) # n_samples, 1, length
        self.labels = F.one_hot(torch.stack(labels).to(self.device)).moveaxis(-1,-2) # n_samples, n_classes, length

    def _create_sample(self):
        # Randomly partition sequence into segments
        num_segments = np.random.randint(1, 8)
        segment_lengths = self._random_partition(num_segments)

        # Generate segments with different patterns
        signal = np.zeros(self.seq_length)
        labels = np.zeros(self.seq_length, dtype=int)
        pos = 0

        for i, length in enumerate(segment_lengths):
            class_id = np.random.randint(self.num_classes)
            seg_signal = self._generate_segment(length, class_id)

            signal[pos:pos+length] = seg_signal
            labels[pos:pos+length] = class_id
            pos += length

        # Add global noise and normalize
        signal += np.random.normal(0, 0.1, self.seq_length)
        signal = (signal - np.mean(signal)) / np.std(signal)

        return signal, labels

    def _random_partition(self, num_segments):
        base = 3
        remaining = self.seq_length - base * num_segments
        add = np.random.multinomial(remaining, np.ones(num_segments)/num_segments)
        return base + add

    def _generate_segment(self, length, class_id):
        if class_id == 0:    # high-frequency sine wave
            cycles = np.random.uniform(1.5, 2.5)
            phase = np.random.uniform(0, 2*np.pi)
            return np.sin(2 * np.pi * cycles * np.linspace(0, 1, length) + phase)

        elif class_id == 1:  # low-frequency sine wave
            cycles = np.random.uniform(0.3, 0.7)
            phase = np.random.uniform(0, 2*np.pi)
            return np.sin(2 * np.pi * cycles * np.linspace(0, 1, length) + phase)

        elif class_id == 2:  # square wave
            phase = np.random.uniform(0, 1)
            duty = np.random.uniform(0.3, 0.7)
            t = np.linspace(0, 1, length) + phase
            return np.where((t % 1) < duty, 1, -1)

        elif class_id == 3:  # sawtooth wave
            direction = np.random.choice([-1, 1])
            return direction * np.linspace(-1, 1, length)

        elif class_id == 4:  # noisy spike
            seg = np.random.normal(0, 0.2, length)
            spike_pos = np.random.randint(0, length)
            seg[spike_pos] = np.random.uniform(1.5, 2.5)
            return seg

        elif class_id == 5: # slope
            min = np.random.uniform(-2, 1)
            max = np.random.uniform(0.5, 2)
            return np.linspace(min, max, length)

        elif class_id == 6: # normal noise
            loc = np.random.uniform(-1, 1)
            sigma = np.random.uniform(0.5, 2)
            return np.random.normal(loc, sigma, size=length)

        elif class_id == 7: # flat area
            return np.full(length, np.random.uniform(-2, 2))

        elif class_id == 8: # uniform noise
            min = np.random.uniform(-2, 1)
            max = np.random.uniform(0.5, 2)
            return np.random.uniform(min, min+max, size=length)

        elif class_id == 9: # binomial noise
            n = np.random.randint(1,4)
            p = np.random.uniform(0.25, 0.75)
            return np.random.binomial(n,p,size=length)

        else:
            raise ValueError("Invalid class ID")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def dataloader(self, batch_size, shuffle):
        """returns a TensorDataloader"""
        from ...utils.light_dataloader import TensorDataLoader
        return TensorDataLoader((self.data, self.labels), batch_size=batch_size, shuffle = shuffle)

class SynthSeg1d(DatasetBenchmark):
    """
    input - (B, 1, seq_length)

    output - (B, num_classes, seq_length)
    """
    def __init__(
        self,
        model,
        criterion=DiceFocalLoss(softmax=True,),
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        test_split=0.8,
        num_samples=10_000,
        seq_length=32,
        num_classes=10,
        seed=0,
        device=CUDA_IF_AVAILABLE,
    ):
        ds = _SyntheticSegmentation1D(num_samples = num_samples, seq_length=seq_length,num_classes=num_classes,seed=seed,device=device)
        x, y = ds.data, ds.labels

        super().__init__(
            data_train = (x, y),
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size = test_batch_size,
            test_split=test_split,
            dtypes = (torch.float32, torch.uint64),
            data_device = device,
        )


if __name__ == "__main__":
    dataset = _SyntheticSegmentation1D(num_samples=1000)
    print(f"Dataset size: {len(dataset)}")
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label shape: {label.shape}")