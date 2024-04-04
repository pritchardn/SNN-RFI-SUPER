import numpy as np
import torch
from matplotlib import pyplot as plt
from rockpool.nn.combinators import Sequential
from rockpool.nn.modules import LinearTorch, ExpSynTorch, LIFTorch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm.autonotebook import tqdm

torch.manual_seed(0)
np.random.seed(0)
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


# - Define a dataset class implementing the indexing interface
class MultiClassRandomSinMapping:
    def __init__(
            self,
            num_classes: int = 2,
            sample_length: int = 100,
            input_channels: int = 50,
            target_channels: int = 2,
    ):
        # - Record task parameters
        self._num_classes = num_classes
        self._sample_length = sample_length

        # - Draw random input signals
        self._inputs = np.random.randn(num_classes, sample_length, input_channels) + 1.0

        # - Draw random sinusoidal target parameters
        self._target_phase = np.random.rand(num_classes, 1, target_channels) * 2 * np.pi
        self._target_omega = (
                np.random.rand(num_classes, 1, target_channels) * sample_length / 50
        )

        # - Generate target output signals
        time_base = np.atleast_2d(np.arange(sample_length) / sample_length).T
        self._targets = np.sin(
            2 * np.pi * self._target_omega * time_base + self._target_phase
        )

    def __len__(self):
        # - Return the total size of this dataset
        return self._num_classes

    def __getitem__(self, i):
        # - Return the indexed dataset sample
        return torch.Tensor(self._inputs[i]), torch.Tensor(self._targets[i])


# - Instantiate a dataset
Nin = 2000
Nout = 2
num_classes = 2
T = 100
ds = MultiClassRandomSinMapping(
    num_classes=num_classes,
    input_channels=Nin,
    target_channels=Nout,
    sample_length=T,
)
# Display the dataset classes
plt.figure()
for i, sample in enumerate(ds):
    plt.subplot(2, len(ds), i + 1)
    plt.imshow(sample[0].T, aspect="auto")
    plt.title(f"Input class {i}")

    plt.subplot(2, len(ds), i + len(ds) + 1)
    plt.plot(sample[1])
    plt.xlabel(f"Target class {i}")

plt.show()


def SimpleNet(Nin, Nhidden, Nout):
    return Sequential(
        LinearTorch((Nin, Nhidden), has_bias=False),
        LIFTorch(
            Nhidden,
            tau_mem=0.002,
            tau_syn=0.002,
            threshold=1.0,
            learning_window=0.2,
            dt=0.001,
        ),
        LinearTorch((Nhidden, Nout), has_bias=False),
        ExpSynTorch(Nout, dt=0.001, tau=0.01),
    )


Nhidden = 100

net = SimpleNet(Nin, Nhidden, Nout)
print(net)
net.to(device)

# - Get the optimiser functions
optimizer = Adam(net.parameters().astorch(), lr=1e-4)

# - Loss function
loss_fun = MSELoss()

# - Record the loss values over training iterations
loss_t = []

num_epochs = 10000
# - Loop over iterations
for _ in tqdm(range(num_epochs)):
    for input, target in ds:
        optimizer.zero_grad()
        input = input.to(device)
        target = target.to(device)
        output, state, recordings = net(input)

        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        # - Keep track of the loss
        loss_t.append(loss.item())

# - Plot the loss over iterations
plt.clf()
plt.plot(loss_t)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training loss")
plt.show()
