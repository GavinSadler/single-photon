from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


def activation_function(act, *args, **kwargs):
    act = act.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "relu6":
        return nn.ReLU6()
    elif act == "leakyrelu":
        return nn.LeakyReLU(0.1)
    elif act == "prelu":
        return nn.PReLU()
    elif act == "rrelu":
        return nn.RReLU(0.1, 0.3)
    elif act == "selu":
        return nn.SELU()
    elif act == "celu":
        return nn.CELU()
    elif act == "elu":
        return nn.ELU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError


def Conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True
    )


def Conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
    )


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, activation="relu"):
        super(DenseLayer, self).__init__()
        self.conv = Conv3x3(in_channels, growth_rate)
        self.act = activation_function(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layer, activation="relu"):
        super(RDB, self).__init__()
        prev_out_channels = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(DenseLayer(prev_out_channels, growth_rate, activation))
            prev_out_channels += growth_rate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = Conv1x1(prev_out_channels, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


class RDNet(nn.Module):
    def __init__(
        self, in_channels, growth_rate, num_layers, num_blocks, activation="relu"
    ):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growth_rate, num_layers, activation))
        self.conv1x1 = Conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = Conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out


class DenoisingModel(nn.Module):
    def __init__(self, in_channels=3, growth_rate=32, num_layer=6, num_blocks=4):
        super(DenoisingModel, self).__init__()
        self.shallow = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.rdnet = RDNet(in_channels, growth_rate, num_layer, num_blocks)
        self.recon = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        shallow = self.shallow(x)
        deep = self.rdnet(shallow)
        out = self.recon(deep)
        return x - out  # residual learning: predict noise and subtract


class DenoisingModel(nn.Module):

    def __init__(self, in_channels=3, growth_rate=32, num_layers=6, num_blocks=4):
        super().__init__()
        self.shallow = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.rdnet = RDNet(in_channels, growth_rate, num_layers, num_blocks)
        self.reconstruct = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Map input to feature space
        d = self.shallow(x)

        # Process output through RDNet
        d = self.rdnet(d)

        # Map output back to
        d = self.reconstruct(d)

        # Return the residual
        return x - d


class DenoisingDataset(Dataset):
    def __init__(self, root_dir, patch_size=48, sigma=25, augment=True):
        self.files = sorted(Path(root_dir).glob("*.png"))
        self.patch_size = patch_size
        self.sigma = sigma
        self.augment = augment
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load and transform image
        img = Image.open(self.files[idx]).convert("RGB")
        clean = self.to_tensor(img)

        # Randomly crop the image
        if self.patch_size:
            i = random.randint(0, clean.shape[1] - self.patch_size)
            j = random.randint(0, clean.shape[2] - self.patch_size)
            clean = clean[:, i : i + self.patch_size, j : j + self.patch_size]

        # Add gaussian noise on the fly
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = (clean + noise).clamp(0.0, 1.0)

        return noisy, clean


dataloader = DenoisingDataset("./data/denoising-datasets/BSD400")

model = DenoisingModel(in_channels=1)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i, (noisy, clean) in enumerate(dataloader):
    print(noisy.shape())
    optimizer.zero_grad()
    output = model(noisy)
    loss = criterion(output, clean)
    loss.backward()
    optimizer.step()

    print(f"Processed file {i + 1} of {len(dataloader)} images...")
    print(f"\tloss={loss}")
