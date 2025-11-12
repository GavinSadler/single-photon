import torch
import torch.nn as nn
from torch.nn.modules.module import Module


# def actFunc(act, *args, **kwargs) -> Module:
def actFunc(act, *args, **kwargs):
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


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True
    )


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True
    )


def conv5x5x5(in_channels, out_channels, stride=1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True
    )


def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=True
    )


# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation="relu"):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), dim=1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation="relu"):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


# RDB fusion module
class RDB_shrink(nn.Module):
    def __init__(
        self, in_channels, growthRate, output_channels, num_layer, activation="relu"
    ):
        super(RDB_shrink, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.shrink = conv3x3(in_channels, output_channels, stride=1)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.shrink(x)
        return out


# DownSampling module
class RDB_DS(nn.Module):
    def __init__(
        self, in_channels, growthRate, output_channels, num_layer, activation="relu"
    ):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, output_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(
        self, in_channels, growthRate, num_layer, num_blocks, activation="relu"
    ):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

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


class RDBConfig:
    def __init__(self, activation="gelu", n_features=64, n_blocks=12, inp_ch=1):
        self.activation = activation
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.inp_ch = inp_ch


class RDBCell(nn.Module):
    def __init__(self, config=None):
        super(RDBCell, self).__init__()

        # Use default config if none provided
        if config is None:
            config = RDBConfig()

        self.activation = config.activation
        self.n_feats = config.n_features
        self.n_blocks = config.n_blocks
        self.F_B01 = conv5x5(config.inp_ch, self.n_feats, stride=1)
        self.F_B01_fuse = conv1x1(2 * self.n_feats, 4 * self.n_feats)
        self.F_B02 = conv3x3(config.inp_ch, self.n_feats, stride=1)
        self.F_B03 = conv3x3(config.inp_ch, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(
            in_channels=4 * self.n_feats,
            growthRate=self.n_feats,
            output_channels=4 * self.n_feats,
            num_layer=3,
            activation=self.activation,
        )
        self.F_B2 = RDB_DS(
            in_channels=6 * self.n_feats,
            growthRate=int(self.n_feats * 3 / 2),
            output_channels=4 * self.n_feats,
            num_layer=3,
            activation=self.activation,
        )
        self.F_R = nn.Sequential(
            RDB_shrink(
                in_channels=8 * self.n_feats,
                growthRate=self.n_feats,
                output_channels=4 * self.n_feats,
                num_layer=3,
                activation=self.activation,
            ),
            RDNet(
                in_channels=4 * self.n_feats,
                growthRate=2 * self.n_feats,
                num_layer=3,
                num_blocks=self.n_blocks - 1,
                activation=self.activation,
            ),
        )
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3(4 * self.n_feats, 2 * self.n_feats),
            RDB(
                in_channels=2 * self.n_feats,
                growthRate=2 * self.n_feats,
                num_layer=3,
                activation=self.activation,
            ),
            conv3x3(2 * self.n_feats, 2 * self.n_feats),
        )

    def forward(self, xs, x_feats, s_last):
        """
        Args:
            xs: Multi-scale input list [x1, x2, x3]
                x1: [B, 1, 256, 256] - Full resolution
                x2: [B, 1, 128, 128] - Half resolution
                x3: [B, 1, 64, 64] - Quarter resolution
            x_feats: Multi-scale feature list [x_feat1, x_feat2, x_feat3]
                x_feat1: [B, 64, 256, 256]
                x_feat2: [B, 64, 128, 128]
                x_feat3: [B, 96, 64, 64]
            s_last: Previous hidden state [B, 128, 64, 64]

        Returns:
            _type_: _description_
        """

        """
        input: size 256 256
        0: torch.Size([2, 64, 256, 256])
        1: torch.Size([2, 64, 128, 128])
        2: torch.Size([2, 96, 64, 64])
        3: torch.Size([2, 96, 64, 64])
        """
        x1, x2, x3 = xs
        b, c, h, w = x1.shape

        x_feat1, x_feat2, x_feat3 = x_feats

        out1 = self.F_B01(x1)
        out1 = self.F_B01_fuse(torch.cat([out1, x_feat1], 1))
        inp2 = self.F_B02(x2)
        inp3 = self.F_B03(x3)
        # in: 2 n_feats, out: 2 * n_feats
        out2 = self.F_B1(out1)
        # in: 2+1+1 n_feats, out: 3 * n_feats
        out3 = self.F_B2(torch.cat([out2, inp2, x_feat2], 1))
        out3 = torch.cat([out3, inp3, x_feat3, s_last], dim=1)
        # in: 3+1+1+2 n_feats, out: 5 * n_feats
        out3 = self.F_R(out3)
        # in: 5 n_feats, out: 2 * n_feats
        s = self.F_h(out3)
        # print(out1.shape, out2.shape, out3.shape)
        return out1, out2, out3, s
