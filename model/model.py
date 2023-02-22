import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMask(nn.Module):
    """
    Calculate attention over the provided input.
    Check the dim over which sum needs to be taken.
    """
    def __init__(self):
        super(AttentionMask, self).__init__()
    
    def forward(self, x):
        # check: the dimension over which summation needs to be taken. If input tensor is [N, H, W, C] (tensorflow). Then its over spatial dimensions. Change correspondngly.
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = x.size()
        return (x/xsum) * xshape[1] * xshape[2] * 0.5


class TSM(nn.Module):
    """
    Temporal shift module
    Tensor order (tensorflow)- shape=(N, H, W, C)
    Tensor order (pytorch)- torch.Size([N, C, H, W])
    """
    def __init__(self, n_frame, fold_div=3):
        super(TSM, self).__init__()
        self.n_frame = n_frame
        self.fold_div = fold_div
    
    def forward(self, x):
        #  shape
        b, c, h, w = x.shape
        # reshape
        x = torch.reshape(x, (-1, self.n_frame, c, h, w))
        fold = c // self.fold_div
        last_fold = c - (self.fold_div - 1) * fold
        # split based on channel dimension. SC has axis=-1
        out1, out2, out3 = torch.split(x, [fold, fold, last_fold], dim=2)

        # shift left
        padding_1 = torch.zeros_like(out1)
        # last frame
        padding_1 = padding_1[:, -1, :, :, :]
        # introduce dimension for frames again
        padding_1 = padding_1.unsqueeze(dim=1)
        _, out1 = torch.split(out1, [1, self.n_frame-1], dim=1)
        out1 = torch.cat((out1, padding_1), dim=1)

        # shift right
        padding_2 = torch.zeros_like(out2)
        # first frame
        padding_2 = padding_2[:, 0, :, :, :]
        # introduce dimension for frames again
        padding_2 = padding_2.unsqueeze(dim=1)
        out2, _ = torch.split(out2, [self.n_frame-1, 1], dim=1)
        out2 = torch.cat((padding_2, out2), dim=1)

        # concatenate outs over channel axis
        out = torch.cat((out1, out2, out3), dim=2)
        out = torch.reshape(out, (-1, c, h, w))
        return out


class TemporalShiftModuleConvolution(nn.Module):
    def __init__(self, n_frame, in_channels, out_channels, kernel_size=(3,3), padding="same", activation="tanh"):
        super(TemporalShiftModuleConvolution, self).__init__()
        self.tsm = TSM(n_frame=n_frame)
        # padding = "same", check if this is going to give problems if trying on onxx
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.tsm(x)
        x = self.conv1(x)
        x = self.activation(x)

        return x


class Conv2DWithActivation(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=(3,3,), padding="valid", activation="tanh"):
        super(Conv2DWithActivation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "relu":
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        return x


class TemporalShiftCAN(nn.Module):
    def __init__(self, n_frame, in_channels, out_channels_1, out_channels_2, kernel_size=(3,3), hidden_size=128):
        super(TemporalShiftCAN, self).__init__()

        # TSM convolution for motion data
        self.tsm_conv1 = TemporalShiftModuleConvolution(n_frame=n_frame, in_channels=in_channels, out_channels=out_channels_1, 
                                                        kernel_size=kernel_size, padding="same", activation="tanh")
        self.tsm_conv2 = TemporalShiftModuleConvolution(n_frame=n_frame, in_channels=out_channels_1, out_channels=out_channels_1, 
                                                        kernel_size=kernel_size, padding="valid", activation="tanh")

        # regular convolution on appearance data
        self.reg_conv1 = Conv2DWithActivation(in_channels=in_channels, out_channels=out_channels_1, kernel_size=kernel_size, 
                                              padding="same", activation="tanh")
        self.reg_conv2 = Conv2DWithActivation(in_channels=out_channels_1, out_channels=out_channels_1, kernel_size=kernel_size, 
                                              padding="valid", activation="tanh")

        # gated convolution 1
        self.g1_conv = Conv2DWithActivation(in_channels=out_channels_1, out_channels=1, kernel_size=(1,1), padding="same", activation="sigmoid")
        # Attention mask
        self.attention_mask = AttentionMask()
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2))
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # TSM Covolution
        self.tsm_conv3 = TemporalShiftModuleConvolution(n_frame=n_frame, in_channels=out_channels_1, out_channels=out_channels_2, 
                                                        kernel_size=kernel_size, padding="same", activation="tanh")
        self.tsm_conv4 = TemporalShiftModuleConvolution(n_frame=n_frame, in_channels=out_channels_2, out_channels=out_channels_2, 
                                                        kernel_size=kernel_size, padding="valid", activation="tanh")

        # regular convolution
        self.reg_conv3 = Conv2DWithActivation(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size, padding="same", 
                                              activation="tanh")
        self.reg_conv4 = Conv2DWithActivation(in_channels=out_channels_2, out_channels=out_channels_2, kernel_size=kernel_size, padding="valid", 
                                              activation="tanh")

        # gated convolution 2
        self.g2_conv = Conv2DWithActivation(in_channels=out_channels_2, out_channels=1, kernel_size=(1,1), padding="same", activation="sigmoid")

        # FC layers
        # check this in feature size
        self.fc1 = nn.Linear(in_features=out_channels_2*7*7, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)
        self.final_activation = nn.Tanh()
        


    def forward(self, x_motion, x_appearance):
        # x_motion: [10, 3, 36, 36], x_appearance: [10,3,36,36]
        d1 = self.tsm_conv1(x_motion)
        d2 = self.tsm_conv2(d1)

        r1 = self.reg_conv1(x_appearance)
        r2 = self.reg_conv2(r1)

        g1 = self.g1_conv(r2)
        g1 = self.attention_mask(g1)
        gated1 = torch.mul(d2, g1)

        d3 = self.avgpool(gated1)
        d4 = self.dropout(d3)

        r3 = self.avgpool(r2)
        r4 = self.dropout(r3)

        d5 = self.tsm_conv3(d4)
        d6 = self.tsm_conv4(d5)

        r5 = self.reg_conv3(r4)
        r6 = self.reg_conv4(r5)

        g2 = self.g2_conv(r6)
        g2 = self.attention_mask(g2)
        gated2 = torch.mul(d6, g2)

        d7 = self.avgpool(gated2)
        d8 = self.dropout(d7)

        # d9 = torch.flatten(d8)
        d9 = d8.view(d8.shape[0], -1)
        d10 = self.fc1(d9)
        d10 = self.final_activation(d10)
        d11 = self.dropout2(d10)
        out = self.fc2(d11)

        return out
