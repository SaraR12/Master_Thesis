from torch import nn
import torch.nn.functional as F

class PathPrediction(nn.Module):

    def __init__(self, M): # M = time steps ahead/before
        super().__init__()

        self.M = M

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 2*M))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 64))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 64))

        self.max_pooling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)

        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 64))
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 64))
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=10, kernel_size=(3, 3, 2*M))

        self.deconv = nn.ConvTranspose3d(in_channels=10, out_channels=2*M, kernel_size=(4, 4, 1), stride=2)

    def forward(self, x):
        # x -> conv1 -> ReLU -> conv2 -> ReLU -> conv3 -> ReLU -> max_pooling -> ReLU -> conv4 -> ReLU -> conv5 -> ReLU -> conv6 -> deconv -> x
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.functional.pad(x, (7, 7, 1, 1, 1, 1))

        x = self.conv2(x)
        x = F.relu(x)
        x = nn.functional.pad(x, (31, 31, 1, 1, 1, 1))

        x = self.conv3(x)
        x = F.relu(x)
        x = nn.functional.pad(x, (31, 31, 1, 1, 1, 1))

        x = self.max_pooling(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = nn.functional.pad(x, (31, 31, 1, 1, 1, 1))

        x = self.conv5(x)
        x = F.relu(x)
        x = nn.functional.pad(x, (31, 31, 1, 1, 1, 1))

        x = self.conv6(x)
        x = self.deconv(x)
        return x
