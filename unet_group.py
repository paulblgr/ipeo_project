import torch
import torch.nn as nn


class unet_group(torch.nn.Module):
    def __init__(self) -> None:
        super(unet_group, self).__init__()
        self.channels = 64
        self.leaky = 0.3
        self.dropout = 0.5
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.channels, 3, 1, 0),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 0),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Dropout2d(self.dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.channels, 2 * self.channels, 3, 1, 0),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Dropout2d(self.dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Dropout2d(self.dropout),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Dropout2d(self.dropout),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(2 * self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, 1, 1),
            nn.Dropout2d(self.dropout),
            nn.Sigmoid(),
        )

        self.shrink1 = nn.MaxPool2d((2, 2))
        self.shrink2 = nn.MaxPool2d((2, 2))

        self.upscale1 = nn.ConvTranspose2d(4 * self.channels, 2 * self.channels, 3, 1)
        self.upscale2 = nn.ConvTranspose2d(2 * self.channels, self.channels, 2, 2)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight.data)
                module.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv1(x)
        shrink1 = self.shrink1(conv1)
        conv2 = self.conv2(shrink1)
        shrink2 = self.shrink2(conv2)
        conv3 = self.conv3(shrink2)
        upscale1 = self.upscale1(conv3)
        merge_1 = conv2[:, :, 3:11, 3:11]
        upscale1 = torch.cat((upscale1, merge_1), dim=1)
        conv4 = self.conv4(upscale1)
        upscale2 = self.upscale2(conv4)
        upscale2 = torch.cat((upscale2, conv1[:, :, 6:22, 6:22]), dim=1)
        conv5 = self.conv5(upscale2)
        return conv5
