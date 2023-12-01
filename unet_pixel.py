import torch
import torch.nn as nn


class unet_pixel(torch.nn.Module):
    def __init__(self) -> None:
        super(unet_pixel, self).__init__()
        self.channels = 32
        self.leaky = 0.1
        self.dropout = 0.3

        self.shrink_1 = nn.Sequential(
            nn.Conv2d(3, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout),
        )

        self.shrink_2 = nn.Sequential(
            nn.Conv2d(self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout),
        )

        self.shrink_3 = nn.Sequential(
            nn.Conv2d(2 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout),
        )

        self.shrink_4 = nn.Sequential(
            nn.Conv2d(4 * self.channels, 8 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(8 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(8 * self.channels, 8 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(8 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(8 * self.channels, 8 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(8 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout),
        )

        self.bottom_conv = nn.Sequential(
            nn.Conv2d(8 * self.channels, 16 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(16 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(16 * self.channels, 16 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(16 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(16 * self.channels, 16 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(16 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(16 * self.channels, 16 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(16 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(16 * self.channels, 8 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(8 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
        )

        self.up_3 = nn.Sequential(
            nn.Conv2d(16 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(4 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.ConvTranspose2d(4 * self.channels, 4 * self.channels, 3, 2, 1, 1),
            nn.Dropout2d(self.dropout),
        )

        self.up_2 = nn.Sequential(
            nn.Conv2d(8 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, 1, 1),
            nn.BatchNorm2d(2 * self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.ConvTranspose2d(2 * self.channels, 2 * self.channels, 3, 2, 1, 1),
            nn.Dropout2d(self.dropout),
        )

        self.up_1 = nn.Sequential(
            nn.Conv2d(4 * self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels, track_running_stats=False),
            nn.LeakyReLU(self.leaky),
            nn.ConvTranspose2d(self.channels, self.channels, 3, 2, 1, 1),
            nn.Dropout2d(self.dropout),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(self.leaky),
            nn.ConvTranspose2d(self.channels, self.channels, 3, 2, 1, 1),
            nn.Dropout2d(self.dropout),
        )

        self.channel_adapter = nn.Sequential(
            nn.Conv2d(self.channels, 1, 3, 1, 1), nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight.data)
                module.bias.data.zero_()

    def forward(self, x):
        shrink1 = self.shrink_1(x)
        shrink2 = self.shrink_2(shrink1)
        shrink3 = self.shrink_3(shrink2)
        shrink4 = self.shrink_4(shrink3)
        up4 = self.bottom_conv(shrink4)
        up4 = torch.cat((up4, shrink4), dim=1)
        up3 = self.up_3(up4)
        up3 = torch.cat((up3, shrink3), dim=1)
        up2 = self.up_2(up3)
        up2 = torch.cat((up2, shrink2), dim=1)
        up1 = self.up_1(up2)
        up1 = torch.cat((up1, shrink1), dim=1)
        output_3c = self.final_conv(up1)
        output = self.channel_adapter(output_3c)
        return output
