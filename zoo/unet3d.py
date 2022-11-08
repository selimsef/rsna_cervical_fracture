import torch
from torch import nn
from torch.nn import Conv3d

from zoo.resnet3d_csn import ResNet3dCSN

encoder_params = {
    "r152ir": {
        "filters": [64, 256, 512, 1024, 2048],
        "decoder_filters": [40, 64, 128, 256],
    },

    "r50ir": {
        "filters": [64, 256, 512, 1024, 2048],
        "decoder_filters": [40, 64, 128, 256],
    },
}


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class LastDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2)),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class ConcatBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class Decoder(nn.Module):
    def __init__(self, decoder_filters, filters, upsample_filters=32,
                 decoder_block=DecoderBlock, bottleneck=ConcatBottleneck):
        super().__init__()
        self.decoder_filters = decoder_filters
        self.filters = filters
        self.decoder_block = decoder_block
        self.decoder_stages = nn.ModuleList([self._get_decoder(idx) for idx in range(0, len(decoder_filters))])
        self.bottlenecks = nn.ModuleList([bottleneck(self.filters[-i - 2] + f, f)
                                          for i, f in enumerate(reversed(decoder_filters))])
        self.last_block = None
        if upsample_filters:
            self.last_block = LastDecoderBlock(decoder_filters[0], out_channels=upsample_filters)
        else:
            self.last_block = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")

    def forward(self, encoder_results: list):
        x = encoder_results[0]
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, encoder_results[-rev_idx])
        if self.last_block:
            x = self.last_block(x)
        return x

    def _get_decoder(self, layer):
        idx = layer + 1
        if idx == len(self.decoder_filters):
            in_channels = self.filters[idx]
        else:
            in_channels = self.decoder_filters[idx]
        return self.decoder_block(in_channels, self.decoder_filters[max(layer, 0)])


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,
                                                                                                                   nn.Linear):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class Conv2P1DBnAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act=nn.ReLU, bn=False) -> None:
        super().__init__()
        self.conv1_s = nn.Conv3d(in_channels,
                                 out_channels,
                                 kernel_size=(1, 3, 3),
                                 stride=(1, 1, 1),
                                 padding=(0, 1, 1),
                                 bias=not bn)
        self.bn1_s = nn.BatchNorm3d(out_channels) if bn else nn.Identity()
        self.conv1_t = nn.Conv3d(out_channels,
                                 out_channels,
                                 kernel_size=(3, 1, 1),
                                 stride=(1, 1, 1),
                                 padding=(1, 0, 0),
                                 bias=not bn)
        self.bn1_t = nn.BatchNorm3d(out_channels) if bn else nn.Identity()
        self.relu = act(inplace=True)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)
        return x


class DecoderBlockConv2P1D(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2)):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            Conv2P1DBnAct(in_channels, out_channels, bn=False)
        )

    def forward(self, x):
        return self.layer(x)


class ConcatBottleneckConv2P1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            Conv2P1DBnAct(in_channels, out_channels, bn=False)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class CSNDecoder(Decoder):
    def _get_decoder(self, layer):
        idx = layer + 1
        if idx == len(self.decoder_filters):
            in_channels = self.filters[idx]
        else:
            in_channels = self.decoder_filters[idx]
        if idx == 1:
            return self.decoder_block(in_channels, self.decoder_filters[max(layer, 0)], (1, 2, 2))
        else:
            return self.decoder_block(in_channels, self.decoder_filters[max(layer, 0)])


class ResNet3dCSN2P1D(nn.Module):
    def __init__(self, encoder="r152ir", num_classes=8) -> None:
        super().__init__()
        self.decoder = CSNDecoder(decoder_filters=encoder_params[encoder]["decoder_filters"],
                                  filters=encoder_params[encoder]["filters"],
                                  decoder_block=DecoderBlockConv2P1D,
                                  bottleneck=ConcatBottleneckConv2P1D)
        self.final = Conv3d(32, out_channels=num_classes,
                            kernel_size=1)
        _initialize_weights(self)

        self.backbone = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=int(encoder[1:-2]),
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=False,
            zero_init_residual=False)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1, 1)[:, :, :, :, :]
        encoder_results = list(reversed(self.backbone(x)))
        x = self.decoder(encoder_results)
        x = self.final(x)
        return {"mask": x}
