from torch import nn

from zoo import ResNet3dCSN
from zoo.utils import _initialize_weights


class ClassifierResNet3dCSN2P1D(nn.Module):
    def __init__(self, encoder="r50ir", pool="avg", norm_eval=False, num_classes=1) -> None:
        super().__init__()

        self.final = nn.Linear(2048, out_features=num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if pool =="avg" else nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        _initialize_weights(self)

        self.backbone = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=int(encoder[1:-2]),
            with_pool2=False,
            bottleneck_mode=encoder[-2:],
            norm_eval=norm_eval,
            zero_init_residual=False)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1, 1)[:, :, :, :, :]
        x = self.backbone(x)[-1]
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.final(x)
        return {"cls": x}
