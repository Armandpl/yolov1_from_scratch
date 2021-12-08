import torch
import torchvision.models as models


class YoloNetwork(torch.nn.Module):
    def __init__(self, feature_extractor="resnet50", S=7,
                 classes=80, pretrained=True):
        super(YoloNetwork, self).__init__()
        self.S = S
        self.classes = classes
        self.feature_extractor = \
            models.__dict__[feature_extractor](pretrained=pretrained)

        # freeze feature extractor
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        self.output_layer = torch.nn.Linear(
            self.feature_extractor.fc.in_features, S*S*(classes+5*2))

        self.feature_extractor.fc = torch.nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        # x = x.reshape((self.S, self.S, self.classes+5*2))
        return x
