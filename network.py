import os
import torch
import torchvision.models as models
import wandb


def save_model(state_dict, artifact_name):
    torch.save(state_dict, f"{artifact_name}.pth")
    artifact = wandb.Artifact(artifact_name, type='model')
    artifact.add_file(f"{artifact_name}.pth")
    wandb.log_artifact(artifact)


def load_model(artifact_alias):
    artifact = wandb.use_artifact(artifact_alias, type="model")
    artifact_dir = artifact.download()
    pth = os.path.join(artifact_dir, f"{artifact_alias.split(':')[0]}.pth")
    checkpoint = torch.load(pth)
    return checkpoint


class YoloNetwork(torch.nn.Module):
    def __init__(self, feature_extractor="resnet50", S=7,
                 classes=91, B=2, pretrained=True):
        super(YoloNetwork, self).__init__()
        self.S = S
        self.B = 2
        self.classes = classes
        self.feature_extractor = \
            models.__dict__[feature_extractor](pretrained=pretrained)

        self.output_layer = torch.nn.Linear(
            self.feature_extractor.fc.in_features, S*S*(classes+5*self.B))

        self.feature_extractor.fc = torch.nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        # x = x.reshape((self.S, self.S, self.classes+5*2))
        return x

    def save_model(self):
        save_model(self.state_dict(), "model")

    def load_model(self, artifact_alias):
        self.load_state_dict(load_model(artifact_alias))

    def save_feature_extractor(self):
        save_model(self.feature_extractor.state_dict(), "feature_extractor")

    def load_feature_extractor(self, artifact_alias):
        self.feature_extractor.load_state_dict(load_model(artifact_alias))

    def freeze_feature_extractor(self, freeze):
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
