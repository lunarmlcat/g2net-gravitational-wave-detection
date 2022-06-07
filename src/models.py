import timm
import torch.nn as nn
import torch.nn.functional as F


def get_model(model_name, device, pretrained=True, num_classes=1000, model_freeze=False):
    model = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
    if model_freeze:
        for param in model.parameters():
            param.requires_grad = False
            
    ##### set nn.Linear
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    model.to(device)
    
    return model

def get_custom_model(model_name, device, pretranied=True, num_classes=1000, model_freeze=False):
    model = CustomModel(model_name, pretrained=pretranied, num_classes=num_classes, model_freeze=model_freeze)
    model.to(device)

    return model


# --------------------------------------
# Pooling layers
# --------------------------------------
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class CustomModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1000, model_freeze=False):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        self.backbone = nn.Sequential(*backbone.children())[:-3]
        if model_freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.gem_pool = gem
        feat_dim = {"eca_nfnet_l2": 1536}
        self.classifier = nn.Linear(feat_dim[model_name], num_classes)

    def forward(self, x):
        x = self.backbone(x)# torch.Size([64, 1536, 12, 12])
        x = self.gem_pool(x)
        x = x.view(x.size(0), -1) # torch.Size([64, 1536])
        x = self.classifier(x)
        return x