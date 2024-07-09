from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def mobilenet_v3s(classes):
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    first_in: int = mobilenet.classifier[0].in_features
    first_out: int = mobilenet.classifier[0].out_features
    last_in: int = mobilenet.classifier[-1].in_features
    
    # modify the last layer to add another dense layer
    mobilenet.classifier[0] = nn.Linear(first_in, first_out) #Change the last layer to have 2 outputs
    mobilenet.classifier[-1] = nn.Linear(last_in, classes) #Change the last layer to have 2 outputs

    #Freeze all layers except the last two (the last two linear layer in classifier)
    for name, param in mobilenet.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False

    return mobilenet


def mobilenet_v3s_more_params(classes):
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    first_in: int = mobilenet.classifier[0].in_features
    first_out: int = mobilenet.classifier[0].out_features
    last_in: int = mobilenet.classifier[-1].in_features
    
    # modify the last layer to add another dense layer
    mobilenet.classifier[0] = nn.Linear(first_in, first_out) #Change the last layer to have 2 outputs
    mobilenet.classifier[-1] = nn.Linear(last_in, classes) #Change the last layer to have 2 outputs

    #Freeze all layers except the last two (the last two linear layer in classifier)
    for name, param in mobilenet.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False
        if "features.12" in name:
            param.requires_grad = True
        if "features.11" in name:
            param.requires_grad = True

    return mobilenet