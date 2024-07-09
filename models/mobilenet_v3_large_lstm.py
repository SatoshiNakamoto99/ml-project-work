import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from models.mobilenet_v3_large import mobilenet_v3l

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(CNNLSTM, self).__init__()
        if not pretrained:
            self.mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        else:
            self.mobilenet = mobilenet_v3l(num_classes)
            self.mobilenet.load_state_dict(torch.load("experiments/mobilenet-v3l/foggia-unbalanced-ext/fold_0_best_model.pth"))
        self.mobilenet.classifier[-1] =nn.Sequential(nn.Linear(self.mobilenet.classifier[-1].in_features, 300)) #Change the last layer to have 2 outputs

        #self.resnet = resnet101(pretrained=True)
        #self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.mobilenet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

def mobilenet_v3l_lstm(classes):
    mobilenet_LSTM = CNNLSTM(classes) 
    mobilenet = mobilenet_LSTM.mobilenet
    first_in: int = mobilenet.classifier[0].in_features
    first_out: int = mobilenet.classifier[0].out_features
   
    # modify the last layer to add another dense layer
    mobilenet.classifier[0] = nn.Linear(first_in, first_out) #Change the last layer to have 2 outputs
    #mobilenet.classifier[-1] = nn.Linear(last_in, classes) #Change the last layer to have 2 outputs

    #Freeze all layers except the last three blocks
    for name, param in mobilenet.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False
        if "features.16" in name:
            param.requires_grad = True
        if "features.15" in name:
            param.requires_grad = True
    

    return mobilenet_LSTM

def mobilenet_v3l_our_weights(classes):
    mobilenet_LSTM = CNNLSTM(classes) 
    mobilenet = mobilenet_LSTM.mobilenet
    first_in: int = mobilenet.classifier[0].in_features
    first_out: int = mobilenet.classifier[0].out_features
   
    # modify the last layer to add another dense layer
    mobilenet.classifier[0] = nn.Linear(first_in, first_out) #Change the last layer to have 2 outputs
    #mobilenet.classifier[-1] = nn.Linear(last_in, classes) #Change the last layer to have 2 outputs

    #Freeze all layers except the last three blocks
    for name, param in mobilenet.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False
    

    return mobilenet_LSTM
