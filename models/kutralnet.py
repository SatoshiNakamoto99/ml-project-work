"""
transform_compose = transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=85)

Training complete in 8m 39s
Best accuracy on epoch 93: 0.891179
Accuracy of the network on the test images: 82.02%
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class KutralBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super(KutralBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.act = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.pool(self.act(x))
        return x

class KutralNet(nn.Module): # test 12
    def __init__(self, classes, initial_filters=32):
        super(KutralNet, self).__init__()
        self.block1 = KutralBlock(in_ch=3, out_ch=initial_filters, kernel_size=3, stride=1, padding=1, bias=False)

        n_filters = initial_filters * 2
        self.block2 = KutralBlock(in_ch=initial_filters, out_ch=n_filters, kernel_size=3, stride=1, padding=1, bias=False)

        initial_filters = n_filters
        n_filters = initial_filters * 2
        self.block3 = KutralBlock(in_ch=initial_filters, out_ch=n_filters, kernel_size=3, stride=1, padding=1, bias=False)

        initial_filters = n_filters
        n_filters = initial_filters // 2
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=initial_filters, out_channels=n_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters)
        )

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=n_filters)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_filters, classes)

        self._init_params()

    def forward(self, x):
        debug = False
        if debug:
            print('x.size()', x.size())
        x = self.block1(x)
        if debug:
            print('block1.size()', x.size())
        shortcut = self.block2(x)
        if debug:
            print('block2.size()', x.size())
        x = self.block3(shortcut)
        if debug:
            print('block3.size()', x.size())
        x = self.block4(x)
        if debug:
            print('block4.size()', x.size())
        x += self.down_sample(shortcut)
        # global average pooling
        x = self.global_pool(F.leaky_relu(x))
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def kutralnet_fismo_bb(classes):
    kutralnet = KutralNet(classes, initial_filters=32)
    kutralnet.load_state_dict(torch.load("models/saved/kutralnet/fismo_balanced_black/model_kutralnet.pth", map_location=torch.device('cuda')))
    kutralnet.classifier = nn.Linear(kutralnet.classifier.in_features, classes)
    
    #Freeze all layers except the last two (the last two linear layer in classifier)
    for name, param in kutralnet.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False
    
    return kutralnet