from ultralytics import YOLO
import ultralytics
import torch
"""
Transfer learning from YOLOv8n modified version to fire and smoke classification
(either binary or multilabel)
"""

class Classify(ultralytics.nn.modules.head.Classify):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k=1, s=1, p=None, g=1)
    
    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x

def yolov8n_cls(classes):
    yolo = YOLO("./models/saved/YOLOv8n_cls/yolov8n-cls.pt")
    model = yolo.model
    
    #attach new head
    backbone_out_channels = model.model[-2].cv2.conv.out_channels
    old_head = model.model[-1]
    new_head = Classify(backbone_out_channels,classes)
    new_head.i, new_head.f, new_head.type = old_head.i, old_head.f, old_head.type
    model.model[-1] = new_head
    
    # freeze backbone for transfer learning
    freeze_layer = 9
    for i, child in enumerate(model.model.children()):
        if i < freeze_layer:
            for param in child.parameters():
                param.requires_grad = False

    return model

def yolov8n_cls_more_params(classes):
    yolo = YOLO("./models/saved/YOLOv8n_cls/yolov8n-cls.pt")
    model = yolo.model
    
    #attach new head
    backbone_out_channels = model.model[-2].cv2.conv.out_channels
    old_head = model.model[-1]
    new_head = Classify(backbone_out_channels,classes)
    new_head.i, new_head.f, new_head.type = old_head.i, old_head.f, old_head.type
    model.model[-1] = new_head
    
    # freeze backbone for transfer learning
    freeze_layer = 8
    for i, child in enumerate(model.model.children()):
        for param in child.parameters():
            if i < freeze_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model

def yolov8s_cls(classes):
    yolo = YOLO("./models/saved/YOLOv8s_cls/yolov8s-cls.pt")
    model = yolo.model
    
    #attach new head
    backbone_out_channels = model.model[-2].cv2.conv.out_channels
    old_head = model.model[-1]
    new_head = Classify(backbone_out_channels,classes)
    new_head.i, new_head.f, new_head.type = old_head.i, old_head.f, old_head.type
    model.model[-1] = new_head
    
    # freeze backbone for transfer learning
    for name, param in model.named_parameters():
        if not "model.8" in name and not "model.9" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model
