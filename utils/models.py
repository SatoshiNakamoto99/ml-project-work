from torch.nn import CrossEntropyLoss
from torch import optim
import albumentations
from albumentations.pytorch import ToTensorV2
from torch.nn import BCEWithLogitsLoss
import math

import cv2

baseline_augmentation = albumentations.Compose(
        [
        # Condizioni ambientali
        albumentations.RandomSunFlare(src_radius=10,num_flare_circles_lower=3, num_flare_circles_upper=7,p=0.20),
            
        # Baseline
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Rotate(limit=7, p=0.75),
        
        albumentations.OneOf([
            #albumentations.RandomToneCurve(scale=0.2,p=1.), #ok
            albumentations.RandomGamma(p=1.), #ok,
            albumentations.RandomBrightnessContrast(contrast_limit=0.30,brightness_limit=0.15,p=1.), #può andare ma certe volte le immagini sono troppo scure,
        ],p=0.75),
        
        #Rumori
        #albumentations.OneOf([
            albumentations.OneOf([
                albumentations.MultiplicativeNoise(p=1.), #ok
                albumentations.MotionBlur(p=1.), #ok
            ],p=1.),
            #albumentations.ISONoise(p=1.), #ok
        #],p=0.75),
        ],p=1.
    )

models_conf = {}

models_conf['resnet50-test'] = {
        'img_dims': (224, 224),
        'model_name': 'ResNet50-Test',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'resnet_sharma',
        'module_name': 'models.resnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {"lr": .001, "eps": 1e-6, "weight_decay": 0},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['resnet50-test-more-params'] = {
        'img_dims': (224, 224),
        'model_name': 'ResNet50-Test-More-Params',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'resnet_sharma_more_params',
        'module_name': 'models.resnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.SGD,
        'optimizer_params': {"lr": 0.001, "momentum": 0.9, "weight_decay": 0},#{"lr": .0001, "eps": 1e-6, "weight_decay": 0},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': None,#optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3l'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3-Large',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3l',
        'module_name': 'models.mobilenet_v3_large',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,#optim.SGD,
        'optimizer_params': {"lr": .001, "eps": 1e-7, "weight_decay": 0},#lr: 0.001#{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/math.sqrt(10), 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3l-lstm'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3 Large with LSTM',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3l_lstm',
        'module_name': 'models.mobilenet_v3_large_lstm',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,#optim.SGD,
        'optimizer_params': {"lr": .0001, "eps": 1e-7, "weight_decay": 0},#lr: 0.001#{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3l-lstm-our-weights'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3 Large with LSTM',
        'model_path': 'fold_0_best_model.pth',
        'class_name': 'mobilenet_v3l_our_weights',
        'module_name': 'models.mobilenet_v3_large_lstm',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,#optim.SGD,
        'optimizer_params': {"lr": .0001, "eps": 1e-7, "weight_decay": 0},#lr: 0.001#{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3l-aug'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3-Large',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3l_more_params',
        'module_name': 'models.mobilenet_v3_large',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,#optim.SGD,
        'optimizer_params': {"lr": .001, "eps": 1e-7, "weight_decay": 0},#lr: 0.001#{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': baseline_augmentation,
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1./10, 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3l-more-params'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3-Large-More-Params',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3l_more_params',
        'module_name': 'models.mobilenet_v3_large',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {"lr": .0001, "eps": 1e-7, "weight_decay": 0},#{"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3l-more-params-multi'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3-Large-More-Params',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3l_more_params',
        'module_name': 'models.mobilenet_v3_large',
        'criterion': BCEWithLogitsLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {"lr": .00001, "eps": 1e-7, "weight_decay": 0},#{"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3s'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3-Small',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3s',
        'module_name': 'models.mobilenet_v3_small',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam, #optim.SGD,#optim.Adam,#optim.SGD,
        'optimizer_params': {"lr": .01, "eps": 1e-7, "weight_decay": 0}, #{"lr": .001, "momentum": 0.9, "weight_decay": 0},#{"lr": .001, "eps": 1e-7, "weight_decay": 0},#lr: 0.001#{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/math.sqrt(10), 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3s-more-params'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3-Small-More-Params',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3s_more_params',
        'module_name': 'models.mobilenet_v3_small',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,#optim.SGD,
        'optimizer_params': {"lr": 10e-4, "eps": 1e-7, "weight_decay": 0},#{"lr": .0001, "eps": 1e-7, "weight_decay": 0}, #{"lr": .001, "momentum": 0.9, "weight_decay": 0},#{"lr": .001, "eps": 1e-7, "weight_decay": 0},#lr: 0.001#{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': None,#optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['mobilenet-v3s-more-params-multi'] = {
        'img_dims': (224, 224),
        'model_name': 'MobileNetV3-Small-More-Params',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'mobilenet_v3s_more_params',
        'module_name': 'models.mobilenet_v3_small',
        'criterion': BCEWithLogitsLoss(),
        'optimizer': optim.Adam,#optim.SGD,
        'optimizer_params': {"lr": 1e-4, "eps": 1e-7, "weight_decay": 0},#{"lr": .0001, "eps": 1e-7, "weight_decay": 0}, #{"lr": .001, "momentum": 0.9, "weight_decay": 0},#{"lr": .001, "eps": 1e-7, "weight_decay": 0},#lr: 0.001#{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        'preprocess': albumentations.Compose([
                        albumentations.Resize(232,232,interpolation=cv2.INTER_LINEAR),
                        albumentations.CenterCrop(224,224),
                        albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': None,#optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['kutralnet-fismo-bb'] = {
        'img_dims': (84, 84),
        'model_name': 'KutralNet Trained on fismo-black-balanced',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'kutralnet_fismo_bb',
        'module_name': 'models.kutralnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {"lr": .01, "eps": 1e-7, "weight_decay": 0},
        'preprocess': albumentations.Compose([
                       albumentations.Resize(84, 84,interpolation=cv2.INTER_LINEAR), #redimension
                       albumentations.ToFloat(max_value=255),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                              ], p=.5),
        'normalization': None,#albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau, #optim.lr_scheduler.StepLR,#optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_params': {'mode':'min', 'factor':1/math.sqrt(10), 'patience':25, 'min_lr':0.000001}
    }

models_conf['YOLOv8n-cls-bin'] = {
        'img_dims': (640, 640),
        'model_name': 'YOLOv8n_cls',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'yolov8n_cls',
        'module_name': 'models.yolo_v8_cls',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.SGD,#optim.AdamW,
        'optimizer_params': {"lr": 0.001, "momentum": 0.9, "weight_decay": 0},#{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.00},
        'preprocess': albumentations.Compose([
                       albumentations.augmentations.geometric.resize.LongestMaxSize(max_size=640),
                       albumentations.augmentations.geometric.transforms.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT),
                       albumentations.ToFloat(max_value=255),
                       #albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                              ], p=.5),
        'normalization': None,#albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': None,#optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': {'mode':'min', 'factor':1/math.sqrt(10), 'patience':25, 'min_lr':0.000001}
    }

models_conf['YOLOv8n-cls-more-params-bin'] = {
        'img_dims': (640, 640),
        'model_name': 'YOLOv8n_cls_more_params',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'yolov8n_cls_more_params',
        'module_name': 'models.yolo_v8_cls',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,#optim.AdamW,
        'optimizer_params': {"lr": .0001, "eps": 1e-7, "weight_decay": 0},#{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.00},
        'preprocess': albumentations.Compose([
                       albumentations.augmentations.geometric.resize.LongestMaxSize(max_size=640),
                       albumentations.augmentations.geometric.transforms.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT),
                       albumentations.ToFloat(max_value=255),
                       #albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': None,#albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['YOLOv8s-cls-bin'] = {
        'img_dims': (640, 640),
        'model_name': 'YOLOv8s_cls',
        'model_path': 'fold_3_best_model.pth',
        'class_name': 'yolov8s_cls',
        'module_name': 'models.yolo_v8_cls',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,#optim.AdamW,
        'optimizer_params': {"lr": .0001, "eps": 1e-7, "weight_decay": 0},#{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.00},
        'preprocess': albumentations.Compose([
                       albumentations.augmentations.geometric.resize.LongestMaxSize(max_size=640),
                       albumentations.augmentations.geometric.transforms.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT),
                       albumentations.ToFloat(max_value=255),
                       #albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': None,#albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

models_conf['YOLOv8n-cls-multi'] = { #TODO: update
        'img_dims': (640, 640),
        'model_name': 'YOLOv8n_cls',
        'model_path': 'fold_0_best_model.pth',
        'class_name': 'yolov8n_cls',
        'module_name': 'models.yolo_v8_cls',
        'criterion': BCEWithLogitsLoss(),
        'optimizer': optim.Adam,#optim.AdamW,
        'optimizer_params': {"lr": .0001, "eps": 1e-7, "weight_decay": 0},#{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.00},
        'preprocess': albumentations.Compose([
                       albumentations.augmentations.geometric.resize.LongestMaxSize(max_size=640),
                       albumentations.augmentations.geometric.transforms.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT),
                       albumentations.ToFloat(max_value=255),
                       #albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
        'augmentation': albumentations.OneOf([
                                 albumentations.HorizontalFlip(p=1.),
                                 albumentations.Rotate(limit=6, p=1.),
                              ], p=.5),
        'normalization': None,#albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': {'mode':'min', 'factor':1/10., 'patience':25, 'min_lr':0.000001}
    }

def get_config(base_model):
    return models_conf[base_model]
