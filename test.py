import cv2, os, argparse, random
import numpy as np
import importlib
import torch
import torch.nn.functional as F
from timeit import default_timer as timer
from utils.models import models_conf
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import ToFloat
import albumentations
from PIL import Image
import matplotlib.pyplot as plt

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    parser.add_argument("--base_model", type=str, default='mobilenet-v3s-more-params', help="Base model")
    parser.add_argument("--training_dataset", type=str, default='foggia-unbalanced-ext', help="Training dataset")
    parser.add_argument("--models_root", type=str, default='experiments', help="Models root")
    parser.add_argument("--model_name", type=str, default='mobilenet-v3s-more-params-stratified-kfold', help="Model name")
    parser.add_argument("--model_path", type=str, default='fold_5_best_model.pth', help="Model path")
    parser.add_argument("--lstm_mode", type=bool, default=False, help="Enable LSTM mode")
    parser.add_argument("--multilabel", type=bool, default=False, help="Enable multilabel mode")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method
if not os.path.exists(args.results):
   os.makedirs(args.results)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
else:
    device = torch.device("cpu")
base_model = args.base_model#"mobilenet-v3l"#"YOLOv8n-cls-bin"#"resnet50-test"
training_dataset = args.training_dataset#"foggia-mod" #"foggia-mod"
models_root = args.models_root#'experiments' #"experiments" #'models/saved'
model_name = args.model_name
model_path = args.model_path

treshold = 3
stride = 24
FPS = 24.0

if base_model not in models_conf:
    raise ValueError('Model {} is not supported'.format(base_model))

config = models_conf[base_model]
img_dims = config['img_dims']
#model_path = config['model_path']
num_classes = 2
module = importlib.import_module(config['module_name'])
ModelClass = getattr(module, config['class_name'])
model = ModelClass(classes=num_classes)
model_path = os.path.join(models_root, model_name, training_dataset, model_path)
preprocess = config['preprocess']
normalization = config['normalization']
if normalization is None:
    normalization = albumentations.Compose([])
pipeline = albumentations.Compose([
    preprocess,
    normalization,
    ToTensorV2()
    ])
print('Loading model', model_name, 'from path', model_path, 'trained with', training_dataset, "base model ", base_model)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)
################################################
# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(os.path.join(args.videos, video))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    stride = int(FPS) #We skip one second of frames each time
    print("Processing video {} @ {} FPS...".format(video, FPS), end="")
    # accum_time = 0
    # curr_fps = 0
    # prev_time = timer()
    pos = 0
    i = 0 #frame index
    sequence_index = 0
    sequence = []
    consecutive_positives = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        # Here you should add your code for applying your metho

        # image_r = cv2.resize(frame, img_dims)
        # # Normalize data.
        # image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
        # image_r = image_r.astype('float32') / 255
        # #pil_im = Image.fromarray(image_r)
        # im_tensor = torch.from_numpy(image_r.transpose((2,0,1)))
        # #image_r = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # #im_pil = Image.fromarray(image_r)
        # #im_tensor = config['preprocess_test'](im_pil)
        # im_tensor=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(im_tensor)
        # #print(im_tensor.shape)
        # im_tensor = torch.unsqueeze(im_tensor, 0)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if not args.lstm_mode:
            # print(type(im_tensor), im_tensor.shape)
            augmented = pipeline(image=image)
            image = augmented['image']
            image = torch.unsqueeze(image, 0)
            
            with torch.inference_mode():
                pred = model(image.to(device))#torch.from_numpy(image.transpose((0, 3, 1, 2))))
                #print("pre:",pred.sum(dim=1))
                if args.multilabel==False:
                    pred = F.softmax(pred, dim=1).squeeze(0)
                    fire_probability = pred[1]
                else:
                    pred = torch.sigmoid(pred).squeeze(0)
                    pred_labels = (pred.detach() > 0.5).to(torch.float32) # Binarize the predictions
                    fire_probability = torch.max(pred_labels).item() # If at least one label is 1, the fire/smoke probability is 1
                    #print("pred_labels:",pred_labels, "fire_probability:",fire_probability)
                #print("post:",pred.sum(dim=1))
                #print(pred)
            #print("Frame analyzed:",i)
            
            
            #print("Video {} frame {} probability {}".format(video, i, fire_probability))
            if(fire_probability > .5):
                consecutive_positives += 1
            else:
                consecutive_positives = 0
            if consecutive_positives >= treshold:
                pos = 1
                break
        else: #LSTM mode
            if args.multilabel==True:
                raise NotImplementedError("Multilabel + LSTM mode not implemented")
            if sequence_index == 0: # A new sequence starts
                sequence = []
            sequence.append(image)
            sequence_index = (sequence_index+1) % treshold
            if sequence_index == 0: # The sequence is complete
                sequence_tensor = [pipeline(image=x)['image'] for x in sequence] #Apply the same pipeline to all images in the sequence
                sequence_tensor = torch.stack(sequence_tensor) #Stack the images in the sequence along the first dimension
                pred = model(sequence_tensor.unsqueeze(0).to("cuda")) #Simulate batch with 1 sample
                pred = F.softmax(pred, dim=1).squeeze(0) # Remove batch dimension and apply softmax
                fire_prob = pred[1].item() #Get the probability of fire
                if fire_prob > .5:
                    pos = 1
                    break
        i += 1 #update frame index
        j = 0
        while ret and j < stride:
            ret = cap.grab() #skip frames
            j += 1 
            i += 1 #update frame index
        
        ########################################################
    cap.release()
    f = open(args.results+video+".txt", "w")
    # Here you should add your code for writing the results
    max_mem_gb=torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    if pos:
        # print(i,stride,consecutive_positives)
        if not args.lstm_mode:
            prediction_frame = i-(consecutive_positives-1)*(stride+1)
        else:
            prediction_frame = i-(treshold-1)*(stride+1)
        predicition_time = int(np.ceil(prediction_frame/FPS))
        f.write(str(predicition_time))
    print(f"video {video} is ",end="")
    print(f"positive on frame {prediction_frame} (at time {predicition_time} seconds)" if pos else "negative", end="")
    if args.multilabel==True and pos:
        print(f" labels detected:",end="")
        if pred_labels[0]>0.:
            print(" (fire)",end="")
        if pred_labels[1]>0:
            print(" (smoke)",end="")
    print(f" with {max_mem_gb} GB of memory used")
    ########################################################
    f.close