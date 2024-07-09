import cv2, os, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer as timer
from PIL import Image
import matplotlib.pyplot as plt
from utils.models import models_conf
from torchvision import transforms
from transformers import pipeline

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method
if not os.path.exists(args.results):
   os.makedirs(args.results)

base_model = "kutralnet"
training_dataset = "fismo_black"
models_root = "models/saved" #'models/saved'
treshold = 1
stride = 24
weights = {'Fire': 1., 'Smoke': 0., 'Normal': 0.} #for the weighted average
tresholds = {'Fire': 1.1, 'Smoke': 0.9} #if frame is above treshold will be classified as positive
FPS = 24.0

model_name = "EdBianchi/vit-fire-detection"
pipe = pipeline(task="image-classification", model=model_name, device=0)

print('Loading pipeline', model_name)

################################################
# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    print("Processing video {}...".format(video))
    ret = True
    cap = cv2.VideoCapture(os.path.join(args.videos, video))
    
    pos = 0
    i = 0
    consecutive_positives = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        # Here you should add your code for applying your metho
        
        # Normalize data.
        image_r = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_r)
        predictions = pipe(im_pil)
        predictions = {x['label']:x['score'] for x in predictions}
        #print(predictions)
        
        fire = False
        for (label, t) in tresholds.items():
            if(predictions[label] >= t):
                fire = True
                print(f"Video {video} frame {i} is positive because {label} is {predictions[label]}")
                break
        
        #print("Video {} frame {} probability {}".format(video, i, fire_probability))
        if(fire):
            consecutive_positives += 1
        else:
            consecutive_positives = 0
        if consecutive_positives >= treshold:
            pos = 1
            break
        
        i += 1
        j = 0
        while ret and j < stride:
            ret = cap.grab()
            j += 1
            i += 1
        
        ########################################################
    cap.release()
    f = open(args.results+video+".txt", "w")
    # Here you should add your code for writing the results
    if pos:
        # print(i,stride,consecutive_positives)
        prediction_frame = i-(consecutive_positives-1)*(stride+1)
        predicition_time = int(np.ceil(prediction_frame/FPS))
        f.write(str(predicition_time))
    print(f"Video {video} is ",end="")
    print(f"positive on frame {prediction_frame} (at time {predicition_time} seconds)" if pos else "negative")
    ########################################################
    f.close