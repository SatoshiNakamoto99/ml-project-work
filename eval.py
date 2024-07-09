import  os, argparse
from striprtf.striprtf import rtf_to_text
from timeit import default_timer as timer
import pandas as pd
from utils.models import models_conf


def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--predictions", type=str, default='results/', help="Classification results")
    parser.add_argument("--gt", type=str, default='gt/', help="Ground truth")
    parser.add_argument("--fold", type=int, default=None, help="Fold number")
    args = parser.parse_args()
    return args

args = init_parameter()

FPS = 24
delta_t = 5

TP = 0
FP = 0
FN = 0
TN = 0
delay = 0
average_delay = 0

# foreach rtf file in gt folder
for dir in os.listdir(args.gt):
    annotations_path = os.path.join(args.gt, dir)
    predictions_path = os.path.join(args.predictions, dir)
    
    for annotation in os.listdir(annotations_path):
        if not annotation.endswith(".rtf"):
            continue

        # open the gt file
        annotation_fp = open(os.path.join(annotations_path, annotation), "r")
        text = rtf_to_text(annotation_fp.read())
        annotation_fp.close()

        if len(text):
            label = 1
            #start_frame = int(text.split(",")[0]) #TODO: this is in seconds, not frames
            #start_time = start_frame/FPS
            #if start_frame == 0:
            #    start_frame = 1
            start_time = int(text.split(",")[0])
        else:
            label = 0
            #start_frame = 1
            start_time = 0
        
        try:
            pred_fp = open(os.path.join(predictions_path, annotation.split(".")[0]+".mp4.txt"), "r")
            text = pred_fp.read()
            pred_fp.close()
        except FileNotFoundError:
            print(annotation.split(".")[0]+".mp4.txt:", "File not found")
            continue
        if len(text):
            pred_label = 1
            #pred_start_frame = int(text)
            #pred_start_time = pred_start_frame/FPS
            pred_start_time = int(text)
        else:
            pred_label = 0
            #pred_start_frame = 1
            pred_start_time = 0


        if label == 1:
            if pred_label == 1 and pred_start_time >= start_time - delta_t:
                print(annotation.split(".")[0]+".mp4.txt:", f"True positive: pred_start_time={pred_start_time} start_time={start_time}")
                TP += 1
                delay += abs(pred_start_time - start_time)
            else:
                print(annotation.split(".")[0]+".mp4.txt:", f"False negative: start_time={start_time}")
                FN += 1
        else:
            if pred_label == 1:
                print(annotation.split(".")[0]+".mp4.txt:", f"False positive: pred_start_time={pred_start_time}")
                FP += 1
            else:
                print(annotation.split(".")[0]+".mp4.txt:", "True negative")
                TN += 1

precision = TP/(TP+FP) if TP+FP > 0 else 0.
recall = TP/(TP+FN) if TP+FN > 0 else 0.
average_delay = delay/TP if TP > 0 else 0.
normalized_average_delay = max(0,60 - average_delay)/60
accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0. 

print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)
print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)
print("Average delay {} seconds ({} normalized)".format(average_delay, normalized_average_delay))

df = pd.DataFrame({'predictions': args.predictions, 'fold': args.fold, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'average_delay': average_delay, 'normalized_average_delay': normalized_average_delay},index=[0])
df.to_csv(os.path.join(args.predictions, f'fold_{args.fold}_results.csv' if args.fold is not None else 'results.csv'))

import sys
with open(os.path.join(args.predictions, 'results.txt'), 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("TP: ", TP)
    print("FP: ", FP)
    print("FN: ", FN)
    print("TN: ", TN)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", accuracy)
    print("Average delay {} seconds ({} normalized)".format(average_delay, normalized_average_delay))
