
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import utils.engine as engine
import torch
from utils.dataset import get_dataset_training_test_modes
from torch.utils.data import default_collate
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class collate_function_videos_to_images():
    """Collate function for PyTorch DataLoader."""
    def __init__(self,frames_per_video):
        self.frames_per_video = frames_per_video
    def __call__(self,batch):
        """Collate function for PyTorch DataLoader.
        This function is necessary when using a VideoFrameDataset with a model that expects a batch of images as input."""
        # Transpose the data to shape (BATCH x NUM_FRAMES x CHANNELS x HEIGHT x WIDTH)
        # batch = list(zip(*batch))
        # batch[0] = torch.stack(batch[0], 0)
        #print("batch",batch[0][1])
        batch = default_collate(batch)
        #print("after collate",batch[1])
        
        #The following two lines are needed because the default_collate function doesn't work as expected with multilabels
        if not torch.is_tensor(batch[1]):
            batch[1] = torch.stack((batch[1][0], batch[1][1]), dim=1) #We need to convert the list of tensors to a tensor of tensors
        
        #print("after collate 2",batch[1])
        #print("BEFORE:",batch[1].shape)
        batch[0] = torch.flatten(batch[0], start_dim=0, end_dim=1) #Attacca i frame di video diversi
        batch[1] = torch.repeat_interleave(batch[1], self.frames_per_video, dim=0) #Attacca le label di video diversi
        #print("AFTER:",batch[1].shape)
        #print("")
        return batch

def train_model_k_fold(dataset_train_mode, dataset_test_mode, ModelClass, OptimizerClass, optimizer_params, loss_function,
                       epochs, batch_size, early_stopping_patience, collate_fn, experiment_name,
                       num_classes = 2,  n_folds = 6, debug_prints = False, SchedulerClass = None, scheduler_params = {}, stratified_kfold = False, task = "binary"):
    if stratified_kfold:
        labels = []
        for (X,y) in dataset_test_mode:
            labels.append(y)
        split = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=np.random.RandomState(seed=5423534)).split(dataset_train_mode,labels)
    else:
        split = KFold(n_splits=n_folds, shuffle=True, random_state=np.random.RandomState(seed=5423534)).split(dataset_train_mode)
    #split = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=np.random.RandomState(seed=5423534)).split(dataset_train_mode,dataset_train_mode.targets)
    for fold_index, (train_indexes, val_indexes) in enumerate(split):
        print(f"Fold {fold_index}:")
        if(debug_prints):
            print(f"  Train: index={train_indexes}")
            print(f"  Test:  index={val_indexes}")

        train_dataset = Subset(dataset_train_mode, train_indexes.tolist())
        val_dataset = Subset(dataset_test_mode, val_indexes.tolist())

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=16, pin_memory=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, pin_memory=True, collate_fn=collate_fn)

        model = ModelClass(classes=num_classes) #Il modello deve cambiare in base al fold
        optimizer = OptimizerClass(model.parameters(), **optimizer_params)

        scheduler = None
        if SchedulerClass is not None:
            scheduler = SchedulerClass(optimizer, **scheduler_params)
        
        results, best_model_results = engine.train(model, train_dataloader, val_dataloader, optimizer, loss_function, epochs, torch.device("cuda"), experiment_name, fold_index, early_stopping_patience, scheduler=scheduler, task=task)
        pd.DataFrame.from_dict(results).to_csv(f"{experiment_name}/fold_{fold_index}_training.csv", index=True, header=True)
        best_model_results["experiment_name"] = experiment_name.split("/", 1)[1]
        best_model_results["fold"] = fold_index
        best_model_results["model"] = ModelClass.__name__
        best_model_results["optimizer"] = optimizer.__class__.__name__
        best_model_results["scheduler"] = SchedulerClass.__name__ if SchedulerClass is not None else "None"
        best_model_results["loss_function"] = loss_function.__class__.__name__
        best_model_results["epochs"] = epochs
        best_model_results["batch_size"] = batch_size
        best_model_results["early_stopping_patience"] = early_stopping_patience
        best_model_results["n_folds"] = n_folds
        best_model_results.update(optimizer_params)
        best_model_results.update(scheduler_params)
        df = pd.DataFrame.from_dict(best_model_results)
        df.to_csv(f"experiments/best_models.csv", index=False, mode='a', header=not os.path.exists(f"experiments/best_models.csv"))
        df.to_csv(f"{experiment_name}/best_models.csv", index=False, mode='a', header=not os.path.exists(f"{experiment_name}/best_models.csv"))




def plot_acc(path_csv):
    # Load data from CSV file
    data = pd.read_csv(path_csv)

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Plot train and validation loss
    #ax.plot(data['train_loss'], label='Train Loss')
    #ax.plot(data['val_loss'], label='Validation Loss')

    # Plot train and validation accuracy
    ax.plot(data['train_acc'], label='Train Accuracy')
    ax.plot(data['val_acc'], label='Validation Accuracy')

    # Set axis labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    # Show the plot
    plt.show()

def plot_loss(path_csv):
    # Load data from CSV file
    data = pd.read_csv(path_csv)

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Plot train and validation loss
    #ax.plot(data['train_loss'], label='Train Loss')
    #ax.plot(data['val_loss'], label='Validation Loss')

    # Plot train and validation accuracy
    ax.plot(data['train_loss'], label='Train Loss')
    ax.plot(data['val_loss'], label='Validation Loss')

    # Set axis labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Show the plot
    plt.show()