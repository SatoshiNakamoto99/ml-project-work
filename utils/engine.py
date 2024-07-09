"""
Contains functions for training and testing a PyTorch model. 
This module was inspired by the following source: mrdbourke/pytorch-deep-learning.
"""
import torch
import torch.utils.data
import os
from torchmetrics import Accuracy

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               output_activation = None,
               task: str = "binary",
               num_classes: int = 2) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    accuracy = Accuracy(task=task, num_classes=num_classes, num_labels=num_classes).to(device)
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        if task == "multilabel":
            y = y.float()
        # 1. Forward pass
        y_pred = model(X)

        if(y_pred.dim() == 1):
            y_pred = y_pred[None,:] #BUGFIX for a mini-batch of size 1
        
        if output_activation is not None:
            y_pred = output_activation(y_pred).squeeze()
        
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        if task == "binary":
            output_probabilities = torch.nn.Softmax(dim=1)(y_pred) #In a binary classification problem we use softmax to get the probabilities
            y_pred_class = output_probabilities.argmax(dim=1) #The predicted label is the one with the highest probability
        elif task == "multilabel":
            output_probabilities = torch.nn.Sigmoid()(y_pred) #In a multilabel classification problem we use sigmoid to get the probabilities
            y_pred_class = (output_probabilities.detach() > 0.5).to(torch.float32) #The predicted label is calculated by thresholding the probabilities
            #metric = MultilabelAccuracy(num_labels=3)
        else:
            raise(Exception(f"task {task} not supported"))
        
        # Calculate and accumulate accuracy metric across all batches
        #y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        #train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        #print(f"TRAIN STEP: ypred: {y_pred.shape} y: {y.shape}")
        #print(f"TRAIN STEP: ypred: {y_pred} y: {y}")
        train_acc += accuracy(y_pred_class, y)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc.item() / len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
               output_activation = None,
               task: str = "binary",
               num_classes: int = 2) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data. Logits are passed to this function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    task: binary or multilabel
    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Turn on inference context manager
    with torch.inference_mode():
        val_loss = []
        val_pred_labels = []
        val_true_labels = []
        accuracyMetric = Accuracy(task=task, num_classes=num_classes, num_labels=num_classes).to(device)
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            if task == "multilabel":
                y = y.float()
            # 1. Forward pass
            output_logits = model(X)
            if(output_logits.dim() == 1):
                output_logits = output_logits[None,:] #BUGFIX for a mini-batch of size 1
            
            if output_activation is not None:
                output_logits_2 = output_activation(output_logits).squeeze() #TODO: ???
            
            # 2. Calculate and accumulate loss
            #print(output_logits,y.shape)
            loss = loss_fn(output_logits, y)# use torch.nn.BCEWithLogitsLoss for multilabel classification
            val_loss.append(loss)

            # 3. Calculate and accumulate accuracy
            if task == "binary":
                output_probabilities = torch.nn.Softmax(dim=1)(output_logits) #In a binary classification problem we use softmax to get the probabilities
                pred_labels = output_probabilities.argmax(dim=1) #The predicted label is the one with the highest probability
            elif task == "multilabel":
                output_probabilities = torch.nn.Sigmoid()(output_logits) #In a multilabel classification problem we use sigmoid to get the probabilities
                pred_labels = (output_probabilities.detach() > 0.5).to(torch.float32) #The predicted label is calculated by thresholding the probabilities
                #metric = MultilabelAccuracy(num_labels=3)
            else:
                raise(Exception(f"task {task} not supported"))
            val_true_labels.append(y)
            val_pred_labels.append(pred_labels)
            
        val_true_labels = torch.concatenate(val_true_labels)
        val_pred_labels = torch.concatenate(val_pred_labels)
        
        val_loss = torch.stack(val_loss).mean().item()
        val_accuracy = accuracyMetric(val_pred_labels, val_true_labels)
    
    return val_loss, val_accuracy.item()

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          experiment_name: str,
          fold_index: int,
          early_stopping_patience: int,
               output_activation = None,
               scheduler = None,
               task: str = "binary",
               num_classes: int = 2) -> Tuple[Dict[str, List],Dict]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    experiment_name: A string indicating the name of the experiment.
    fold_index: An integer indicating the index of the fold.
    early_stopping_patience: An integer indicating how many epochs to wait before stopping training if the validation loss doesn't improve.
    scheduler: A PyTorch scheduler to adjust the learning rate during training.
    
    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
    }
    best_model_results = {"train_loss": None,
                          "train_acc": None,
                          "val_loss": None,
                          "val_acc": None}
    # Make sure model on target device
    model = model.to(device)
    
    early_stopping_counter = early_stopping_patience
    min_val_loss = 1e10
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device,
                                            output_activation=output_activation,
                                            task=task,
                                            num_classes=num_classes)
        val_loss, val_acc = val_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device,
                                            output_activation=output_activation,
                                            task=task,
                                            num_classes=num_classes)

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        generalization_gap = val_loss - train_loss
        
        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f}"
        )
        
        # save the best model and check the early stopping criteria
        if val_loss < min_val_loss: # save the best model
            min_val_loss = val_loss
            early_stopping_counter = early_stopping_patience # reset early stopping counter
            torch.save(model.state_dict(), os.path.join(experiment_name,'fold_{}_best_model.pth'.format(fold_index)))
            print("- saved best model: val_loss =", val_loss, "val_accuracy =", val_acc)
            best_model_results = {"train_loss": [train_loss],
                                  "train_acc": [train_acc],
                                  "val_loss": [val_loss],
                                  "val_acc": [val_acc]}
        if epoch>0: # early stopping counter update
            if generalization_gap > results["val_loss"][-1] - results["val_loss"][-1] or val_loss >= min_val_loss:
                early_stopping_counter -= 1 # update early stopping counter
                print("Generalization gap is increased: ",generalization_gap, "Decreasing early stopping counter: ",early_stopping_counter)
            else:
                early_stopping_counter = early_stopping_patience # reset early stopping counter
        
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Early stopping
        if early_stopping_counter == 0: 
            break
    #print(results,best_model_results)
    # Return the filled results at the end of the epochs
    return results, best_model_results
