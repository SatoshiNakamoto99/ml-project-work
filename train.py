import os
from utils.training import train_model_k_fold
from utils.dataset import get_dataset_training_test_modes
from utils.training import collate_function_videos_to_images
from utils.models import get_config
import importlib


selected_model = "mobilenet-v3l"
#selected_model = "YOLOv8n-cls-bin"

model_config = get_config(selected_model)
model_module = importlib.import_module(model_config['module_name'])
ModelClass = getattr(model_module, model_config['class_name'])


preprocessing = model_config['preprocess']
augmentation = model_config['augmentation']
normalization = model_config['normalization']
loss_function = model_config['criterion']
OptimizerClass = model_config['optimizer']
optimizer_params = model_config['optimizer_params']
SchedulerClass = model_config['scheduler']
scheduler_params = model_config['scheduler_params']

batch_size = 64
epochs = 300
early_stopping_patience = 70
#experiment_name = "experiments/YOLOv8n-cls-bin/foggia-mod/"
experiment_name = "experiments/mobilenet-v3l/foggia-mod/"
selected_dataset = "foggia-mod"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

dataset_train_mode, dataset_test_mode = get_dataset_training_test_modes(selected_dataset, preprocessing, augmentation, normalization)

print(f"Training model {ModelClass} on dataset {selected_dataset}\n\twith optimizer {OptimizerClass} with params {optimizer_params}\n\twith loss function {loss_function}\nwith scheduler {SchedulerClass} with params {scheduler_params}")

train_model_k_fold(dataset_train_mode, dataset_test_mode, ModelClass, OptimizerClass,
                   optimizer_params, loss_function, epochs, batch_size,
                   early_stopping_patience, collate_function_videos_to_images(3),
                   experiment_name, num_classes = 2,  n_folds = 6, debug_prints = False, SchedulerClass=SchedulerClass, scheduler_params=scheduler_params if SchedulerClass is not None else {})

# train_model_k_fold(selected_dataset, ModelClass, OptimizerClass,
#                    optimizer_params, loss_function, epochs, batch_size,
#                    early_stopping_patience, collate_function_videos_to_images,
#                    experiment_name, num_classes = 2,  n_folds = 6, debug_prints = False)