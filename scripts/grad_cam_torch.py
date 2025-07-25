import subprocess
import time
import warnings
from copy import deepcopy
from pathlib import Path
import torch
from pytorch_grad_cam.metrics.road import ROADCombined
from torchinfo import summary
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torcheval.metrics import (MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall,
                               MulticlassAUROC,
                               MulticlassConfusionMatrix)

# from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix, Precision, Recall, Specificity, ROC, AUROC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import numpy as np
import cv2
import gc
import torch.hub

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print('Running on the GPU')
else:
    DEVICE = torch.device('cpu')
    print('Running on the CPU')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
class_number = 5


def train(model, max_epochs, train_flag=True, verbose=False, callback_break=20):
    training_loader, val_loader = get_datasets()
    if not train_flag:
        max_epochs = 1
        print(f'Running evaluation on {model.name}')
    else:
        print(f'Running training on {model.name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    torcheval_metrics = dict(acc=MulticlassAccuracy(device=device), f1_score=MulticlassF1Score(device=device),
                             precision=MulticlassPrecision(device=device), recall=MulticlassRecall(device=device),
                             auroc=MulticlassAUROC(device=device, num_classes=class_number))
                             # confusion_matrix=MulticlassConfusionMatrix(device=device, num_classes=class_number))
    # torch_metrics = dict(
    #     acc_metric=Accuracy(task="multiclass", num_classes=class_number),
    #     f1_metric=F1Score(task="multiclass", num_classes=class_number),
    #     precision_macro=Precision(task="multiclass", average='macro', num_classes=class_number),
    #     precision_micro=Precision(task="multiclass", average='micro', num_classes=class_number),
    #     recall_macro=Recall(task="multiclass", average='macro', num_classes=class_number),
    #     recall_micro=Recall(task="multiclass", average='micro', num_classes=class_number),
    #     specificity_macro=Specificity(task="multiclass", average='macro', num_classes=class_number),
    #     specificity_micro=Specificity(task="multiclass", average='micro', num_classes=class_number),
    #     roc=ROC(task='multilabel', num_labels=class_number),
    #     auroc=AUROC(task="multiclass", num_classes=class_number),
    #     confusion_mat=ConfusionMatrix(task="multiclass", num_classes=class_number))

    tb_writer = SummaryWriter(f'../runs/{model.name}')

    # Move model to GPU if available

    model = model.to(device)

    time_stamp = time.time()
    # Init variables
    best_weights = {}
    best_validation_loss = 1000
    n_epoch_worse = 0

    # Epoch loop
    for epoch in range(max_epochs):
        gc.collect()
        if verbose:
            print(80 * '-')
            print(f'Epoch: {epoch}.')

        # Training procedure
        if train_flag:
            model.train()
            # Reset train epoch variables
            for metric in torcheval_metrics.values():
                metric.reset()
            correct, total = 0, 0
            train_loos = 0.

            # Batch loop
            for batch_idx, (inputs, targets) in enumerate(training_loader):
                batch_loss = 0.
                # Every data instance is an input and targets
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                # Training core
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_fcn(predictions, targets)
                loss.backward()
                optimizer.step()

                # Metrics
                for metric in torcheval_metrics.values():
                    metric.update(predictions, targets)

                # Manual metric evaluation
                _, predicted = torch.max(predictions, 1)  # get predicted class indices
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Gather data and report
                batch_loss += loss.item()
                train_loos += loss.item()
                if batch_idx % 100 == 99:
                    last_loss = batch_loss / 100  # loss per batch
                    if verbose:
                        print(f'Batch {batch_idx} loss: {last_loss}')

            train_loos /= len(training_loader)  # loss per batch
            for key, metric in torcheval_metrics.items():
                m = metric.compute()
                if key != 'confusion_matrix':
                    tb_writer.add_scalar(f'Train/{key}', m, epoch)
                if verbose:
                    print(f'Train/{key} = {m}', end=' | ')
            train_accuracy = correct / total
            tb_writer.add_scalar(f'Train/loss', train_loos, epoch)
            tb_writer.add_scalar(f'Train/accuracy_man', train_accuracy, epoch)
            if verbose:
                print()
                print(f'Train/loss = {train_loos} | Train/accuracy: {train_accuracy}')
                # Evaluation
        model.eval()

        # Clear metrics for validation
        validation_loss = 0.0
        correct, total = 0, 0
        for metric in torcheval_metrics.values():
            metric.reset()

        with torch.no_grad():
            # print(f'Epoch: {epoch} -- validation')
            # Batch loop
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                batch_loss = 0.
                # Every data instance is an input + label pair
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(inputs)
                loss = loss_fcn(predictions, targets)

                # Metrics
                for metric in torcheval_metrics.values():
                    metric.update(predictions, targets)

                # Manual metric evaluation
                _, predicted = torch.max(predictions, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Gather data and report
                batch_loss += loss.item()
                validation_loss += loss.item()
                if batch_idx % 100 == 99:
                    last_loss = batch_loss / 100  # loss per batch
                    if verbose:
                        print(f'Batch {batch_idx} loss: {last_loss}')

            validation_loss /= len(val_loader)  # loss per batch
            for key, metric in torcheval_metrics.items():
                m = metric.compute()
                if key != 'confusion_matrix':
                    tb_writer.add_scalar(f'Val/{key}', m, epoch)
                if verbose:
                    print(f'Train/{key} = {m}', end=' | ')
            val_accuracy = correct / total
            tb_writer.add_scalar(f'Val/loss', validation_loss, epoch)
            tb_writer.add_scalar(f'Val/accuracy_man', val_accuracy, epoch)
            if verbose:
                print()
                print(f'Val/loss = {validation_loss} | Val/accuracy: {val_accuracy}')
            # or not to train

        if train_flag:
            # Log the running loss averaged per batch for both training and validation,
            tb_writer.add_scalars('Training vs. Validation Loss',
                                  {'Training': train_loos,
                                   'Validation': validation_loss},
                                  epoch)
            # Tracks best performance, and save the model's state
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_epoch = epoch
                n_epoch_worse = 0
                best_weights = deepcopy(model.state_dict())
                model_path = f'./weights/{model.name}_best_val.pth'
                if not Path(model_path).parent.exists():
                    Path(model_path).parent.mkdir(exist_ok=True, parents=True)
                torch.save(model.state_dict(), model_path)
            else:
                n_epoch_worse += 1
                print(f'Number of epochs without improve: {n_epoch_worse}')
            if n_epoch_worse == callback_break:
                print(f'Validation loss did not improve in {n_epoch_worse} epochs -- breaking')
                break
    if best_weights:
        model.load_state_dict(best_weights)
    if verbose:
        time_elapsed = time.time() - time_stamp
        print(f'Training complete in {time_elapsed // 60:.6f}m {time_elapsed % 60:.6f}s')
    tb_writer.flush()
    tb_writer.close()
    return model


def get_datasets(data_dir=Path('../dataset'), data_transforms=transforms.Compose([])):
    # Load your dataset
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_datasets = ImageFolder(data_dir, transform=data_transforms)

    # Determine the sizes of the training and validation sets
    train_size = int(0.8 * len(image_datasets))
    val_size = len(image_datasets) - train_size
    print(f'Using training dataset with size of {train_size}')
    print(f'Using validation dataset with size of {val_size}')

    # Split the dataset
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_dataset, val_dataset = random_split(image_datasets, [train_size, val_size], generator=generator)

    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader

def get_eval(eval_dir=Path('./eval_data')):
    # Load your dataset
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_datasets = ImageFolder(eval_dir, transform=data_transforms)
    print(f'Using eval dataset with size of {len(image_datasets)}')
    # Create DataLoaders for the training and validation sets
    eval_loader = DataLoader(image_datasets, batch_size=1, shuffle=False)
    return eval_loader


def grad_cams(model):
    model.eval()
    eval_dataset = get_eval()
    for org_input, org_target in eval_dataset:
        img = org_input.to_numpy()
        targets = [ClassifierOutputTarget(org_target)]
        target_layers = [model.layer4]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cams = cam(input_tensor=org_input, targets=targets)
            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        images = np.hstack((np.uint8(255*img), cam , cam_image))
        Image.fromarray(images)

    # image_url = "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0"
# img = np.array(Image.open(requests.get(image_url, stream=True).raw))
# img = cv2.resize(img, (224, 224))
# img = np.float32(img) / 255
# input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
# # The target for the CAM is the Bear category.
# # As usual for classication, the target is the logit output before softmax, for that category.
# targets = [ClassifierOutputTarget(295)]
# target_layers = [model.layer4]
# with GradCAM(model=model, target_layers=target_layers) as cam:
#     grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
#     cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
# cam = np.uint8(255*grayscale_cams[0, :])
# cam = cv2.merge([cam, cam, cam])
# images = np.hstack((np.uint8(255*img), cam , cam_image))
# Image.fromarray(images)

def run_tensorboard():
    subprocess.run(["tensorboard", "--logdir=../runs"])


def modify_model_output(model, num_classes=5):
    """
    Replace the final classification layer with a Linear + Softmax
    """
    for param in model.parameters():
        param.requires_grad = False
    # ResNet or others with .fc
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    # Models with .classifier
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            # Find last linear layer in the classifier
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    in_features = model.classifier[i].in_features
                    model.classifier[i] = nn.Linear(in_features, num_classes)
                    break
        elif isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True
    # Vision Transformers or models with .heads.head
    elif hasattr(model, 'heads'):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        for param in model.heads.head.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Unknown model architecture – customize this function for your case.")
    return model


# Showing the metrics on top of the CAM :
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "Remove and Debias", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return visualization


def benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=281):
    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, use_cuda=True))]

    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    targets = [ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputSoftmaxTarget(category)]

    visualizations = []
    percentiles = [10, 50, 90]
    for name, cam_method in methods:
        with cam_method:
            attributions = cam_method(input_tensor=input_tensor,
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[0, :]
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score = scores[0]
        visualization = show_cam_on_image(cat_and_dog, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualizations.append(visualization)
    return Image.fromarray(np.hstack(visualizations))


# cat_and_dog_image_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
# cat_and_dog = np.array(Image.open(requests.get(cat_and_dog_image_url, stream=True).raw))
# cat_and_dog = np.float32(cat_and_dog) / 255
# input_tensor = preprocess_image(cat_and_dog, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# target_layers = [model.layer4]
#
# model.cuda()
# input_tensor = input_tensor.cuda()
# np.random.seed(42)
# benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)

if __name__ == '__main__':
    models_list = ['resnet50']
    # models_list = models.list_models(module=models)
    for m in models_list:
        model = models.get_model(m, weights="DEFAULT")
        model = modify_model_output(model, num_classes=class_number)
        model.name = m
        print(model)
        model = train(model, max_epochs=500, train_flag=True, verbose=True)
        grad_cams(model)