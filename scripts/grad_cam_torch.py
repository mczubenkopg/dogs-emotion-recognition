import subprocess
import time
import warnings
from collections import defaultdict, namedtuple
from copy import deepcopy
from pathlib import Path
import torch
from pytorch_grad_cam.metrics.road import ROADCombined
from torchinfo import summary
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torcheval.metrics import (MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall,
                               MulticlassAUROC,
                               MulticlassConfusionMatrix)

# from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix, Precision, Recall, Specificity, ROC, AUROC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, \
    LayerCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
import numpy as np
import cv2
import gc
import torch.hub
import requests
import matplotlib.pyplot as plt

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
ModelHub = namedtuple('ModelHub', ['repo', 'name'])

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad
}
class_numbers = {'angry': 0, 'curious': 1, 'happy': 2, 'sad': 3, 'sleepy': 4}
numbers_class = {n:k for k, n in class_numbers.items()}


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
    best_validation_epoch = 0
    n_epoch_worse = 0

    # Epoch loop
    for epoch in range(max_epochs):
        gc.collect()
        if verbose:
            print(80 * '-')
            print(f'Epoch: {epoch} model {model.name}.')

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
                model_path = f'../weights/{model.name}_best_val.pth'
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
        model_path = f'../weights/{model.name}_e{best_validation_epoch}_v{best_validation_loss}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Best validation score {best_validation_loss} in {best_validation_epoch} epoch')
    if verbose:
        time_elapsed = time.time() - time_stamp
        print(f'Training complete in {time_elapsed // 60:.6f}m {time_elapsed % 60:.6f}s')
    tb_writer.flush()
    tb_writer.close()
    return model


def load_model(model_name):
    if isinstance(model_name, str):
        model = models.get_model(model_name, weights="DEFAULT")
    elif isinstance(model_name, ModelHub):
        model = torch.hub.load(model_name.repo, model_name.name, pretrained=True)
    model = modify_model_output(model, train=False, num_classes=class_number)
    model.name = model_name
    model_path = Path(f'../weights/{model.name}_best_val.pth')
    if model_path.exists():
        best_weights = torch.load(model_path)
        model.load_state_dict(best_weights)
        return model
    else:
        return None


def train_model(model_name):
    if isinstance(model_name, str):
        model = models.get_model(model_name, weights="DEFAULT")
    elif isinstance(model_name, ModelHub):
        model = torch.hub.load(model_name.repo, model_name.name, pretrained=True)
    model = modify_model_output(model, train=True, num_classes=class_number)
    model.name = model_name
    model = train(model, max_epochs=500, train_flag=True, verbose=True)
    return model


def get_balanced_subset(dataset, samples_per_class):
    class_indices = defaultdict(list)
    # Group indices by class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    # Sample fixed number per class
    selected_indices = []
    for label, indices in class_indices.items():
        selected = indices[:samples_per_class]
        selected_indices.extend(selected)
    return Subset(dataset, selected_indices)


def get_datasets(data_dir=Path('../dataset'), data_transforms=transforms.Compose([])):
    # Load your dataset
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_datasets = ImageFolder(data_dir, transform=data_transforms)

    # TODO balancing
    # balanced_subset = get_balanced_subset(image_datasets, samples_per_class=1500)

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


def get_eval(eval_dir=Path('../eval_data')):
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


def get_target_layer(model_name, model):
    if model_name.startswith("resnet") or model_name.startswith("resnext") or "wide_resnet" in model_name:
        return [model.layer4[-1]]
    elif model_name.startswith("vgg") or model_name.startswith("alexnet"):
        return [model.features[-1]]
    elif model_name.startswith("mobilenet"):
        return [model.features[-1]]
    elif model_name.startswith("densenet"):
        return [model.features[-1]]
    elif model_name.startswith("shufflenet"):
        return [model.conv5]
    elif model_name.startswith("efficientnet"):
        return [model.features[-1]]
    elif model_name.startswith("vit"):
        return [model.encoder.ln]
    elif model_name.startswith("deit"):
        return [model.blocks[-1].norm1]
    elif model_name.startswith("swin"):
        return [model.layers[-1].blocks[-1].norm2]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    input_tensor = transform(image).unsqueeze(0)
    rgb_image = np.array(image).astype(np.float32) / 255.0
    return input_tensor, rgb_image


def grad_cams(model, eval_path=Path('../eval_data')):
    for param in model.parameters():
        param.requires_grad = True
    model.eval()
    model_path = Path.joinpath(eval_path.parent, f'./results')
    if not model_path.exists():
        model_path.mkdir()
    model_path = Path.joinpath(model_path, f'./{model.name}')
    if not model_path.exists():
        model_path.mkdir()
    eval_dataset = get_eval()
    to_pil = transforms.ToPILImage()
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    for i, (org_input, org_target) in enumerate(eval_dataset):
        c = numbers_class[int(org_target)]
        rgb_image = np.array(to_pil(torch.clamp(unnormalize(org_input[0]), 0, 1))).astype(np.float32) / 255.
        targets = [ClassifierOutputTarget(org_target)]
        target_layers = get_target_layer(model.name, model)
        for cam_name, cam in CAM_METHODS.items():
            cam_path = Path.joinpath(model_path, f'./{cam_name}')
            if not cam_path.exists():
                cam_path.mkdir()
            if model.name.startswith("vit") or model.name.startswith("swin") or model.name.startwith("deit"):
                if cam_name == "ablationcam":
                    cam = cam(model=model, target_layers=target_layers, reshape_transform=reshape_transform, ablation_layer=AblationLayerVit())
                elif cam_name == "fullgrad":
                    continue
                else:
                    cam = cam(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
            else:
                cam = cam(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=org_input, targets=targets)[0]
            cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            images = np.hstack((np.uint8(255 * rgb_image), cam_image))
            images = Image.fromarray(images)
            cam_image = Image.fromarray(cam_image)
            images.save(Path.joinpath(cam_path, Path(f'join_{i}_{c}_{cam_name}.jpg')))
            cam_image.save(Path.joinpath(cam_path, Path(f'cam_{i}_{c}_{cam_name}.jpg')))
        image = Image.fromarray(np.uint8(255 * rgb_image))
        image.save(Path.joinpath(model_path, Path(f'org_{i}_{c}.jpg')))



def generate_gradcam(model, image_path, num_classes=5, class_index=None):
    """
    Based on model but from pure image
    """
    model.eval()
    input_tensor, rgb_image = load_image(image_path)
    target_layers = get_target_layer(model.name, model)
    for cam_name, cam in CAM_METHODS:
        cam = cam(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(class_index)] if class_index is not None else None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        images = np.hstack((np.uint8(255 * rgb_image), cam, cam_image))
        result = Image.fromarray(images)
        result.save(Path.joinpath(image_path.parent,
                                  image_path.name.replace('.', f'_{model.name}_{cam_name}.')))


def run_tensorboard():
    subprocess.run(["tensorboard", "--logdir=../runs"])


def modify_model_output(model, train=True, num_classes=5):
    """
    Replace the final classification layer with a Linear + Softmax
    """
    if train:
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
    elif hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Unknown model architecture – customize this function for your case.")
    return model


def gradcams_from_files():
    models_list = ['resnet50']
    eval_path = Path('../eval_data')
    class_numbers = {'angry': 0, 'curious': 1, 'happy': 2, 'sad': 3, 'sleepy': 4}
    for m in models_list:
        model = load_model(m)
        for image_path in eval_path.rglob('*.jpg'):
            generate_gradcam(model, image_path, class_index=class_numbers[image_path.parent.name])


# Showing the metrics on top of the CAM :
# TODO CAM metrics
# def visualize_score(visualization, score, name, percentiles):
#     visualization = cv2.putText(visualization, name, (10, 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
#     visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
#     visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
#     visualization = cv2.putText(visualization, "Remove and Debias", (10, 70),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
#     visualization = cv2.putText(visualization, f"{score:.5f}", (10, 85),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
#     return visualization
#
#
# def benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=281):
#     methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=True)),
#                ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)),
#                ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=True)),
#                ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=True)),
#                ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, use_cuda=True))]
#
#     cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
#     targets = [ClassifierOutputTarget(category)]
#     metric_targets = [ClassifierOutputSoftmaxTarget(category)]
#
#     visualizations = []
#     percentiles = [10, 50, 90]
#     for name, cam_method in methods:
#         with cam_method:
#             attributions = cam_method(input_tensor=input_tensor,
#                                       targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
#         attribution = attributions[0, :]
#         scores = cam_metric(input_tensor, attributions, metric_targets, model)
#         score = scores[0]
#         visualization = show_cam_on_image(cat_and_dog, attribution, use_rgb=True)
#         visualization = visualize_score(visualization, score, name, percentiles)
#         visualizations.append(visualization)
#     return Image.fromarray(np.hstack(visualizations))

def train_grad_all():
    models_list = ["resnet18", "resnet50", "resnext50_32x4d",
                   "vgg16", "alexnet", "mobilenet_v2", "mobilenet_v3_large",
                   "densenet121", "shufflenet_v2_x1_0", "efficientnet_b0",
                   "vit_b_16", "swin_t", ModelHub(repo='facebookresearch/deit:main', name='deit_tiny_patch16_224')]
    # All torchvision - to run
    # models_list = models.list_models(module=models)
    for m in models_list:
        # model = load_model(m)
        model = train_model(m)
        if model:
            grad_cams(model)


def train_grad_models(models_list):
    """
    Train models and calculate all gradcams
    """
    for m in models_list:
        # model = load_model(m)
        model = train_model(m)
        if model:
            grad_cams(model)

def calc_gradcams(models_list):
    """
    Load trained models (in folder ./weights/model_best_val.pth) and calculates all gradcams
    """
    for m in models_list:
        model = load_model(m)
        if model:
            grad_cams(model)


if __name__ == '__main__':
    # models_list = models.list_models(module=models) - TODO
    all_models_list = ["resnet18", "resnet50", "resnext50_32x4d",
               "vgg16", "alexnet", "mobilenet_v2", "mobilenet_v3_large",
                   "densenet121", "shufflenet_v2_x1_0", "efficientnet_b0",
                   "vit_b_16", "swin_t", ModelHub(repo='facebookresearch/deit:main', name='deit_tiny_patch16_224')]
    models_list = all_models_list[-1:]
    train_grad_models(models_list)
    # train_grad_all()