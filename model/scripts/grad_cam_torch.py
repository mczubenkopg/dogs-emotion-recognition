import time
import warnings
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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


def train(model, max_epochs, train_flag=True, verbose=False):
    training_loader, val_loader = get_datasets()
    # Set up loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    tb_writer = SummaryWriter(f'../runs/{model.model_name}')

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    time_stamp = time.time()
    if not train_flag:
        max_epochs = 1

    # Epoch loop
    for epoch in range(max_epochs):
        gc.collect()
        epoch_loss = 0.
        if verbose:
            print(f'Running epoch no. {epoch}')

        # Training procedure
        if train_flag:
            # Make sure gradient tracking is on, and do a pass over the data
            # Train
            model.train()
            correct, total = 0, 0
            loss = None
            train_loos = 0.
            # Batch loop
            for batch_idx, (inputs, targets)  in enumerate(training_loader):
                batch_loss = 0.
                # Every data instance is an input + label pair
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_fcn(predictions, targets)
                loss.backward()
                # Adjust learning weights
                optimizer.step()
                _, predicted = torch.max(predictions, 1)  # get predicted class indices
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Gather data and report
                batch_loss += loss.item()
                train_loos += loss.item()
                if batch_idx % 1000 == 999:
                    last_loss = batch_loss / 1000  # loss per batch
                    if verbose:
                        print(f'Batch {batch_idx} loss: {last_loss}')
                    tb_writer.add_scalar(f'Batch {train_flag.name} loss', last_loss,
                                              batch_idx)
                    batch_loss = 0.
            train_loos /= len(training_loader)  # loss per batch
            if verbose:
                epoch_accuracy = correct / total
                print(f'Epoch {epoch}, {train_flag.name} loss = {epoch_loss} | accuracy: {epoch_accuracy}')
            tb_writer.add_scalar(f'{train_flag.name} loss', epoch_loss, epoch)
            # or not to train
            avg_train_loss = epoch_loss

        # Evaluation
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            correct, total = 0, 0
            loss = None
            validation_loss = 0.
            # Batch loop
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                batch_loss = 0.
                # Every data instance is an input + label pair
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(inputs)
                loss = loss_fcn(predictions, targets)

                _, predicted = torch.max(predictions, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Gather data and report
                batch_loss += loss.item()
                validation_loss += loss.item()
                if batch_idx % 1000 == 999:
                    last_loss = batch_loss / 1000  # loss per batch
                    if verbose:
                        print(f'Batch {batch_idx} loss: {last_loss}')
                    tb_writer.add_scalar(f'Batch {batch_idx} loss', last_loss)
                    batch_loss = 0.
            validation_loss /= len(val_loader)  # loss per batch
            if verbose:
                epoch_accuracy = correct / total
                print(f'Epoch {epoch}, {model.name} loss = {epoch_loss} | accuracy: {epoch_accuracy}')
            tb_writer.add_scalar(f'model.name} loss', epoch_loss, epoch)
            # or not to train
            avg_train_loss = epoch_loss

        if train_flag:
            # Log the running loss averaged per batch for both training and validation,
            tb_writer.add_scalars('Training vs. Validation Loss',
                                       {'Training': avg_train_loss,
                                        'Validation': avg_validation_loss},
                                       epoch)
            tb_writer.flush()
            # Tracks best performance, and save the model's state
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                best_validation_epoch = epoch
                best_weights = deepcopy(model.state_dict())
                model_path = f'./weights/{model.model_name}_best_val.pth'

                if not Path(model_path).parent.exists():
                    Path(model_path).parent.mkdir(exist_ok=True, parents=True)

                torch.save(model.state_dict(), model_path)
    model.load_state_dict(best_weights)
    if save:
        model_path = f'./weights/{model.model_name}_{round(best_validation_loss, 5)}__{time.strftime("%y.%b.%d", time.gmtime())}.pth'
        torch.save(model.state_dict(), model_path)
    if verbose:
        time_elapsed = time.time() - time_stamp
        print(
            f'Training complete in {time_elapsed // 60:.6f}m {time_elapsed % 60:.6f}s')
    return best_validation_loss, best_validation_epoch

    # Train the model
    num_epochs = 25
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs} loss: {loss.item()}')

    # Save the trained model
    torch.save(model, model.__name__)

    # After training, evaluate on the validation set
    model.eval()

    pass


    def one_epoch(model, epoch_index, train_flag, loader):
        """
        One epoch training
        """

        return epoch_loss


def get_datasets(data_dir=Path('./Dataset'), data_transforms=transforms.Compose([])):
    # Load your dataset
    image_datasets = ImageFolder(data_dir, transform=data_transforms)

    # Determine the sizes of the training and validation sets
    train_size = int(0.8 * len(image_datasets))
    val_size = len(image_datasets) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(image_datasets, [train_size, val_size])

    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


# model.eval()
# image_url = "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0"
# img = np.array(Image.open(requests.get(image_url, stream=True).raw))
# img = cv2.resize(img, (224, 224))
# img = np.float32(img) / 255
# input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
# # The target for the CAM is the Bear category.
# # As usual for classication, the target is the logit output
# # before softmax, for that category.
# targets = [ClassifierOutputTarget(295)]
# target_layers = [model.layer4]
# with GradCAM(model=model, target_layers=target_layers) as cam:
#     grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
#     cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
# cam = np.uint8(255*grayscale_cams[0, :])
# cam = cv2.merge([cam, cam, cam])
# images = np.hstack((np.uint8(255*img), cam , cam_image))
# Image.fromarray(images)

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

models = [models.resnet50(pretrained=True), models.densenet121(pretrained=True), models.inception_v3(pretrained=True),
          models.mobilenet_v2(pretrained=True),
          models.mnasnet1_3(pretrained=True), models.efficientnet_b0(pretrained=True),
          models.convnext_base(pretrained=True), models.shufflenet_v2_x1_0(pretrained=True),
          models.resnext50_32x4d(pretrained=True), models.wide_resnet50_2(pretrained=True),
          torch.hub.load('moskomule/senet.pytorch','se_resnet50', pretrained=True)]
