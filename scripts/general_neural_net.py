import gc
from pathlib import Path

import torch
import torch.nn as nn
import time
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter


class GeneralModule(nn.Module):
    INPUTS = 10
    OUTPUTS = 3

    def __init__(self, verbose: bool = False, set_loaders=False):
        super().__init__()
        self.model_name = self.__class__.__name__
        self.optimizer = torch.optim.Adam
        self.loss_fcn = nn.CrossEntropyLoss()
        self.tb_writer = SummaryWriter(f'../runs/{self.model_name}')
        self.layers = nn.ModuleList()

        self.training_loader = None
        self.validation_loader = None
        self.test_loader = None

        self.verbose = verbose
        if set_loaders:
            self.auto_set_loaders(batch_size=16)

    def set_loaders(self, training_loader, validation_loader, test_loader):
        """
        Sets loaders from external
        """

        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader


    def forward(self, x):
        """
        Forward function
        """
        for layer in self.layers:
            x = layer(x)

        return x
        # else:
        #     return nn.Softmax(dim=1)(x)

    def loss_function(self, outputs, predictions):
        """
        Loss function for future overriding
        :param outputs: targets of model
        :param predictions: results of model
        :return: loss of batch
        """
        return self.loss_fcn(predictions, outputs)

    def one_epoch(self, epoch_index, train_flag: DatasetEnum = DatasetEnum.TRAIN):
        """
        One epoch training
        """
        epoch_loss = 0.
        batch_loss = 0.
        loss = None
        if train_flag == DatasetEnum.TRAIN:
            loader = self.training_loader
            self.train()
        elif train_flag == DatasetEnum.VALIDATE:
            loader = self.validation_loader
            self.eval()
        else:
            loader = self.test_loader
            self.eval()

        correct, total = 0, 0
        #TODO add metrics
        for batch_idx, data in enumerate(loader):
            # Every data instance is an input + label pair
            inputs, targets = data
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            if train_flag == DatasetEnum.TRAIN:
                self.optimizer.zero_grad()
                predictions = self(inputs)
                loss = self.loss_fcn(predictions, targets)
                loss.backward()
                # Adjust learning weights
                self.optimizer.step()

                _, predicted = torch.max(predictions, 1)  # get predicted class indices
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

            elif train_flag == DatasetEnum.VALIDATE or train_flag == DatasetEnum.TEST:
                predictions = self(inputs)
                loss = self.loss_fcn(predictions, targets)

                _, predicted = torch.max(predictions, 1)  # get predicted class indices
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

            # Gather data and report
            batch_loss += loss.item()
            epoch_loss += loss.item()
            if batch_idx % 1000 == 999:
                last_loss = batch_loss / 1000  # loss per batch
                if self.verbose:
                    print(f'Batch {batch_idx} loss: {last_loss}')
                self.tb_writer.add_scalar(f'Batch {train_flag.name} loss', last_loss,
                                          batch_idx)
                batch_loss = 0.
        epoch_loss /= len(loader)  # loss per batch
        if self.verbose:
            epoch_accuracy = correct / total
            print(f'Epoch {epoch_index}, {train_flag.name} loss = {epoch_loss} | accuracy: {epoch_accuracy}')
        self.tb_writer.add_scalar(f'{train_flag.name} loss', epoch_loss, epoch_index)
        return epoch_loss

    def train_model(self, train_flag: bool = True, max_epochs: int = 256, save=True):
        """
        Training procedure
        """
        self.to(DEVICE)
        best_weights = deepcopy(self.state_dict())
        timestamp = time.time()
        if train_flag:
            self.optimizer = self.optimizer(self.parameters(), lr=0.0005)
        else:
            if self.verbose:
                print(f'Running only one epoch of validation')
            max_epochs = 1
        best_validation_loss = torch.inf
        best_validation_epoch = 0
        avg_train_loss = 0
        for epoch in range(max_epochs):
            gc.collect()
            if self.verbose:
                print(f'Running epoch no. {epoch}')
            if train_flag:
                # Make sure gradient tracking is on, and do a pass over the data
                self.train()
                avg_train_loss = self.one_epoch(epoch, train_flag=DatasetEnum.TRAIN)

            # Evaluation
            self.eval()
            running_validation_loss = 0.0
            with torch.no_grad():
                avg_validation_loss = self.one_epoch(epoch,
                                                     train_flag=DatasetEnum.VALIDATE)

            if train_flag:
                # Log the running loss averaged per batch for both training and validation,
                self.tb_writer.add_scalars('Training vs. Validation Loss',
                                           {'Training': avg_train_loss,
                                            'Validation': avg_validation_loss},
                                           epoch)
                self.tb_writer.flush()
                # Tracks best performance, and save the model's state
                if avg_validation_loss < best_validation_loss:
                    best_validation_loss = avg_validation_loss
                    best_validation_epoch = epoch
                    best_weights = deepcopy(self.state_dict())
                    model_path = f'./weights/{self.model_name}_best_val.pth'

                    if not Path(model_path).parent.exists():
                        Path(model_path).parent.mkdir(exist_ok=True, parents=True)

                    torch.save(self.state_dict(), model_path)
        self.load_state_dict(best_weights)
        if save:
            model_path = f'./weights/{self.model_name}_{round(best_validation_loss, 5)}__{time.strftime("%y.%b.%d", time.gmtime())}.pth'
            torch.save(self.state_dict(), model_path)
        if self.verbose:
            time_elapsed = time.time() - timestamp
            print(
                f'Training complete in {time_elapsed // 60:.6f}m {time_elapsed % 60:.6f}s')
        return best_validation_loss, best_validation_epoch

    def test_model(self, test_loader):
        self.test_loader = test_loader
        avg_test_loss = self.one_epoch(epoch_index=0, train_flag=DatasetEnum.TEST)
        return avg_test_loss

    def predict(self, inputs):
        self.eval()
        inputs = inputs.to(DEVICE)
        predictions = self(inputs)
        return predictions
