import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, Module, ModuleList
from torch.optim import Optimizer
from functools import partial
from typing import List
from torchmetrics import Metric


class LitMLP(LightningModule):
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 layer_sizes: list,
                 loss: Module,
                 accuracy_list: List[Metric],
                 optimizer: partial):
        super().__init__()
        self.loss = loss
        self.accuracy_list = accuracy_list
        self.optimizer = optimizer

        assert len(layer_sizes) > 0, layer_sizes
        self.layers = ModuleList([Linear(input_shape, layer_sizes[0])])

        for i in range(len(layer_sizes) - 1):
            self.layers.append(ReLU())
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.layers.append(Linear(layer_sizes[-1], output_shape))

        # TODO: self.save_hyperparameters()


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        return x

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        log_dict = {"train_loss": loss}
        for accuracy in self.accuracy_list:
            accuracy.update(logits, y)
            log_dict["train_" + accuracy.name] = accuracy.compute()
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self):
        # log_dict = dict()
        for accuracy in self.accuracy_list:
            # log_dict["train_" + accuracy.name + "_end"] = accuracy.compute()
            accuracy.reset()
        # self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # log_dict = {"val_loss": loss}
        for accuracy in self.accuracy_list:
            accuracy.update(logits, y)
        #     log_dict["val_" + accuracy.name] = accuracy(logits, y)
        # self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        log_dict = dict()
        for accuracy in self.accuracy_list:
            log_dict["val_" + accuracy.name + "_end"] = accuracy.compute()
            accuracy.reset()
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

    def on_epoch_start(self):
        # in order to print a new pbar for every epoch
        print(666)
        print("\n")

