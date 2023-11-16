import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, Module, ModuleList
from functools import partial
from typing import List
from torchmetrics import Metric


class LitMLP(LightningModule):
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 layer_sizes: list,
                 loss_list: List[Module],
                 accuracy_list: List[Metric],
                 optimizer: partial):
        super().__init__()
        self.loss_list = loss_list
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
        loss_dict = {loss.name: loss(logits, y) for loss in self.loss_list}
        loss_sum = sum(loss_dict.values())

        log_dict = dict()
        log_dict["train_loss"] = loss_sum

        for name, loss in loss_dict.items():
            log_dict["train_" + name] = loss

        for accuracy in self.accuracy_list:
            accuracy.update(logits, y)
            log_dict["train_" + accuracy.name] = accuracy.compute()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return sum(loss_dict.values())

    def on_train_epoch_end(self):
        for accuracy in self.accuracy_list:
            accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        for accuracy in self.accuracy_list:
            accuracy.update(logits, y)

    def on_validation_epoch_end(self):
        log_dict = dict()
        for accuracy in self.accuracy_list:
            log_dict["val_" + accuracy.name] = accuracy.compute()
            accuracy.reset()
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        print("\n\n")


