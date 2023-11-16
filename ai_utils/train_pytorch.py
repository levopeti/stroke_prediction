import torch
import os
from time import time

import pytorch_lightning as pl
from datetime import datetime
from pprint import pprint

from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy
from functools import partial

from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df, save_params
from ai_utils.training_utils.pytorch_utils.lit_model import LitMLP
from ai_utils.training_utils.pytorch_utils.loss_and_accuracy import MSELoss, StrokeLoss, StrokeAccuracy, Accuracy, OnlyFiveAccuracy
from measurement_utils.measure_db import MeasureDB


torch.multiprocessing.set_start_method('spawn', force=True)


class ClearDataset(Dataset):
    def __init__(self,
                 data_type: str,  # train or validation
                 clear_measurements: ClearMeasurements,
                 batch_size: int,
                 length: int,
                 sample_per_meas: int,
                 steps_per_epoch: int) -> None:
        self.data_type = data_type
        self.batch_size = batch_size
        self.meas_id_list = clear_measurements.get_meas_id_list(data_type)
        self.clear_measurements = clear_measurements
        self.sample_per_meas = sample_per_meas
        self.length = length
        self.steps_per_epoch = steps_per_epoch

        # self.to_tensor = ToTensor()

    def __len__(self):
        if self.data_type == "validation":
            return len(self.meas_id_list) * self.sample_per_meas
        elif self.data_type == "train":
            return self.steps_per_epoch * self.batch_size

    def __getitem__(self, idx):
        meas_idx = idx // self.sample_per_meas % len(self.meas_id_list)
        meas_id = self.meas_id_list[meas_idx]
        meas_df = self.clear_measurements.get_measurement(meas_id)

        class_value_dict = self.clear_measurements.get_class_value_dict(meas_id=meas_id)
        input_array = get_input_from_df(meas_df, self.length, class_value_dict)
        input_tensor = torch.from_numpy(input_array).float()

        label = min(class_value_dict.values())
        return input_tensor, label


def train():
    params = {"accdb_path": "./data/WUS-v4measure202307311.accdb",
              "ucanaccess_path": "./ucanaccess/",
              "folder_path": "./data/clear_data/",
              "clear_json_path": "./data/clear_train_test_ids.json",
              "model_base_path": "./models/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M')),
              "length": int(1.5 * 60 * 60 * 25),  # 1.5 hours, 25 Hz
              "train_sample_per_meas": 10,
              "val_sample_per_meas": 100,
              "train_batch_size": 100,  # 100
              "val_batch_size": 100,
              "input_shape": 12,
              "output_shape": 6,
              "layer_sizes": [1024, 512, 256],
              "patience": 20,
              "learning_rate": 0.001,
              "wd": 0,
              "num_epoch": 1000,
              "steps_per_epoch": 100,  # 100
              "stroke_loss_factor": 1,
              "cache_size": 1,
              "num_workers": 10
              }

    pprint(params)
    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, params["folder_path"], params["clear_json_path"],
                                           cache_size=params["cache_size"])
    clear_measurements.print_stat()

    params["train_id_list"] = clear_measurements.get_meas_id_list("train")
    params["val_id_list"] = clear_measurements.get_meas_id_list("validation")
    save_params(params)

    train_dataset = ClearDataset("train",
                                 clear_measurements,
                                 params["train_batch_size"],
                                 params["length"],
                                 params["train_sample_per_meas"],
                                 params["steps_per_epoch"])
    val_dataset = ClearDataset("validation",
                               clear_measurements,
                               params["val_batch_size"],
                               params["length"],
                               params["val_sample_per_meas"],
                               params["steps_per_epoch"])

    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"], shuffle=False, num_workers=params["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=params["val_batch_size"], shuffle=False, num_workers=params["num_workers"])

    optimizer = partial(torch.optim.Adam,
                        lr=params["learning_rate"],
                        weight_decay=params["wd"],
                        amsgrad=True)

    if params["output_shape"] == 6:
        # classification problem
        xe = CrossEntropyLoss()
        xe.name = "xe_loss"
        loss_list = [xe, StrokeLoss(params["stroke_loss_factor"])]
    else:
        # regression problem
        assert params["output_shape"] == 1, params["output_shape"]
        loss_list = [MSELoss()]  # , StrokeLoss(params["stroke_loss_factor"])]

    accuracy_list = [Accuracy(), StrokeAccuracy(), OnlyFiveAccuracy()]

    model = LitMLP(input_shape=params["input_shape"],
                   output_shape=params["output_shape"],
                   layer_sizes=params["layer_sizes"],
                   loss_list=loss_list,
                   accuracy_list=accuracy_list,
                   optimizer=optimizer,
                   )

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    trainer = pl.Trainer(accelerator="cpu",
                         devices=1)
    trainer.fit(model, train_loader, val_loader)

    # TODO
    # early_stop_callback=True,

    # metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

    # from pytorch_lightning.loggers import TensorBoardLogger
    # logger = TensorBoardLogger('tb_logs', name='my_model')
    # trainer = Trainer(logger=logger)

    # vanishing gradient
    # https://discuss.pytorch.org/t/how-to-check-for-vanishing-exploding-gradients/9019/13
    # https://machinelearningmastery.com/visualizing-the-vanishing-gradient-problem/
    # for name, param in model.named_parameters():
    #     print(name, param.grad.norm())


if __name__ == "__main__":
    train()
