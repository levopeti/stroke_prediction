import torch
import os

import pytorch_lightning as pl
from datetime import datetime
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy
from functools import partial

from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df, save_params
from ai_utils.training_utils.pytorch_utils.lit_model import LitMLP
from ai_utils.training_utils.pytorch_utils.loss_and_accuracy import StrokeXELoss, StrokeMSELoss, StrokeClasAccuracy, StrokeRegAccuracy, RegAccuracy
from measurement_utils.measure_db import MeasureDB


# torch.multiprocessing.set_start_method('spawn')


class ClearDataset(Dataset):
    def __init__(self,
                 data_type: str,  # train or test
                 clear_measurements: ClearMeasurements,
                 measDB: MeasureDB,
                 length: int,
                 sample_per_meas: int) -> None:
        self.meas_id_list = clear_measurements.get_meas_id_list(data_type)
        self.clear_measurements = clear_measurements
        self.measDB = measDB
        self.sample_per_meas = sample_per_meas
        self.length = length

        # self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.meas_id_list) * self.sample_per_meas

    def __getitem__(self, idx):
        # print(idx)
        meas_idx = idx // self.sample_per_meas
        meas_id = self.meas_id_list[meas_idx]
        # start = time.time()
        # print(111)
        meas_df = self.clear_measurements.get_measurement(meas_id)
        # print("get meas {} took {:.2}s".format(meas_id, time.time() - start))

        class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
        # start = time.time()
        input_array = get_input_from_df(meas_df, self.length, class_value_dict)
        # print("get input_array for {} took {:.2}s".format(meas_id, time.time() - start))
        input_tensor = torch.from_numpy(input_array).float()
        # input_tensor = self.to_tensor(input_array)
        # input_tensor = input_array

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
              "test_sample_per_meas": 100,
              "train_batch_size": 100,
              "test_batch_size": 100,
              "input_shape": 12,
              "output_shape": 6,
              "layer_sizes": [1024, 512, 256],
              "patience": 20,
              "learning_rate": 0.001,
              "wd": 0,
              "num_epoch": 1000,
              "steps_per_epoch": 10,  # 100
              "stroke_loss_factor": 0.1,
              "cache_size": 1,
              "num_workers": 0
              }

    pprint(params)
    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, params["folder_path"], params["clear_json_path"],
                                           cache_size=params["cache_size"])
    clear_measurements.print_stat()

    params["train_id_list"] = clear_measurements.get_meas_id_list("train")
    params["test_id_list"] = clear_measurements.get_meas_id_list("test")
    save_params(params)

    train_dataset = ClearDataset("train", clear_measurements, measDB, params["length"], params["train_sample_per_meas"])
    val_dataset = ClearDataset("test", clear_measurements, measDB, params["length"], params["test_sample_per_meas"])

    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"], shuffle=False, num_workers=params["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=params["test_batch_size"], shuffle=False, num_workers=params["num_workers"])

    # input_size = 12
    # layer_sizes = [512, 128]
    # output_size = 6
    #
    # model = nn.Sequential(
    #     nn.Linear(input_size, layer_sizes[0]),
    #     nn.ReLU(),
    #     nn.Linear(layer_sizes[0], layer_sizes[1]),
    #     nn.ReLU(),
    #     nn.Linear(layer_sizes[1], output_size)
    # )
    #
    # loss = torch.nn.CrossEntropyLoss()

    optimizer = partial(torch.optim.Adam,
                        lr=params["learning_rate"],
                        weight_decay=params["wd"],
                        amsgrad=True)
    # model = model.float()
    # train_loop = TrainLoop(model,
    #                        optimizer,
    #                        loss,
    #                        train_dataloader,
    #                        test_dataloader,
    #                        params["num_epoch"])
    #
    # train_loop.run_loop()

    acc = Accuracy(task="multiclass", num_classes=6)
    acc.name = "acc"
    model = LitMLP(input_shape=params["input_shape"],
                   output_shape=params["output_shape"],
                   layer_sizes=params["layer_sizes"],
                   loss=StrokeXELoss(params["stroke_loss_factor"]) if params["output_shape"] == 6 else StrokeMSELoss(params["stroke_loss_factor"]),
                   accuracy_list=[acc, StrokeClasAccuracy()] if params["output_shape"] == 6 else [RegAccuracy(), StrokeRegAccuracy()],
                   optimizer=optimizer,
                   )

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    trainer = pl.Trainer(accelerator="cpu",
                         devices=1)
    trainer.fit(model, train_loader, val_loader)

    # early_stop_callback=True,

    # metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

    # from pytorch_lightning.loggers import TensorBoardLogger
    # logger = TensorBoardLogger('tb_logs', name='my_model')
    # trainer = Trainer(logger=logger)


if __name__ == "__main__":
    train()
