import torch
import torch.nn as nn
import time

import numpy as np
from datetime import datetime
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn import Module
from torch.nn.functional import sigmoid, one_hot
from torch.optim import Optimizer
from tqdm import tqdm

from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df, save_params
from measurement_utils.measure_db import MeasureDB

# torch.multiprocessing.set_start_method('spawn')



class ClearDataset(Dataset):
    def __init__(self,
                 data_type: str, # train or test
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
        #print(idx)
        meas_idx = idx // self.sample_per_meas
        meas_id = self.meas_id_list[meas_idx]
        #start = time.time()
        #print(111)
        meas_df = self.clear_measurements.get_measurement(meas_id)
        #print("get meas {} took {:.2}s".format(meas_id, time.time() - start))

        class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
        start = time.time()
        input_array = get_input_from_df(meas_df, self.length, class_value_dict)
        #print("get input_array for {} took {:.2}s".format(meas_id, time.time() - start))
        input_tensor = torch.from_numpy(input_array).float()
        # input_tensor = self.to_tensor(input_array)
        # input_tensor = input_array

        label = min(class_value_dict.values())
        return input_tensor, label

def validation_step(model: Module,
                    criterion: Module,
                    valid_loader: DataLoader):

    with torch.no_grad():
        model.eval()
        correct = 0
        loss = list()

        total = 0

        for batch_idx, (x, y) in enumerate(valid_loader):
            total += y.size(0)

            model_out = torch.squeeze(model(x.float()))
            # print(model_out.shape, y.shape)
            loss.append(criterion(model_out, y).item())

            _, predicted = model_out.max(1)
            correct += predicted.eq(y).sum().item()
            acc = correct / total

        print("val_acc: {:.1f}%, val_loss: {:.2f}".format(acc * 100, sum(loss) / len(loss)))

class TrainLoop(object):
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 criterion: Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 num_epoch: int):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epoch = num_epoch

    def run_loop(self):
        for epoch in range(self.num_epoch):
            # print("\nepoch: {}".format(epoch))
            self.model.train()

            correct = 0
            total = 0

            tq = tqdm(total=(len(self.train_loader)))
            tq.set_description('epoch {}'.format(epoch))
            for batch_idx, (x, y) in enumerate(self.train_loader):
                # print(batch_idx, x, y)
                self.optimizer.zero_grad()

                model_out = torch.squeeze(self.model(x.float()))

                # print(model_out.shape, y.shape)
                loss = self.criterion(model_out, y)  # index of the max log-probability
                loss.backward()

                self.optimizer.step()

                total += y.size(0)

                _, predicted = model_out.max(1)
                correct += predicted.eq(y).sum().item()

                acc = correct / total

                tq.update(1)
                tq.set_postfix(acc='{:.1f}%'.format(acc * 100),
                               loss='{:.2f}'.format(loss.item()))
                # print("{:.2f}%, loss: {:.2f}, acc: {:.1f}%".format(batch_idx/len(self.train_loader), loss.item(), acc * 100))

            tq.close()
            validation_step(self.model, self.criterion, self.valid_loader)

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
              "output_shape": 1,
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
    test_dataset = ClearDataset("test", clear_measurements, measDB, params["length"], params["test_sample_per_meas"])

    train_dataloader = DataLoader(train_dataset, batch_size=params["train_batch_size"], shuffle=False, num_workers=params["num_workers"])
    test_dataloader = DataLoader(test_dataset, batch_size=params["test_batch_size"], shuffle=False, num_workers=params["num_workers"])

    input_size = 12
    layer_sizes = [512, 128]
    output_size = 6

    model = nn.Sequential(
        nn.Linear(input_size, layer_sizes[0]),
        nn.ReLU(),
        nn.Linear(layer_sizes[0], layer_sizes[1]),
        nn.ReLU(),
        nn.Linear(layer_sizes[1], output_size)
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params["learning_rate"],
                                 weight_decay=params["wd"],
                                 amsgrad=True)
    model = model.float()
    train_loop = TrainLoop(model,
                           optimizer,
                           criterion,
                           train_dataloader,
                           test_dataloader,
                           params["num_epoch"])

    train_loop.run_loop()

if __name__ == "__main__":
    train()

