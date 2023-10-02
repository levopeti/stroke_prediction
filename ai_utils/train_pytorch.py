import torch
import torch.nn as nn
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn import Module
from torch.nn.functional import sigmoid, one_hot
from torch.optim import Optimizer
from tqdm import tqdm

from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df
from measurement_utils.measure_db import MeasureDB



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

        self.to_tensor = ToTensor()

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
            print(model_out.shape, y.shape)
            loss.append(criterion(model_out, y).item())

            _, predicted = model_out.max(1)
            correct += predicted.eq(y).sum().item()
            acc = correct / total

        print("loss: {:.2f}, acc: {:.1f}%".format(sum(loss) / len(loss), acc * 100))

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
            print("\nepoch: {}".format(epoch))
            self.model.train()

            correct = 0
            total = 0

            #tq = tqdm(total=(len(self.train_loader)))
            #tq.set_description('ep {}'.format(epoch))
            for batch_idx, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                model_out = torch.squeeze(self.model(x.float()))

                print(model_out.shape, y.shape)
                loss = self.criterion(model_out, y)  # index of the max log-probability
                loss.backward()

                self.optimizer.step()

                total += y.size(0)

                _, predicted = model_out.max(1)
                correct += predicted.eq(y).sum().item()

                acc = correct / total

                #tq.update(1)
                #tq.set_postfix(loss='{:.2f}'.format(loss.item()),
                #               acc='{:.1f}%'.format(acc * 100))
                print("{:.2f}%, loss: {:.2f}, acc: {:.1f}%".format(batch_idx/len(self.train_loader), loss.item(), acc * 100))

            validation_step(self.model, self.criterion, self.valid_loader)

if __name__ == "__main__":
    db_path = "../data/WUS-v4measure202307311.accdb"
    ucanaccess_path = "../ucanaccess/"
    folder_path = "../data/clear_data/"
    clear_json_path = "../data/clear_train_test_ids.json"

    length = int(1.5 * 60 * 60 * 25)  # 1.5 hours, 25 Hz
    sample_per_meas = 1

    measDB = MeasureDB(db_path, ucanaccess_path)
    clear_measurements = ClearMeasurements(folder_path, clear_json_path)

    train_dataset = ClearDataset("train", clear_measurements, measDB, length, sample_per_meas)
    test_dataset = ClearDataset("test", clear_measurements, measDB, length, sample_per_meas)

    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=1)

    inpit_size = 12
    layer_sizes = [512, 128]
    output_size = 6

    lr = 0.001
    wd = 0

    num_epoch = 10

    model = nn.Sequential(
            nn.Linear(inpit_size, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], output_size)
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd,
                                 amsgrad=True)
    model = model.float()
    train_loop = TrainLoop(model,
                           optimizer,
                           criterion,
                           train_dataloader,
                           test_dataloader,
                           num_epoch)

    train_loop.run_loop()