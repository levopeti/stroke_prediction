import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torchmetrics import Metric

# accuracy
class StrokeMSELoss(Module):
    def __init__(self, stroke_loss_factor: float):
        super().__init__()
        self.stroke_loss_factor = stroke_loss_factor

    def forward(self, predictions: Tensor, targets: Tensor):
        predictions = torch.softmax(predictions, dim=1)
        # scale the predictions from [0, 1] to [-0.5, 5.4]
        # Values equidistant from two integers are rounded towards the
        # nearest even value (zero is treated as even) (-0.5 -> 0; 5.5 -> 6)
        predictions *= 5.9
        predictions -= 0.5
        predictions = torch.round(predictions)

        mse_loss = torch.mean((predictions - targets) ** 2)
        stroke_loss = torch.mean(torch.logical_xor(targets == 5, predictions == 5).float())
        loss = self.stroke_loss_factor * stroke_loss + mse_loss
        return loss


class StrokeXELoss(Module):
    def __init__(self, stroke_loss_factor: float):
        super().__init__()
        self.stroke_loss_factor = stroke_loss_factor
        self.xe_loss = CrossEntropyLoss()

    def forward(self, predictions: Tensor, targets: Tensor):
        # pytorch xe loss contains softmax
        ce_loss = self.xe_loss(predictions, targets)
        # softmax is not necessary before argmax (result would be the same)
        predictions = torch.argmax(predictions, dim=1)
        stroke_loss = torch.mean(torch.logical_xor(targets == 5, predictions == 5).float())
        loss = self.stroke_loss_factor * stroke_loss + ce_loss
        return loss

# loss
class RegAccuracy(Metric):
    higher_is_better = True
    name = "acc"
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor):
        predictions = torch.softmax(predictions, dim=1)
        # scale the predictions from [0, 1] to [-0.5, 5.4]
        # Values equidistant from two integers are rounded towards the
        # nearest even value (zero is treated as even) (-0.5 -> 0; 5.5 -> 6)
        predictions *= 5.9
        predictions -= 0.5
        predictions = torch.round(predictions)
        assert predictions.shape == targets.shape

        self.correct += torch.sum(predictions == targets)
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class StrokeRegAccuracy(Metric):
    higher_is_better = True
    name = "stroke_acc"
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor):
        predictions = torch.softmax(predictions, dim=1)
        # scale the predictions from [0, 1] to [-0.5, 5.4]
        # Values equidistant from two integers are rounded towards the
        # nearest even value (zero is treated as even) (-0.5 -> 0; 5.5 -> 6)
        predictions *= 5.9
        predictions -= 0.5
        predictions = torch.round(predictions)
        assert predictions.shape == targets.shape

        self.correct += torch.sum(torch.logical_not(torch.logical_xor(targets == 5, predictions == 5)))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class StrokeClasAccuracy(Metric):
    higher_is_better = True
    name = "stroke_acc"
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor):
        # softmax is not necessary before argmax (result would be the same)
        predictions = torch.argmax(predictions, dim=1)
        assert predictions.shape == targets.shape

        self.correct += torch.sum(torch.logical_not(torch.logical_xor(targets == 5, predictions == 5)))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total