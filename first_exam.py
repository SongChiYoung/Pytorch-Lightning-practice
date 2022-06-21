import os

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import functional as FM

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import MNIST


import pytorch_lightning as pl

class MODEL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(28*28, 1024)
        self.hidden = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.out(x))

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        acc = FM.accuracy(pred, y)
        loss = F.cross_entropy(pred, y)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


batch_size = 2048

dataset = MNIST(
    os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
)
train_dataset, val_dataset = random_split(dataset, [55000, 5000])
test_dataset = MNIST(
    os.getcwd(), train=False, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


trainer = pl.Trainer(
	max_epochs=10
	,accelerator="gpu"
	,enable_progress_bar=True
	,progress_bar_refresh_rate=10
)
model = MODEL()
trainer.fit(model, train_loader, val_loader)

trainer.test(model, test_loader)
