from dataset import GDSCDataset
from dataloader import return_dataloaders
from model import InceptionV3, MyInceptionV3, ModalClassifier
from trainer import Train

from torchsummary import summary

import pandas as pd
import torch
import torch.nn as nn


train_df = pd.read_csv('/content/GDSC-1st-AIML-Study/Dacon/train.csv')

train_loader, val_loader = return_dataloaders(df=train_df, ver='3')

model = ModalClassifier()

# train configs
NUM_EPOCH = 10
CRITERION = nn.CrossEntropyLoss()
LR = 1e-2
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)
SCHEDULRER = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(OPTIMIZER)


print('Model Architecture 🚩')
print(model)

trainer = Train(model=model, 
                num_epoch=NUM_EPOCH,
                optimizer=OPTIMIZER,
                scheduler=SCHEDULRER,
                criterion=CRITERION,
                tr_loader=train_loader,
                val_loader=val_loader,
                )

trainer.training()