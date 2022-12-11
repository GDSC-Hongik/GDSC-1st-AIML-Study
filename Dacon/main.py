from dataset import GDSCDataset
from dataloader import return_dataloaders
from model import InceptionV3
from trainer import Train

from torchsummary import summary

import pandas as pd
import torch
import torch.nn as nn


train_df = pd.read_csv('/content/train.csv')

train_loader, val_loader = return_dataloaders(df=train_df, ver='3')

model = InceptionV3()

# train configs
NUM_EPOCH = 10
CRITERION = nn.CrossEntropyLoss()
LR = 1e-7
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)


print('Model Architecture 🚩')
print(model)

trainer = Train(model=model, 
                num_epoch=NUM_EPOCH,
                optimizer=OPTIMIZER,
                criterion=CRITERION,
                tr_loader=train_loader,
                val_loader=val_loader,
                )

trainer.training()