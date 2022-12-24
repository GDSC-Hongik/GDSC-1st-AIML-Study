from dataset import GDSCDataset
from dataloader import return_dataloaders
from model import InceptionV3, MyInceptionV3, ModalClassifier
from trainer import Train

from torchsummary import summary

import pandas as pd
import torch
import torch.nn as nn

from inference import *

train_df = pd.read_csv('/content/GDSC-1st-AIML-Study/Dacon/train.csv')
test_df = pd.read_csv('/content/GDSC-1st-AIML-Study/Dacon/test.csv')

train_loader, val_loader, test_loader = return_dataloaders(tr_ori_df=train_df, test_df=test_df, ver='3')

model = ModalClassifier()

# train configs
NUM_EPOCH = 10
CRITERION = nn.CrossEntropyLoss()
LR = 1e-2
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)
SCHEDULRER = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(OPTIMIZER, T_0=2)


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

print("✔Training is Done!!✔")
print("✨Start Evaluation✨")
model.load_state_dict(torch.load('./BEST_MODEL.pt'))
preds = inference(model=model, test_loader=test_loader, device=trainer.device)
make_submission(preds=preds, path='./SUBMISSION.csv')
print("Done!")
