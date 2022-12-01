from dataset import GDSCDataset
from dataloader import return_dataloaders
import pandas as pd


train_df = pd.read_csv('/content/train.csv')

train_loader, val_loader = return_dataloaders(df=train_df)

