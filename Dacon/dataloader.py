from torch.utils.data import DataLoader
from dataset import GDSCDataset, GDSCDatasetV2
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def return_dataloaders(df : pd.DataFrame, V2=False) -> torch.utils.data.DataLoader :

    train_df, val_df, train_labels, val_labels = train_test_split(
                                                        df.drop(columns=['N_category']), 
                                                        df['N_category'], 
                                                        test_size=0.2, 
                                                        random_state=42
                                                    )

    if V2 == False :
        train_dataset = GDSCDataset(medical_df=train_df, labels=train_labels.values, train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=1)

        val_dataset = GDSCDataset(medical_df=val_df, labels=val_labels.values, train_mode=False)
        val_loader = DataLoader(val_dataset, batch_size=1)

    elif V2 == True :
        train_dataset = GDSCDatasetV2(medical_df=train_df, labels=train_labels.values, train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=16)

        val_dataset = GDSCDatasetV2(medical_df=val_df, labels=val_labels.values, train_mode=False)
        val_loader = DataLoader(val_dataset, batch_size=16)

    print(f'✅ # of Train Datas : {len(train_dataset)}')
    print(f'✅ # of Validation Datas : {len(val_dataset)}')

    return train_loader, val_loader