from torch.utils.data import DataLoader
from dataset import GDSCDataset
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def return_dataloaders(df : pd.DataFrame) -> torch.utils.data.DataLoader :
    
    train_df, val_df, train_labels, val_labels = train_test_split(
                                                        train_df.drop(columns=['N_category']), 
                                                        train_df['N_category'], 
                                                        test_size=0.2, 
                                                        random_state=42
                                                    )


    train_dataset = GDSCDataset(medical_df=train_df, labels=train_labels.values, train_mode=True)
    train_loader = DataLoader(train_dataset, batch_size=1)

    val_dataset = GDSCDataset(medical_df=val_df, labels=val_labels.values, train_mode=False)
    val_loader = DataLoader(val_dataset, batch_size=1)


    print(f'✅ # of Train Datas : {len(train_dataset)}')
    print(f'✅ # of Validation Datas : {len(val_dataset)}')


    return train_loader, val_loader