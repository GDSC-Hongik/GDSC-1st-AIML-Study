from torch.utils.data import DataLoader
from dataset import GDSCDataset, GDSCDatasetV2, GDSCDatasetV3
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from prep import Prep

def return_dataloaders(tr_ori_df : pd.DataFrame, test_df: pd.DataFrame, ver='2') -> torch.utils.data.DataLoader :


    train_df, val_df, train_labels, val_labels = train_test_split(
                                                        tr_ori_df.drop(columns=['N_category']), 
                                                        tr_ori_df['N_category'], 
                                                        test_size=0.2, 
                                                        random_state=42
                                                    )

    ## Scaling
    preper = Prep(train_df=train_df, val_df=val_df, test_df=test_df)
    scaled_tr, scaled_val, scaled_test = preper.run()

    if ver == '1' :
        train_dataset = GDSCDataset(medical_df=scaled_tr, labels=train_labels.values, train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=1)

        val_dataset = GDSCDataset(medical_df=scaled_val, labels=val_labels.values, train_mode=False)
        val_loader = DataLoader(val_dataset, batch_size=1)

    elif ver == '2' :
        train_dataset = GDSCDatasetV2(medical_df=scaled_tr, labels=train_labels.values, train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=16)

        val_dataset = GDSCDatasetV2(medical_df=scaled_val, labels=val_labels.values, train_mode=False)
        val_loader = DataLoader(val_dataset, batch_size=16)

    elif ver =='3' :
        train_dataset = GDSCDatasetV3(medical_df=scaled_tr, labels=train_labels.values, train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = GDSCDatasetV3(medical_df=scaled_val, labels=val_labels.values, train_mode=False)
        val_loader = DataLoader(val_dataset, batch_size=32)

        test_dataset = GDSCDatasetV3(medical_df=scaled_test, labels=None, train_mode=False)
        test_loader = DataLoader(test_dataset, batch_size=32)

    print(f'✅ # of Train Datas : {len(train_dataset)}')
    print(f'✅ # of Validation Datas : {len(val_dataset)}')

    return train_loader, val_loader, test_loader