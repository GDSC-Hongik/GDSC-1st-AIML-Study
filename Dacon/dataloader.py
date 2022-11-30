from torch.utils.data import DataLoader
from dataset import GDSCDataset
from sklearn.model_selection import train_test_split
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def return_dataloaders() :
    train_df, val_df, train_labels, val_labels = train_test_split(
                                                        train_df.drop(columns=['N_category']), 
                                                        train_df['N_category'], 
                                                        test_size=0.2, 
                                                        random_state=42
                                                    )

    train_transforms = A.Compose([
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_transforms = A.Compose([                               
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])


    train_dataset = GDSCDataset(medical_df=train_df, labels=train_labels.values, train_mode=True)
    train_loader = DataLoader(train_dataset, batch_size=1, transform=train_transforms)

    val_dataset = GDSCDataset(medical_df=val_df, labels=val_labels.values, train_mode=False)
    val_loader = DataLoader(val_dataset, batch_size=1, transform=test_transforms)

    return train_loader, val_loader


    # print(f'✅ # of Train Datas : {len(tr_dataset)}')
    # print(f'✅ # of Validation Datas : {len(val_dataset)}')
    # print(f'✅ # of Test Datas : {len(te_dataset)}')

    # return train_loader, val_loader, te_loader


