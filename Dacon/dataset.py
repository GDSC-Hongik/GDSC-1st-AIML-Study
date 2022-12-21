from torch.utils.data import Dataset, DataLoader
from utils import *
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from prep import Prep

import torch


class GDSCDataset(Dataset) :
    '''
    Attention-Based MIL 논문의 MNIST 예제를 참고하여 만들었습니다.
    참고 링크
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/dataloader.py

    1. split이 되어 있다는 가정
    2. train이든 test든 일단 똑같이 진행 하다가 crop을 하고 나서 train인 경우 augmentation 진행
    3. train - 각자 다르게 aug된 crop들의 리스트를 반환함 
    4. train은 aug된 crop을 기반으로 tiling을 해서 bag 하나에 집어 넣음
    5. bag은 tile들이 엄청 많이 담긴 하나의 가방이 됨
        5-1. tile들이 엄청 많이 담겼으니 이걸.. 다 쓰면 또 안 될 거 같은데... 고민좀 해봐야할듯
    6. 이 bag과 label 하나를 매칭시킬 예정
    '''

    def __init__(self, medical_df : pd.DataFrame, labels : np.array, train_mode=True):
        self.medical_df = medical_df
        self.labels = labels
        self.train_mode = train_mode
        self.totensor = A.Compose([
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                    max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])


    def __getitem__(self, idx) :
        img_path = self.medical_df['img_path'].iloc[idx]
        label = self.labels[idx]
        cropped_imgs = find_bbox(img_path=img_path)
        
        ## Augmentation
        if self.train_mode :
            cropped_imgs = train_aug(crop_lst=cropped_imgs)
        
        # else :
        #     cropped_imgs = test_aug(crop_lst=cropped_imgs)

        # 가방 하나는 여러 개의 tiles의 list로 이루어져 있습니다
        bag = []
        for crop in cropped_imgs :
            for tile in get_tiles(img=crop, tile_size=(150, 150), offset=(30, 30)) :
                bag.append(self.totensor(image=tile)['image'])
        
        print('✅ # of Tiles.. : ', len(bag))
        

        return bag, label

    def __len__(self) :
        return len(self.medical_df)


class GDSCDatasetV2(Dataset) :
    def __init__(self, medical_df : pd.DataFrame, labels : np.array, train_mode=True):
        self.medical_df = medical_df
        self.labels = labels
        self.train_mode = train_mode
        self.train_augs = A.Compose([
                        A.HorizontalFlip(),
                        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                    max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])
        self.test_augs = A.Compose([
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                    max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])
        

    def __getitem__(self, idx) :
        img_path = self.medical_df['img_path'].iloc[idx]
        label = self.labels[idx]
        cropped_imgs = find_bbox(img_path=img_path, bbox_h=50, bbox_w=50)

        tile_lst = []
        for crop in cropped_imgs :
                    for tile in get_tiles(img=crop, tile_size=(100, 100), offset=(15, 15)) :
                        tile_lst.append(tile)

        fin_img = concat_crops(tile_lst)

        if self.train_mode :
            fin_img = self.train_augs(image=fin_img)['image']

        else :
            fin_img = self.test_augs(image=fin_img)['image']

        return fin_img, label

    def __len__(self) :
        return len(self.medical_df)


class GDSCDatasetV3(Dataset):
    def __init__(self, medical_df : pd.DataFrame, labels : np.array, train_mode=True):
        self.medical_df = medical_df
        self.labels = labels
        self.train_mode = train_mode
        self.train_augs = A.Compose([
                        A.HorizontalFlip(),
                        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                    max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])
        self.test_augs = A.Compose([
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                    max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])


    def __getitem__(self, idx):
        img_path = self.medical_df['img_path'].iloc[idx]
        label = self.labels[idx]
        tabular = Prep(self.medical_df)
        scaled_df = tabular.run()

        patch_lst = V3_patches(img_path=img_path, patch_size=(299, 299))
        patch_lst_2 = V3_patch_filter(patch_lst=patch_lst, mean_thresh=244)
        patch_lst_3 = V3_patch_selector(patch_lst=patch_lst_2, n=42)
        fin_img = V3_grid(patch_lst=patch_lst_3)

        if self.train_mode:
            aug_img = self.train_augs(image=fin_img)['image']
        else:

            aug_img = self.test_augs(image=fin_img)['image']


        
        return aug_img, scaled_df, label

    def __len__(self):
        return len(self.medical_df)


