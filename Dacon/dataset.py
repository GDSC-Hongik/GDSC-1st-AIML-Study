from torch.utils.data import Dataset
from utils import find_bbox, crop_rect, show_crops, get_tiles, show_tiles

class GDSCDataset(Dataset) :
    def __init__(self, medical_df, labels, transforms=None):
        self.medical_df = medical_df
        self.transforms = transforms
        self.labels = labels

    def __getitem__(self, idx) :
        img_path = self.medical_df['img_path'].iloc[idx]
        cropped_imgs = find_bbox(img_path=img_path)
        
        # TODO
        for crop in cropped_imgs :
            tile_lst = get_tiles(img=crop, tile_size=(150, 150), offset=(30, 30))
        

        return 

    def __len__(self) :
        return len(self.medical_df)


