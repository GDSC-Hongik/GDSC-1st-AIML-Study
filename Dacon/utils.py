import matplotlib.pyplot as plt
from glob import glob
import cv2
import numpy as np
import math
from google.colab.patches import cv2_imshow
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# TODO
'''
1. Tiling된 것들 중에서 유의미한 영역과 적당한 갯수를 찾아내는 방법 생각해내기
    -> 지금은 crop에서 tile들을 얻어내는데... 문제는 이러면 몇 가지 걱정거리? 가 발생함
    -> a. 첫번째 단계인 crop은 threshold 기반의 bbox 탐지로 얻어냄 -> 이 bbox가 유의미한 곳에 안 쳐진다면? (mask가 쳐진 곳을 탐지해야 하는데 그러지 못하는 경우)
    -> b. 클래스는 다행히 서로 비슷하긴 한데 그래도 뭐 어느 정도 tile끼리 밸런싱을 맞추면 좋겠음
2. 적절한 Augmetation 논문 보면서 재설정하기
'''


def show_crops(crop_lst : list, figsize=(50,50)) -> plt.figure :
    plt.figure(figsize=figsize)

    for idx, crop in enumerate(crop_lst) :
        plt.subplot(5, 10, idx+1)
        plt.imshow(crop)
    plt.show()

def find_bbox(img_path : str, 
                thresh_min=200, thresh_max=255, bbox_w=150, bbox_h=150) -> list(np.array) :
    '''
    한 개의 이미지에 대한 지정한 크기 만큼의 bbox를 찾아주는 함수입니다.

    input
        imgs_path : 이미지의 path 입니다.
                    별도의 config 파일을 만들어 관리할 예정입니다.
        thresh_min : contour를 찾기 위한 threshold의 최소값입니다. 
        thresh_max : contour를 찾기 위한 threshold의 최대값입니다.
        bbox_w : 최소한으로 요구되는 bbox의 최소 너비입니다.
        bbox_h : 최소한으로 요구되는 bbox의 최소 높이입니다.
    
    output
        바운딩 박스의 정보와 crop된 이미지의 리스트를 반환합니다.
    '''    
    imgray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(img_path)
    dst = cv2.cvtColor(dst, cv2.COLOR_BAYER_BG2RGB)

    _, imthres = cv2.threshold(imgray, thresh_min, thresh_max, cv2.THRESH_BINARY_INV)

    contours, hier = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPOX_SIMPLE)

    try : hierarchy = hier[0]
    except : hierarchy = []

    cropped_imgs = []
    for contour, hier in zip(contours, hierarchy) :
        rect = cv2.minAreaRect(contour)

        x, y, w, h, =rect[0][0], rect[0][1], rect[1][0], rect[1][1]

        if w > bbox_w and h > bbox_h :
            img_cropped, _ = crop_rect(dst, rect)
            cropped_imgs.append(img_cropped)

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(dst, [box], -1, (0, 0, 255), 10) 

    # cv2.imshow(dst) local에서 돌릴 땐 이게 될 거고
    cv2_imshow(dst) # colav에서 돌릴 땐 이걸 쓰십쇼

    return cropped_imgs


def crop_rect(img : np.array, rect : np.array) -> np.array :
    '''
    contour에 기반한 bbox를 crop 해주는 함수입니다.

    input
        img : 어디서 잘라낼지를 말합니다.
        rect : cv2.minAreaRect로 얻어낸 좌표정보입니다.

    output
        crop 된 이미지와 돌리기 전 이미지를 반환합니다.
    
    ```````````````````````````````````
    해당 함수는 find_bbox 함수에 이용됩니다.
    ```````````````````````````````````
    '''
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[0], img.shape[1]
    
    M = cv2.getRotationsMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def get_tiles(img : np.array , tile_size : tuple, offset : tuple) -> list(np.array) :
    '''
    들어온 이미지에 대하여 tiling을 수행하는 함수입니다.
    위의 find_bbox 함수 이후에 수행합니다.

    input 
        img : crop된 이미지를 기준으로 합니다.
        tile_size : sliding하면서 자르는 것이기 때문에 그 slide의 크기를 지정합니다.
        offset : 몇 pixel씩 이동하면서 tiling할 것인지 지정합니다.
    
    output
        tile 들이 담긴 list를 반환합니다.

    ===============================================================================================================
    tile_size와 offset은 학습하는 모델과 이미지의 상태를 보고 결정합니다.

    몇 개 돌려본 결과 tile_size=(150,150) / offset=(30,30) 이면 적당히 직관적인 결과가 나왔습니다.
    ===============================================================================================================
    
    '''
    img_shape = img.shape

    tile_lst = []
    for i in range(int(math.ceil(img_shape[0] / offset[1] * 1.0))) :
        for j in range(int(math.ceil(img_shape[1] / offset[0] * 1.0))) :
            cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]),\
                                offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
            
            if cropped_img.shape[0] >= tile_size[0] and cropped_img.shape[1] >= tile_size[1] :
                tile_lst.append(cropped_img)

    return tile_lst


def show_tiles(tile_lst : list, figsize=(50,50)) -> plt.figure :
    plt.figure(figsize=figsize)

    for idx, tile in enumerate(tile_lst) :
        plt.subplot(5, 10, idx+1)
        plt.imshow(tile)
    plt.show()


def train_aug(crop_lst : list) -> list :
    # TODO 
    ## Augmentation 적절하게 다시 설정하기   Tiling을 위해 resized되거나 들쭉날쭉 해지는 건 없어야 함

    def brightness(gray, val):
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = int(random.uniform(-val, val))
        if brightness > 0:
            gray = gray + brightness
        else:
            gray = gray - brightness
        gray = np.clip(gray, 10, 255)
        return gray

    def contrast(gray, min_val, max_val):
        #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = int(random.uniform(min_val, max_val)) # Contrast control
        adjusted = cv2.convertScaleAbs(gray, alpha=alpha)
        return adjusted

    def fill(img, h, w):
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def rotation(img, angle):
        angle = int(random.uniform(-angle, angle))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img

    def vertical_shift_down(img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = h*ratio
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
        img = fill(img, h, w)
        return img

    def vertical_shift_up(img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return img
        ratio = random.uniform(0.0, ratio)
        h, w = img.shape[:2]
        to_shift = h*ratio
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
        img = fill(img, h, w)
        return img

    def horizontal_shift(img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img = img[:, :int(w-to_shift), :]
        if ratio < 0:
            img = img[:, int(-1*to_shift):, :]
        img = fill(img, h, w)
        return img

    def vertical_flip(img, flag):
        if flag:
            return cv2.flip(img, 0)
        else:
            return img

    def horizontal_flip(img, flag):
        if flag:
            return cv2.flip(img, 1)
        else:
            return img

    
    totensor = A.Compose([
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                    max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])

    train_aug_lst = []
    for crop in crop_lst :
        # img = cv2.resize(crop, dsize=(299, 299),interpolation=cv2.INTER_LINEAR)
        img = brightness(crop, 30)
        img = contrast(img, 1, 1.5)
        img = horizontal_flip(img, 1)
        img = rotation(img, 180)
        img = horizontal_shift(img, 0.1)
        #if random.uniform(0,1) > 0.5:
        #    img = vertical_flip(img, 1)
        
        img = totensor(image=img)['image']

        train_aug_lst.append(img)

    return train_aug_lst


def test_aug(crop_lst : list) -> list :
    
    totensor = A.Compose([
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                    max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])

    test_aug_lst = []
    for crop in crop_lst :
        img = totensor(image=crop)['image']

        test_aug_lst.append(img)
    return test_aug_lst

