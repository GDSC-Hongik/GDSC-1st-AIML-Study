# 데이콘 유방암의 임파선 전이 예측 AI경진대회
GDSC Hongik 11월-12월 활동입니다.

<br>

## 🚩 개요
이미지처리에 대한 이해력을 높이기 위해 데이콘에서 개최하는 대회에 참가하였습니다.

<br>

[해당 대회](https://dacon.io/competitions/official/236011/overview/description)는 임파선 전이 여부를 판단하는 것을 목표로 하며, 유방암 병리 슬라이드 영상과 임상 항목을 조합하여 유방암의 임파선 전이 여부를 Binary하게 Classification 하는 것을 요구합니다.

<br>

WSI(Whole Slide Image)와 Tabular Data를 동시에 활용해야 하는 Multi Modal Task로 다양한 흥미를 가진 GDSC 멤버들의 특성에 적절하다고 판단하여 해당 대회를 선택했습니다.  

<br>

WSI 학습은 일반적인 Image Classification Task와 다른 전처리를 요구하기 때문에 이러한 과정을 파악하기 위한 논문 스터디가 선행되었습니다. 논문 리뷰에 관한 내용은 [이곳](https://www.notion.so/Paper-Reading-776c5c01bccc427bb774aad421463829)을 참고해주시기 바랍니다.

<br>

주어진 Tabular 데이터는 환자의 건강상태 정보와 같은 임상 항목에 관련한 정보였습니다. 그러나 실무의 특성상 주어진 데이터엔 결측치가 매우 많았지만 다행히 Class Imbalance는 과하지 않은 상태였습니다.  

<br>

Image와 Tabular Data를 동시에 고려해야 하는 Multi Modal Task 였기 때문에 2팀으로 나눠 진행하기로 했습니다. 멤버의 흥미와 경험을 바탕으로 팀을 구성하였고, 각 도메인에 집중하여 최선의 결과를 낸 다음 이 결과를 이후 Ensemble 하거나 Feature를 함께 고려하는 방안을 계획하였습니다.

<br>

<br>

## 🚀 학습 전략
### 📷 이미지

### Preprocessing
기본적으로 주어진 이미지의 크기가 매우 컸기 때문에 찾아봤던 논문에 기반하여 이미지를 `patch`로 나눠 학습 시키는 전략을 구상했습니다. 해당 코드는 `utils.py`를 참고하시기 바랍니다.  
patch로 쪼갠 이미지를 학습하는데에 있어 3가지 전략을 구상했습니다. 이에 따라 `dataset.py`의 version이 3가지로 나뉘게 됩니다.

<br>

- **Ver 1** : 먼저 세포를 `cv2`기반으로 탐지하고 탐지된 세포의 bbox 안에서 patch를 나누는 전략입니다.
    - 기본적으로 MIL의 전략을 따라가려고 했으나, Supervised Learning이었던 기본적인 MIL방식에 비해 주어진 데이터셋은 Weakly Supervised Learning 에 적합한 데이터셋이었기 때문에 Instance 하나하나 Label이 있는 예시를 참고하기 어려웠습니다. 
    - 이에 나누어진 patch들을 하나의 bag 안에 넣고 이를 하나의 label과 매칭시키는 전략을 수립했습니다. 

<br>

- **Ver 2** : 세포를 탐지하고 그 안에서 patch를 나누는 방식은 동일하나 그렇게 모여진 patch를 하나의 Grid Image로 만드는 과정으로 시도해봤습니다. 
    - Weakly Supervised Learning 이었기 때문에 나누어진 patch를 하나의 이미지로 합치는 건 어떨까 하는 Idea 였습니다. 
    - 나누어진 patch 중, 빈 공간이 비교적 적은 patch 9개를 선별하여 이들을 3x3 Grid로 만드는 방식입니다. 이 방식을 좀 더 발전시켜 ver3가 됩니다.

<br>

- **Ver 3** : 세포를 탐지하고 patch를 선별하는 과정은 아래과 같은 단점이 존재 했습니다.
> 1. 기본적으로 탐지된 영역 안에서 patch를 나누게 되는데, 탐지된 영역이 양성과 음성을 가르는 핵심 지역이 아닐 가능성이 존재한다.
>
> 2. 탐지된 영역이 세포 전체를 반영하지 않는 경우가 많았다. 이에 이미지에 있는 세포 전체의 정보를 반영하기 까다로웠다. 
> 3. 위의 과정을 적절히 만족하는 threshold와 contour를 찾아야 했는데 그 과정이 사실 매우 오랜 시간을 요구하는 일이었다.  
- 이에 새로운 방법을 구상하였는데, 일단 전체 이미지를 patch로 나눈 다음 patch의 평균 **pixel value**를 활용하여 빈 공간이 적은 patch만을 선별하는 과정으로 수정했습니다.  
- 그렇게 얻은 patch를 Grid로 합쳐 이들을 모델 내부에서 `AdaptivePooling`으로 원하는 size로 바꿔주는 과정으로 대체했습니다.
    - 해당 방법은 임시로 적용한 것이며 개선 가능성이 있는 파트입니다. 
    - 요지는 __어떻게 Grid의 정보를 효율적으로 모델에 전달할지__ 에 대한 문제이므로 매우 다양한 방법을 적용할 수 있을 것같습니다.
        - ex) YOLO의 Grid 별 Loss Function 방법 적용 etc
    
### Modeling
기본적으로 EfficientNet 계열을 이용합니다. 사실 모델은 그냥 바꿔끼기만 하면 되는 수준이라 원하는 모델로 바꿔끼셔도 됩니다. 한 epoch당 10분 언저리로 걸리므로 더 무겁고 큰 모델을 이용해도 무난할 것이라고 예상됩니다.
코드 상엔 efficientnet_b0를 이용합니다. 
다만 모델을 바꾸실 때 `forward` 부분의 `AdaptivePooling`의 파라미터를 모델 resolution에 맞게 조정해주시면 됩니다.  

<br>

### 📃 Tabular
Baseline의 전처리를 똑같이 적용했습니다. 사실상 아무런 처리를 하지 않은 수준이라 이 부분도 충분한 개선 여지가 있을 것으로 기대됩니다.

<br>

<br>

## 👨‍💻 파일 설명
1. `prep.py` : Tabular Data 전처리를 위한 부분입니다.
2. `utils.py` : Image Data 전처리를 위한 코드입니다. 아직 더러운 상태입니다.. (사용 하지 않는 코드 정리를 안 했습니다 ㅎㅎ..)
3. `trainer.py` : 학습 과정을 담은 코드입니다. 
4. `inference.py` : Test Data를 Inference할 코드입니다.
5. `dataloader.py` : loader를 반환하는 함수입니다.
6. `dataset.py` : Dataset을 만들어주는 class 입니다. 3가지 버전으로 나누어져 있으며 저는 3을 이용했습니다.
7. `model.py` : 사용대상이 됐던 모델들이 모여있는 파일입니다. 
8. `main.py` : 해당 파일만 실행 시키면 전처리부터 학습까지 가능하게 만들었습니다.

<br>

<br>

## 👀 실행 예시
**Colab에서 실행하고 Drive 안에 Data의 Zip 파일이 있다고 가정합니다.**  
Drive Mount
```python
from google.colab import drive
drive.mount('/content/drive')
```
Code Clone
```
!git clone https://github.com/GDSC-Hongik/GDSC-1st-AIML-Study.git
```
Get Data Files 
```
%cd /content/GDSC-1st-AIML-Study/Dacon/
!unzip -qq "{zip파일 경로를 입력하세요. ({}이 표시는 빼고..)}"
```
Training & Inference
```
!python main.py
```
_**혹시 파라미터나 모델을 수정하고 싶으시면 content에 압축 해제된 코드들을 수정하고 저장하시면 수정 내역이 바로 반영됩니다. 다만 해당 코드들을 런타임이 끝나면 삭제되니 어떻게 수정했는지 잘 기억해두시거나 바로 원본 코드에 반영해주시면 됩니다.**_


<br>

<br>

## ✨ 개선점
1. Patch Image 정보를 모델에 전달하는 방법 찾기
2. Tabular Data 전처리 신경 써보기
3. `main.py`에서 `argparser`로 파라미터 조정 쉽게 하기
4. `config.json` 관리
5. `InceptionV3`와 같은 다양한 모델 사용해보기
