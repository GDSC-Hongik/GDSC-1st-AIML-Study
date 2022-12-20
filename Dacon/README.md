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


## 👨‍💻 파일 설명