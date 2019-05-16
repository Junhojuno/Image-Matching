```
해당 논문에 있는 오타로 인해 다시 정리한다.
오타는 3절과 figure 3의 network 설명이 잘못 짝지어져있다.
3.1 = (a), 3.2 = (c), 3.3 = (b), 3.4 = (d)로 정정해야한다.
```

# Localizing and Orienting Street Views Using Overhead Imagery (2017)
: 인공위성의 사진을 통해 거리뷰 사진을 보고 위치를 추정한다.

### 0. Abstact
- 거리뷰 이미지(ground level)를 위에서 찍은 이미지(overhead)와 matching시켜 해당 사진의 위치와 방향을 정하는 게 핵심.
- 이를 위해 100만쌍의 이미지를 모았고, 4개의 Deep CNN 구조를 사용하였다.
- 그리고 새로운 loss function을 사용하였는데, 이를 통해 정확도를 향상시켰다.
- image matching이 어려운 이유는 극단적 시각차(거리뷰와 위성사진의 시각차)뿐만 아니라
- 거리뷰의 방위각(?)을 몰라서 두 이미지 pair를 대응시키기 힘들기 때문.
  - 방위각의 개념은 위키피디아에서 찾아보고,
  - 멀리서 해당 거리를 바라본건지, 바로 앞에서 본건지 거리뷰만 보고는 알 수 없다는 의미인거 같다.
- 해결책까지는 아니지만 rotation invariance를 학습에 사용하였다.
  - 위성뷰를 받아서 몇 도 회전시킬지 sampling을 하고
  - 그 다음 거리뷰와 위성뷰의 상대적 회전정도(?)를 neural network로 예측한다.(이게 핵심이겠네)

| Keywords | 읽는 초점 |
|:----------:|:---------:|
| image matching (cross domain matching) | 어떻게 Matching이 이루어지는가? |
| 4 Deep CNN architectures | 각각의 구조는 어떠한가? |
| 새로운 loss function (Distance Based Logistic) | DBL을 활용하여 loss를 어떻게 설정하였는가? |
| rotation invariance | 전처리 과정에서 어떻게 rotation이 이루어지며, relative rotation을 어떻게 구하는가? |

### 1. Introduction
- 앞서 말했던 시점의 차이와 더불어 조명/계절변화 등의 요인이 challenging points.
- figure 1에서와 같이, 한번 매칭이 끝나면 ranking이 정해지고, 위치 추정을 한다.
- siamese network는 저차원 feature map(거리뷰/위성뷰)을 학습하는데 쓰였고,
- 두 이미지의 feature map을 비교해 matching score를 결정한다.
- **여기서 갈래가 두가지로 나뉜다.**
  - matching task와 ranking task로 다른 딥러닝 접근방식이 들어간다.
  - 각각 DBL loss와 <rotation invariance + orientation regression> 사용
  - ranking task랑 orientation regression하고 연관성이 직관적으로 와닿지 않는데..?

  ##### 1.1 Ralated work
  1) Image geolocalization
  - 이걸 써서 거리뷰로 해당 위치를 찾는 것이다.
  - 관련된 이전 연구들......
  2) Deep Learning
  - 패스..!
  
### 2. Dataset of street view and overhead image pairs
- Google panorama image에 위치정보(geo-tag, depth)가 포함되어 있는 것 같다.

### 3. Cross-view matching and ranking with CNN
- 앞에서 task가 2개로 나뉜다고 했다.(matching과 ranking) 여기서는 matching에 초점을 두고 설명한다.
- matching task를 먼저 설명하면
  - 학습시에 matched 데이터가 (street, overhead) pair로 들어서 학습시킵니다.(positive examples)
  - 테스트시에 위에서 학습시킨 모델로 match or not 분류한다.
- figure 3를 보면, CNN이 2개의 카테고리로 구분되는데,(여기가 기존 논문에 오표기된 부분)
  - 좌측 1열 (a), (b) 구조는 classification이고, 우측 1열 (c), (d)는 features를 이용해 distance를 구조이다.

  ##### 3.1 Classification CNN for image matching : figure 3 (a)
  - match or not을 분류한다.
  - (street, overhead) input image를 합쳤다.(AlexNet의 input은 1개이기 때문이다.)

  ##### 3.2 Siamese-like CNN for learning image features : figure 3 (c)
  <p align="center">
  <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL_figure.PNG?raw=true' height=350 width=650>
  </p>
  
  - 제목과 같이 siamese-like이지 siamese network가 아니라는 점을 명심하자.
  - street과 overhead 각각은 input size가 다르다.
  - 각각 AlexNet을 통과하고 나온 f(A), f(B)으로 distance를 구해서 충분히 작으면 match
  - non-match인데 distance가 가까운 경우, 멀리보내는 정도가 크다.(화살표의 길이가 더 길다, 위 그림)
  - 참고로 convolution은 pre-trained Network (AlexNet). (learned network라는 걸 보니)

  ##### 3.3 Siamese-classification hybrid network : figure 3 (b)
  - 이 hybrid network는 이미지가 독립적인 convolution에서 처리된다는 점에서 siamese network와 유사하고
  - convolution에서 나온 두 feature representation을 합쳐서 matching prob을 구하는 점에서 classification과 유사.
  - 정리하면, street/overhead 각각 convolution 통과시키고 concat해서 fc를 통과시켜 match or not 판단
  
  ##### 3.4 Triplet network for learning image features
  - 3.3 network가 2개의 CNN을 사용하는데 반해, Triplet은 3개를 사용한다.
  - 즉, 이미지가 3개(A,B,C)가 들어가는데 (A,B)는 match pair이고 (A,C)는 non-match pair다. 
  - match pair의 거리는 최소화, non-match pair의 거리는 최대화 --> hinge loss(margin maximum)
  
  ##### 3.5 Learning image representation with distance-based logistic loss
  - 새로운 loss function, distance-based logistics layer를 제안.
  - `siamese network의 경우`
  <p align="center">
  <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL1.PNG?raw=true' height=90 width=200>
  </p>
  <p align="center">
  <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL2.PNG?raw=true' height=50 width=300>
  </p>
  
  - `triplet network의 경우`
  <p align="center">
  <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL3.PNG?raw=true' height=100 width=320>
  </p>
  <p align="center">
  <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL4.PNG?raw=true' height=60 width=350>
  </p>
  
  - `loss의 작동`
  <p align="center">
  <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL5.PNG?raw=true' height=350 width=650>
  </p>
  
  - **Implementation detail** : m=10, D=euclidean distance 사용, feature normalization사용하지 않음. 

### 4. Learning to perform rotation invariant matching
- street/overhead matching시에 orientation alignment가 중요하다.
  
  ##### 4.1 Partial rotation invariance by data augmentation
  `rotation invariance : 이미지를 회전시켜도 그 이미지를 분류해낼수 있는 것을 말함`
  - orientation alignment를 orientation regression으로 학습한다는 것 같다.
  - **Training with multiple rotation samples**
    - overhead를 여러 각도로 회전시켜 학습
    - partial RI는 일부 각도에서만 분류능력을 갖춤.
    - RI를 90도로 정하면 -45~45도 사이의 각도로 overhead를 여러번 회전시킨다.
    - 이러면 90도에 대해서만 Rotation Invariance한 상태가 되는 것이다.
    - 360도로 정하면 fully RI가 된다.
  - **Testing with multiple rotation samples/crops**
    - test에서는 정확한 orientation alignment를 모르기 때문에,
    - training에서 patial RI였다면, partial RI한 범위에서 sample하나뽑고 그외에서 여러개 뽑아야한다.
    - 예를들어 360도로 학습시켰다면, test에서 sample은 1개면 충분하다.
    - 180도라면? 180도에서 1개, -180도에서 1개
    - 90도라면? 16개...???(16개까지 더 뽑았더니 성능이 약간 올랐다고함..)
  - **Multi-orientation feature averaging**
    - 16개의 samples들을 뽑았다면, 평균을 때려 하나의 sample처럼 만들자.
    
