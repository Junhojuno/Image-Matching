# Localizing and Orienting Street Views Using Overhead Imagery (2017)
: 인공위성의 사진을 통해 거리뷰 사진을 보고 위치를 추정한다.

| Keywords | 읽는 초점 |
|:----------:|:---------:|
| image matching (cross domain matching) | Matching은 어떻게 시키는가? |
| 4 Deep CNN architectures | 기존 siamese/triplet과의 차이는 어떠한가? |
| Distance Based Logistic | DBL을 활용하여 loss를 어떻게 설정하였는가? |
| 2 Loss function | loss function 2개는 어떻게 이어지는가? |
| rotation invariant | 전처리 과정에서 어떻게 rotation이 이루어졌는가? |

### 0. Abstact
- 지상에서 찍은 사진의 위치와 방향(?)을 위성사진과 매칭시켜 결정해보는 것을 목표로 함(image matching task)
- 이와 관련하여 deep CNN architectures를 볼건데, 각각의 장단점이 있다.(Classification, Hybrid vs Siamese, Triplet)
- image matching task에 몇가지 어려움이 있다.(거리뷰와 위성사진의 극단적 시점차이, **거리뷰의 방위각을 모른다는점**)
- 중간에 rotation invariance라는 말이 나오는데,
  - **rotational invariance** if its value does not change when arbitrary rotations are applied to it
  - 여기서 말하는 value가 target(label)을 의미하는건가?
- 명확하게 방위각같은 orientation을 지정해주면 accuracy가 향상된다.(당연한 얘기)
- siamese network baseline보다 약 2.5배 향상된 아키텍쳐를 개발하였다고 함.

### 1. Introduction
- 시점변수(예를 들면 동일한 물체를 바라보는 방향)와 조명이나 계절 등의 다양한 변수가 존재한다.(이게 까다로움)
- siameses network를 이용하여 거리뷰와 위성뷰 각각을 CNN에 통과시켜 저차원의 feature로 만든다.
- 그리고 나서 matching score를 비교하는데
- pre-trained network를 사용하면 결과가 향상되나보다...당연한건가
- 그리고 새로운 loss function인 Distance Based Logistic(DBL)을 사용한다.
- 여기에 rotational variance(RI)와 orientation regression(OR)을 통합하여 학습한다.(성능 향상에 기여)

  ##### 1.1 Ralated work
  1) Image geolocalization
  - 이걸 써서 거리뷰로 해당 위치를 찾는 것이다.
  - 관련된 이전 연구들......
  2) Deep Learning
  - 패스..!
  
### 2. Dataset of street view and overhead image pairs
- 음...데이터셋 추출과정 및 몇가지 사항이 적혀있는데..나중에 필요하면 한번더 읽어보는걸로!

### 3. Cross-view matching and ranking with CNN
- 학습단계에선, 거리뷰와 위성뷰 pair가 positive examples로 제공되고, negative samples는 non-matched images로!
- 테스트 단계에선 이미지 pair를 받아서 이것이 match인지 아닌지 classification
- figure 3을 보면 좌/우로 카테고리가 나뉘어져있는데, 좌측을 upper bound(비교용)로 사용했다고 한다.

  ##### 3.1 Classification CNN for image matching
  - AlexNet을 변형하여 사용하였다.
  - 먼저 input image를 합쳤다.(AlexNet의 input은 1개이기 때문이다.)
  - 그래서 input은 6 channels이고(거리뷰+위성뷰 concat), 최종 output은 2개(binary면 1개 아닌가...??)
  - 그리고 loss는 network를 통과해 나온 값과 0또는1의 값을 갖는 label의 차이를 줄이는 binary crossentropy
  
  ##### 3.2 Siamese-like CNN for learning image features
  - image matching과 retrieval(유사도를 뱉어내는 검색)에 사용된다고 하는데...
  <p align="center">
  <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL_figure.PNG?raw=true' height=350 width=550>
  </p>
  - 두개의 CNN이 각각 거리뷰와 위성뷰를 받아 feature vector를 각각 뱉어낸다.(f(A), f(B))
  - f(A)와 f(B)의 거리의 제곱을 D라 하고, loss function은 다음과 같다.
    - `L(A,B, l) = l * D + (1 − l) * max(0,m − D)`
  - 이 loss function은 label(l)이 1인 경우 두 벡터가 유사하도록, 0이면 멀어지도록 만든다고 한다.
  - l=0의 경우, m-D를 최소화 하기위해선 D를 maximize시켜야 하고 결국 이 경우 멀어지게 된다.
  - 기존 siamese network에선 두 network가 weights를 공유하는데 여기선 공유하지 않는다.
  
  ##### 3.3 Siamese-classification hybrid network
  - 핵심은 AlexNet 두개가 각각 거리뷰와 위성뷰를 convolution시킨 후 FC layer에서 합친다.
  
  ##### 3.4 Triplet network for learning image features
  - L(A,B,C) = max(0,m + D(A,B) − D(A,C)) ; 이걸 hinge loss라고 하는 것 같다.
    - [what is hinge loss?](https://ratsgo.github.io/machine%20learning/2017/10/12/terms/)
  
  ##### 3.5 Learning image representation with distance-based logistic loss
  - siamese network의 경우(3번째 network) 
    - contrastive loss대신에 마지막 layer를 DBL layer로 두고 다음의 output을 추출한다.
    <p align="center">
      <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL_output.PNG?raw=true' height=70 width=250>
    </p>
    - 위 값 p(A,B)가 0~1사이의 값을 뱉어내 마치 distance로부터 매칭확률처럼 보인다.
    - 이렇게 나온 p(A,B)로 logloss를 구한다.
  - triplet network의 경우(4번째 network)
    <p align="center">
      <img src='https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/DBL_output_triplet.PNG?raw=true' height=80 width=250>
    </p>
  - **Test시 DBL layer는 제외한다.**

### 4. Learning to perform rotation invariant matching
- 위성뷰 이미지에 rotation을 하는 방식에 관한 내용이다.
- rotation invariant라는게 간단히 말해 기존 이미지를 회전시킬거라는 의미인데,
- label은 고정하다보니 invaiant라는 말을 넣은 거 같다. (수학적으로 f(Rx)=f(x)=label)
- 다만, rotation을 하되 일정 구간에서 random rotation하는 방식이다.(Partial Rotation Invariant)
  - 예를 들면, 90으로 설정하면 -45~45도 사이의 값을 랜덤하게 뽑아 rotation시킨다는 거 같다.
  
  ##### 4.1 Partial rotation invariance by data augmentation
  - 90도로 정하면, -45~45도 사이의 각도로 회전을 시켜 augmentation시킨다.
  - train에서와는 다르게 test에서 16 rotation을 썻다고 한다.
  - 16개의 rotation samples들을 평균을 취해서 하나의 vector로 만들어준다.
  
  ##### 4.2 Learning better representations with orientation regression
  - 마지막 hidden layer로부터 나온 features(거리뷰/위성뷰)들을 concat시켜준다.
  - 이후 2개의 fc layer(1 hidden, 1 output layer)를 통과시킨다.
  - 그런다음 이 값과 orientation의 RMSE를 구한다.(orientation이 좌표인가..? 뭘 의미하는거지?)
  
### 5. Experiments
  


