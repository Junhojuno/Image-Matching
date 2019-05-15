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
  

  ##### 1.1 Ralated work
  1) Image geolocalization
  - 이걸 써서 거리뷰로 해당 위치를 찾는 것이다.
  - 관련된 이전 연구들......
  2) Deep Learning
  - 패스..!
  
### 2. Dataset of street view and overhead image pairs
- Google panorama image에 위치정보(geo-tag, depth)가 포함되어 있는 것 같다.

- 음...데이터셋 추출과정 및 몇가지 사항이 적혀있는데..나중에 필요하면 한번더 읽어보는걸로!

### 3. Cross-view matching and ranking with CNN
- 앞에서 task가 2개로 나뉜다고 했다.(matching과 ranking)
- matching task를 먼저 설명하면
  - 학습시에 matched 데이터가 (street, overhead) pair로 들어서 학습시킵니다.(positive examples)
  - 테스트시에 위에서 학습시킨 모델로 match or not 분류한다.
- figure 3를 보면, CNN이 2개의 카테고리로 구분되는데,
  - 첫번째 카테고리(figure3의 좌측1열)는 matching을 담당하는 구조들이다.
  - convolution layer까지만 통과하고 나온 representation들은 feature embedding에 사용된다.
  - 두번째 카테고리(figure3의 우측1열)는...(따로 말은 없지만 ranking이지 않을까?)
  - 두번째 카테고리에서 새로운 loss function인 DBL을 사용한다.
- 정리하면, 두 카테고리의 조합으로 matching과 ranking task를 해결한다는 것이다.
  - 조합을 한다면 아래와 같이 각각 학습시킨다는 것인가...?
  - 첫번째 카테고리에서 위에꺼, 두번째 카테고리에서 위에꺼를 예로 들면,
  - (street, overhead) --> AlexNet --> match or not
  - (street, overhead) --> Siamese Network(AlexNet 2개) --> ranking
- 이 논문에서는 Siamese-Hybrid Network를 제안한다
  - (street, overhead) --> Siamese Network --> concatenation --> fc --> fc --> 
  

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
  


