# Localizing and Orienting Street Views Using Overhead Imagery (2017)
: 인공위성의 사진을 통해 거리뷰 사진을 보고 위치를 추정한다.

| Keywords |
|:----------:|
| image geolocalization |
| image matching (cross domain matching) |
| 4 Deep CNN architectures |
| Distance Based Logistic |


### Abstact
- 지상에서 찍은 사진의 위치와 방향(?)을 위성사진과 매칭시켜 결정해보는 것을 목표로 함(image matching task)
- 이와 관련하여 deep CNN architectures를 볼건데, 각각의 장단점이 있다.(Classification, Hybrid vs Siamese, Triplet)
- image matching task에 몇가지 어려움이 있다.(거리뷰와 위성사진의 극단적 시점차이, **거리뷰의 방위각을 모른다는점**)
- 중간에 rotation invariance라는 말이 나오는데,
  - **rotational invariance** if its value does not change when arbitrary rotations are applied to it
  - 여기서 말하는 value가 target(label)을 의미하는건가?
- 명확하게 방위각같은 orientation을 지정해주면 accuracy가 향상된다.(당연한 얘기)
- siamese network baseline보다 약 2.5배 향상된 아키텍쳐를 개발하였다고 함.

### Introduction
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
  
### Dataset of street view and overhead image pairs
- 음...데이터셋 추출과정 및 몇가지 사항이 적혀있는데..나중에 필요하면 한번더 읽어보는걸로!

### Cross-view matching and ranking with CNN
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
  - 두개의 CNN이 각각 거리뷰와 위성뷰를 받아 feature vector를 각각 뱉어낸다.
  - 








