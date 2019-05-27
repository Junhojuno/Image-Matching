## Triplet-based Deep Similarity Learning for Person Re-Identification
[논문 링크](https://arxiv.org/pdf/1802.03254.pdf)

![figure2](https://github.com/Junhojuno/Image-Matching/blob/master/paper-review/img/triplet-based_figure2.PNG?raw=true)

### 0. Abstract
- person re-identification을 위한 새로운 framework을 제안
- triplet 기반의 유사도 학습
- same class label의 두 이미지와 different class label의 이미지 하나를 input으로 넣는다.
- 3개의 이미지를 CNN을 통과시켜 deep feature representation을 뽑아낸다.
- 같은 class의 이미지끼리는 distance를 줄이고, 다른 class와는 늘린다.(euclidean distance)
- **좀 특이한 점은 1개의 데이터셋이 아닌 6개의 서로다른 데이터셋으로 모델을 학습시킨다는 것.**
- 이와 관련하여 double sampling 구조가 추가되는데, 이 점은 좀 더 읽어봐야 알 수 있을 거 같다.

### 정리하면...
- 데이터는 서로 다른 6개의 datasets에서 3개를 뽑아 input으로 넣는다.(anchor image, same, different)
- 그러나 위와 같이하면 input으로 들어갈 image의 조합이 굉장히 많아지게 된다.
- 이를 해결하기 위해 double sampling scheme을 사용한다.
- 

| Keywords | 
|:--------:|
| triplet-based |
| six different datasets |
| double sampling scheme |

### 1. Introduction
- 6종류의 데이터셋이 input으로 들어가는데,
- 6개의 이미지가 input으로 들어가는게 아니고, 6종류에서 3개만 뽑아서 triplet에 집어넣는다.(anchor image, same, different)
- 이렇게 되면 input으로 넣을 이미지 조합이 굉장히 많을 것이다.
- 이를 해결하기위해 **double sampling scheme**을 사용하였다.

### 2. Related Work
- pass (추후 필요하면 읽도록 하겠다.)

### 3. The Proposed Approach
- feature representation을 jointly하게 학습시킬 pipeline에 대해 다룬다.

  ##### 3.1 The Overall Framework
  - 위의 설명과 동일하다.
  - `여기서 X의 의미는 뭐지?`
  
  ##### 3.2 Double Sampling
  - 앞서 언급했듯이 input으로 들어갈 이미지의 조합이 굉장히 다양하고,
  - 이게 다 CNN으로 들어가면 계산량도 늘어난다. 그래서 double sampling을 쓰는데....
  - 먼저 mini-batch를 구성한다.
    - 서로 다른 10명의 사람을 선정하고, 각 사람마다 5장의 사진을 뽑는다.
    - 이렇게 완성된 5 x 10 matrix가 하나의 mini-batch인 것이다.
  - 구성한 mini-batch를 CNN에 넣어 feature representation을 만들고 cache에 저장한다.
  - 10명중 1명을 뽑고, 이 1명에 해당하는 2개의 feature representation을 뽑는다.(F0, F+)
  - 나머지 9명중 1명을 뽑아 이사람의 feature representation을 뽑는다.(F-)
