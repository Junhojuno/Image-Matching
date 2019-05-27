## Triplet-based Deep Similarity Learning for Person Re-Identification
[논문 링크](https://arxiv.org/pdf/1802.03254.pdf)

### Abstract
- person re-identification을 위한 새로운 framework을 제안
- triplet 기반의 유사도 학습
- same class label의 두 이미지와 different class label의 이미지 하나를 input으로 넣는다.
- 3개의 이미지를 CNN을 통과시켜 deep feature representation을 뽑아낸다.
- 같은 class의 이미지끼리는 distance를 줄이고, 다른 class와는 늘린다.(euclidean distance)
- **좀 특이한 점은 1개의 데이터셋이 아닌 6개의 서로다른 데이터셋으로 모델을 학습시킨다는 것.**
- 이와 관련하여 double sampling 구조가 추가되는데, 이 점은 좀 더 읽어봐야 알 수 있을 거 같다.

| Keywords | 
|:--------:|
| triplet-based |
| six different datasets |
| double sampling scheme |
