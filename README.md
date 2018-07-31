
## Final result
### 1차 30/280
### 2차 10/30

![최종결과](https://github.com/junwoopark92/2018-SK-TnB-CodeChallenge/blob/master/asset/%C2%A0%EC%B5%9C%EC%A2%85%EC%88%9C%EC%9C%84.png?raw=true)

## history

### 0. baselien model
기간동안 시청이력이 많은 순서로 50개의 영화를 동일하게 추천
mAP:14%

### 1. lstm model
시청이력 시퀀스를 입력으로 받아 이후 시청할 영화 1개를 예측
영화 예측확률로 argmax 50으로 50개의 영화 추천
mAP: 25%
10epoch: 25.7%

### 2. lstm continuous output model
lstm의 output을 onehot이 아닌 word2vec의 벡터를 예측하도록 한후 cosine most similiar 50으로 50개의 영화 추천
1epoch mAP: 19.2%
5epoch mAP: 19.4%


### 3. lstm model + add movie release month feature
기존 lstm model에 movie의 release month를 추가
