# Center Loss
Implementing center loss with pytorch

### Introduction

[Center loss](https://ydwen.github.io/papers/WenECCV16.pdf) 논문을 읽고 논문 속 toy example로 이해를 확인하고자 했습니다.
학습 데이터셋은 mnist를 사용했습니다.

### Requirements
After cloning the repo, run this line below:
```
pip install -r requirements.txt
```

### Usage
```
# train
python -m centerloss.train

# inference
python -m centerloss.inference
```

### Statistics

##### 1. train summary

| model   | Loss function  | lambda |
|---------|----------------|--------|
| LeNet++ | SoftMax        | -      |
| LeNet++ | SoftMax+Center | 1      |
| LeNet++ | SoftMax+Center | 0.1    |
| LeNet++ | SoftMax+Center | 0.01   |
| LeNet++ | SoftMax+Center | 0.001  |


- optimizer1: SGD(lr=0.01, momentum=0.9, weight decay=0.0005)
- optimizer2: SGD(lr=0.5)
- total epochs: 100
- params: 796K


##### 2. plots

- softmax loss only

<img src="./result/softmax_train.png" width="400" height="300">
<img src="./result/softmax_valid.png" width="400" height="300">

- softmax loss + center loss

<img src="./result/center_train.png" width="400" height="300">
<img src="./result/center_valid.png" width="400" height="300">

- softmax loss + center loss with different lambda

<img src="./result/center(0.001).png" width="400" height="300">
<img src="./result/center_train.png" width="400" height="300">

<img src="./result/center(0.1).png" width="400" height="300">
<img src="./result/center(1).png" width="400" height="300">
