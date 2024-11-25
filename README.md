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

|  class  | Images | Instances | Box(P) | Box(R) | Box(mAP50) | Box(mAP50-95) | 
|---------|--------|-----------|--------|--------|------------|---------------|
|  All    | 3347   | 10299     | 0.891  | 0.798  | 0.869      | 0.578         |

- optimizer: Adam(lr=0.01, momentum=0.937)
- total epochs: 100
- params: 2.5M
- GFLOPs: 6.3


##### 2. plots

<img src="./result/plots.png" width="600" height="300">

<img src="./result/PR_curve.png" width="600" height="400">
<img src="./result/confusion_matrix_normalized.png" width="600" height="400">

##### 3. inference result

<img src="./result/output.png" width="810" height="1080">

