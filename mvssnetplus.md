# MVSS-Net++
+ **MVSS-Net++, an improved version of MVSS-Net, more details are reported in the [paper](https://arxiv.org/abs/2112.08935).**
+ **Models listed are trained on CASIAv2.**

#### Performance metric: pixel-f1 (threshold=0.5)
|   Model  |  CASIAv1plus | Columbia |  COVER | DEFACTO-12k | IMD |
|:--------:|:-------:|:--------:|:------:|:-----------:|:-----------:|
| MVSS-Net++ |  0.513   | 0.660    | 0.482  | 0.095       | 0.270|
| MVSS-Net |  0.452   | 0.638    | 0.453  | 0.137       | 0.260|
| FCN      |  0.441   | 0.223    | 0.199  | 0.130       | 0.210|


#### Performance metric: AUC
|   Model  | CASIAv1plus | Columbia |  COVER | DEFACTO-12k | IMD|
|:--------:|:-------:|:--------:|:------:|:-----------:|:------:|
| MVSS-Net++ | 0.862   | 0.984    | 0.726  | 0.531     |0.658|
| MVSS-Net | 0.932   | 0.980    | 0.731  | 0.573       |0.656|
| FCN      | 0.769   | 0.762    | 0.541  | 0.551       |0.502|


#### Performance metric: Image-level F1 (threshold=0.5)

|   Model  |  CASIAv1plus | Columbia |  COVER | DEFACTO-12k |IMD|
|:--------:|:-------:|:--------:|:------:|:-----------:|:------:|
| MVSS-Net++ |  0.694   | 0.930    | 0.685  | 0.478       |0.614| 
| MVSS-Net |  0.759   | 0.802    | 0.244  | 0.404       |0.355
| FCN      |  0.684   | 0.481    | 0.180  | 0.458       |0.262
