# Public and Self-made Datasets
Txts in this directory provide the index lists of datasets used in [***MVSS-Net***](https://github.com/dong03/MVSS-Net). 

Each line in the txts contains:
```angular2html
img_path mask_path label
```
Please refer to the [paper](https://arxiv.org/abs/2104.06832) and [code](https://github.com/dong03/MVSS-Net) for more details.
## DEFACTO-84k & 12k
+ You need to download [DEFACTO](https://defactodataset.github.io/) and [MSCOCO](https://cocodataset.org/) and obey their usage policies.
+ Labels with manipulation types can be found [here](https://github.com/dong03/DEFACTO-84k-12k).

## CASIAv1plus 
There seems to be an overlap between authentic images from CASIAv1 and CASIAv2, which is seldom discussed. To avoid data leaking, we collect 782 authentic images from [Corel](https:/sites.google.com/site/dctresearch/Home/content-based-image-retrieval), the source dataset of CASIAv1, and bulid CASIAv1plus togeher with non-overlap images as well as manipulated images.

For future evaluation, we recommend to use CASIAv1plus instead of CASIAv1.

