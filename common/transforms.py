from albumentations.pytorch.functional import img_to_tensor
import pdb




def direct_val(imgs):
    normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}
    if len(imgs) != 1:
        pdb.set_trace()
    imgs = img_to_tensor(imgs[0], normalize).unsqueeze(0)
    return imgs