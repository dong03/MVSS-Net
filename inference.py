import sys
import os
import numpy as np
from common.utils import Progbar, read_annotations
import torch.backends.cudnn as cudnn
from models.mvssnet import get_mvss
from models.resfcn import ResFCN
import torch.utils.data
from common.tools import inference_single
import cv2
from apex import amp
import argparse


def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--model_path", type=str, help="Path to the pretrained model", default="ckpt/mvssnet.pth")
    parser.add_argument("--test_file", type=str, help="Path to the image list")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--resize", type=int, default=512)
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_opt()
    print("in the head of inference:", opt)
    cudnn.benchmark = True

    # read test data
    test_file = opt.test_file
    dataset_name = os.path.basename(test_file).split('.')[0]
    model_type = os.path.basename(opt.model_path).split('.')[0]
    if not os.path.exists(test_file):
        print("%s not exists,quit" % test_file)
        sys.exit()
    test_data = read_annotations(test_file)
    new_size = opt.resize

    # load model
    model_path = opt.model_path
    if "mvssnet" in model_path:
        model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=3,
                         )
    elif "fcn" in model_path:
        model = ResFCN()
    else:
        print("model not found ", model_path)
        sys.exit()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        print("load %s finish" % (os.path.basename(model_path)))
    else:
        print("%s not exist" % model_path)
        sys.exit()
    model.cuda()
    amp.register_float_function(torch, 'sigmoid')
    model = amp.initialize(models=model, opt_level='O1', loss_scale='dynamic')
    model.eval()

    save_path = os.path.join(opt.save_dir, dataset_name, model_type)
    print("predicted maps will be saved in :%s" % save_path)
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        progbar = Progbar(len(test_data), stateful_metrics=['path'])
        pd_img_lab = []
        lab_all = []
        scores = []

        for ix, (img_path, _, _) in enumerate(test_data):
            img = cv2.imread(img_path)
            ori_size = img.shape
            img = cv2.resize(img, (new_size, new_size))
            seg, _ = inference_single(img=img, model=model, th=0)
            save_seg_path = os.path.join(save_path, 'pred', os.path.split(img_path)[-1].split('.')[0] + '.png')
            os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
            seg = cv2.resize(seg, (ori_size[1], ori_size[0]))
            cv2.imwrite(save_seg_path, seg.astype(np.uint8))
            progbar.add(1, values=[('path', save_seg_path), ])

