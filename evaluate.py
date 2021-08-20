import os
import cv2
import numpy as np
import sys
import argparse
import pdb
from sklearn import metrics
from tqdm import tqdm
from common.utils import read_annotations, calculate_img_score, calculate_pixel_f1
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--pred_dir', type=str, default='save_out')
    parser.add_argument('--gt_file', type=str, default='None')
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument("--model_name", type=str, help="Path to the pretrained model", default="ckpt/mvssnet.pth")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    annotation_file = opt.gt_file
    dataset = os.path.basename(annotation_file).split('.')[0]
    model_type = os.path.basename(opt.model_name).split('.')[0]

    if not os.path.exists(annotation_file):
        print("%s not exists, quit" % annotation_file)
        sys.exit()
    annotation = read_annotations(annotation_file)
    scores, labs = [], []
    f1s = [[], [], []]

    results = []
    for ix, (img, mask, lab) in enumerate(tqdm(annotation)):
        pred_path = os.path.join(opt.pred_dir, dataset, model_type, 'pred', os.path.basename(img).split('.')[0] + '.png')
        try:
            pred = cv2.imread(pred_path, 0) / 255.0
        except:
            print("%s not exists" % pred_path)
            continue
        score = np.max(pred)
        scores.append(score)
        labs.append(lab)
        f1 = 0
        if lab != 0:
            try:
                gt = cv2.imread(mask, 0) / 255.0
            except:
                pdb.set_trace()
            if pred.shape != gt.shape:
                print("%s size not match" % pred_path)
                continue
            pred = (pred > opt.th).astype(np.float)
            try:
                f1, p, r = calculate_pixel_f1(pred.flatten(), gt.flatten())
            except:
                import pdb
                pdb.set_trace()
            f1s[lab-1].append(f1)

    fpr, tpr, thresholds = metrics.roc_curve((np.array(labs) > 0).astype(np.int), scores, pos_label=1)
    try:
        img_auc = metrics.roc_auc_score((np.array(labs) > 0).astype(np.int), scores)
    except:
        print("only one class")
        img_auc = 0.0
    with open(os.path.join(opt.pred_dir, dataset, model_type, 'roc.pkl'), 'wb') as f:
        pickle.dump({'fpr': fpr, 'tpr': tpr, 'th': thresholds, 'auc': img_auc}, f)
        print("roc save at %s" % (os.path.join(opt.pred_dir, dataset, model_type, 'roc.pkl')))

    meanf1 = np.mean(f1s[0]+f1s[1]+f1s[2])
    print("pixel-f1: %.4f" % meanf1)

    acc, sen, spe, f1_imglevel, tp, tn, fp, fn = calculate_img_score((np.array(scores) > 0.5), (np.array(labs) > 0).astype(np.int))
    print("img level acc: %.4f sen: %.4f  spe: %.4f  f1: %.4f auc: %.4f"
          % (acc, sen, spe, f1_imglevel, img_auc))
    print("combine f1: %.4f" % (2*meanf1*f1_imglevel/(f1_imglevel+meanf1+1e-6)))