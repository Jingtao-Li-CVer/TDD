from gc import get_threshold
import logging
import matplotlib.pyplot as plt
from numpy import ndarray as NDArray
from sklearn.metrics import roc_auc_score, roc_curve
import os
import random
import numpy as np
from scipy import integrate
from tqdm import tqdm
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import sys
sys.path.append("/home/ljt21/ad/RSAD/metrics/")


def compute_auroc(epoch: int, ep_amaps, ep_gt, working_dir: str, save_image=False,) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    Args:
        epoch (int): Current epoch
        ep_amaps (NDArray): Anomaly maps in a current epoch
        ep_gt (NDArray): Ground truth masks in a current epoch
    Returns:
        float: AUROC score
    """
    save_dir = os.path.join(working_dir, "epochs-" + str(epoch))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    y_score, y_true = [], []
    for i, (amap, gt) in enumerate(tqdm(zip(ep_amaps, ep_gt))):
        anomaly_scores = amap[np.where(gt == 0)]
        normal_scores = amap[np.where(gt == 1)]
        y_score += anomaly_scores.tolist()
        y_true += np.zeros(len(anomaly_scores)).tolist()
        y_score += normal_scores.tolist()
        y_true += np.ones(len(normal_scores)).tolist() 
        
    scoreDF = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    scoreDt = abs(integrate.trapz(tpr, thresholds))
    scoreFt = abs(integrate.trapz(fpr, thresholds))
    scoreTD = scoreDF + scoreDt
    scoreBS = scoreDF - scoreFt
    scoreODP = 1 + scoreDt - scoreFt
    scoreTDBS = scoreDt - scoreFt
    scoreMDP = scoreDt + 1 - scoreFt
    scoreSNPR = scoreDt / scoreFt
    scoreOADP = scoreDF + scoreDt + 1 - scoreFt
    logging.info("scoreDF: " + str(scoreDF))
    logging.info("scoreDt: " + str(scoreDt))
    logging.info("scoreFt: " + str(scoreFt))
    logging.info("scoreTD: " + str(scoreTD))
    logging.info("scoreBS: " + str(scoreBS))
    logging.info("scoreODP: " + str(scoreODP))
    logging.info("scoreTDBS: " + str(scoreTDBS))
    logging.info("scoreMDP: " + str(scoreMDP))
    logging.info("scoreSNPR: " + str(scoreSNPR))
    logging.info("scoreOADP: " + str(scoreOADP))


    if save_image:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {round(scoreDF, 3)}")
        plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
        plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,"roc_curve.png"))
        plt.close()

    return scoreDF
