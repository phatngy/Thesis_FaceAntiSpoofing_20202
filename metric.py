import numpy as np
import torch
from scipy import interpolate

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils import *
import roc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)

    acc = float(tp +tn ) /dist.shape[0]
    return tpr, fpr, acc

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def ACER(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    apcer = fp / (tn*1.0 + fp*1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer,tp, fp, tn,fn

def TPR_FPR( dist, actual_issame):
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []

    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)

    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)

    thresholds = np.arange(0.0, 1.0, 0.00001)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)
            TPR = tp / (tp*1.0 + fn*1.0)

        fpr[threshold_idx] = FPR
    
    Thrs = {
        "TPR@FPR=10E-2": 0.01, 
        "TPR@FPR=10E-3": 0.001, 
        "TPR@FPR=10E-4": 0.0001
        }

    TPRs = {
        "TPR@FPR=10E-2": 0.01, 
        "TPR@FPR=10E-3": 0.001, 
        "TPR@FPR=10E-4": 0.0001
        }
    for k, fpr_target in Thrs.items():
        if np.max(fpr) >= fpr_target:
            f = interpolate.interp1d(np.asarray(fpr), thresholds, kind= 'slinear', fill_value="extrapolate")
            # f = interpolate.interp1d(np.asarray(fpr), thresholds, kind= 'slinear')
            threshold = f(fpr_target)
        else:
            threshold = 0.0
        Thrs[k] = threshold
        tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

        FPR = fp / (fp * 1.0 + tn * 1.0)
        TPR = tp / (tp * 1.0 + fn * 1.0)

        TPRs[k] = TPR
    print(Thrs)
    print(TPRs)
    return TPRs

import torch.nn.functional as F
def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob

def do_valid( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []

    for input, truth in test_loader:
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]

def do_valid_test( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []
    batch_time = AverageMeter()
    points_to_plot = []
    pca = PCA(n_components=40)
    tsne = TSNE(n_components=2)

    for i, (input, truth) in enumerate(tqdm(test_loader)):
    # for input, truth in test_loader:
        # print(input.size())
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            start = time.time()
            logit,_,ft   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            correct, prob = metric(logit, truth)
            batch_time.update(time.time() - start)
            loss    = criterion(logit, truth, False)
            # ft = ft.view(b, n, 256)
            # p = torch.mean(ft, dim=1, keepdim=False).float().detach()
            # p_result = pca.fit_transform(np.array(p.cpu()))
            # p_result = tsne.fit_transform(p_result)
            # points = torch.cat((torch.Tensor(p_result).cuda(), truth.float().detach().view(-1, 1)), 1)
            # print(points)
        # points_to_plot = np.append(points_to_plot, np.array(points.cpu()))
        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))   
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())
        # if i >= 3:
        #     break
    # assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------
    # plot(points_to_plot)
    correct = np.concatenate(corrects)
    # print(losses)
    # loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    plot_curve(tpr, fpr)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])
    predicted_list = np.argmax(probs, axis=1)
    test_report = classification_report(labels.tolist(), predicted_list.tolist(), target_names=['fake', 'real'])
    res_confusion_mtx = confusion_matrix(labels, predicted_list).ravel()
    tn, fp, fn, tp = res_confusion_mtx
    print(test_report)
    print('confusion matrix', res_confusion_mtx)
    apcer = fp/(tn + fp)
    npcer = fn/(fn + tp)
    acer_ = (apcer + npcer)/2
    # print(acer, acer_)
    metrics = roc.cal_metric(labels, probs[:, 1])
    eer = metrics[0]
    tprs = metrics[1]
    auc = metrics[2]
    xy_dic = metrics[3]
    TPRs = TPR_FPR(probs[:, 1], labels)
    print(tprs)
    if b == 1:
        print(batch_time.avg)
    return valid_loss,[probs[:, 1], labels], TPRs

def infer_test( net, test_loader):
    valid_num  = 0
    probs = []

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)
        input = input.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)
            prob = F.softmax(logit, 1)

        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())

    probs = np.concatenate(probs)
    return probs[:, 1]



