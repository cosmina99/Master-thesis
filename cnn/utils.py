import numpy as np
from torchvision import transforms
from cnn.cnn import *

#Evaluation metrics

def gap(pred, score, class_ids):
    #Implementation of the GAP metric

    rel = np.zeros(len(pred)) #rel is the binary indicator, 1 if prediction is correct, 0 if false
    rel = np.equal(pred,class_ids)

    idx = np.argsort(-np.array(score), axis=0)
    rel = rel[idx] #sorting rel in descending order

    nq_object = len(pred)-len([x for x in class_ids if x == [-1]]) #label of a distractor is -1

    prec = np.cumsum(rel) / (1+np.array(np.arange(len(rel))))
    gap = (prec * rel).sum() / nq_object

    return gap

def gap_non_distr(pred, score, class_ids):
    #Same as GAP, but doesn't take distractors into account

    pred = np.array(pred)
    score = np.array(score)
    
    class_ids = np.array(class_ids)
    idx_distr = np.where(class_ids!=-1)

    rel = np.zeros(len(pred[idx_distr]))
    rel = np.equal(pred[idx_distr],class_ids[idx_distr])

    idx = np.argsort(-np.array(score[idx_distr]), axis=0)
    rel = rel[idx]

    nq_object = len(pred[idx_distr])

    prec = np.cumsum(rel) / (1+np.array(range(len(rel))))
    gap = (prec * rel).sum() / nq_object

    return gap

def classif_accuracy(predictions,ground_truth):
    #Implementation of the accuracy metric

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    #distractors are not taken into account
    query_idx = np.where(ground_truth!=-1)

    accuracy = np.sum(np.equal(predictions[query_idx],ground_truth[query_idx])) / len(predictions[query_idx])

    return accuracy

def evaluate(preds,confs,test_labels):
    #Implementing all the evaluation metrics
    gap_score = gap(preds, confs, test_labels)

    gap_score_non_distr = gap_non_distr(preds, confs, test_labels)

    acc_score = classif_accuracy(preds,test_labels)

    print("GAP is :" + str(gap_score))
    print("GAP without distractors is :" + str(gap_score_non_distr))
    print("Accuracy without distractors is :" + str(acc_score))

    return gap_score, gap_score_non_distr, acc_score

#Utils for kNN classifier

def normalise(a,axis=-1,order=2):
    #L2 normalisation of descriptors

    l2 = np.linalg.norm(a, order, axis)
    l2[l2==0] = 1

    return a / np.expand_dims(l2, axis)


def apply_pca_whiten_and_normalise(X, m, P):
    #Apply given learned pca whitening matrix to given descriptors after subtracting the learned mean

    X = np.dot(X-m, P)

    return normalise(X,axis = 1)


def estimate_pca_whiten_with_shrinkage(X, shrinkage=1.0, dimensions=None):
    '''
    Learn pca whitening with given shrinkage
    "dimensions" argument is the dimensions that we keep after the pca-whitening procedure
    shrinkage = 1 corresponds to pca whitening
    shrinkage = 0 corresponds to pca
    '''
    n,d = X.shape[0],X.shape[1]

    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*n)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]   
    eigval = eigval[:dimensions]
    eigvec = eigvec[:,:dimensions]
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval,0.5*shrinkage))), eigvec.T)

    return m,P.T

#Augmentations based on ImageNet statistics

def augmentation(key):

	augment_dict = {

		"augment":
			transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])

	}

	return augment_dict[key]

