import sklearn.datasets
import scipy.linalg
import numpy
import matplotlib
import matplotlib.pyplot as plt
import math

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def predictLabels(S,priorProb):
    SJoint = priorProb * S
    SPost= SJoint/vrow(SJoint.sum(0))
    predictedLables = SPost.argmax(0)
    return predictedLables

def predictLabelsLog(logSJoint):
    logSJoint += vcol(numpy.array([math.log(1/3),math.log(1/3),math.log(1/3)]))
    logSMarginal = scipy.special.logsumexp(logSJoint,0)
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)
    predictedLables = SPost.argmax(0)
    return predictedLables

if __name__ =='__main__':
    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    mu2,C2=compute_mu_C(DTR[:,LTR==2])

    pdfGau0 = numpy.exp(logpdf_GAU_ND(DTE, mu0, C0))
    pdfGau1 = numpy.exp(logpdf_GAU_ND(DTE, mu1, C1))
    pdfGau2 = numpy.exp(logpdf_GAU_ND(DTE, mu2, C2))
    # Creazione della matrice S concatenando le soluzioni insieme lungo l'asse delle righe
    S = numpy.vstack((pdfGau0, pdfGau1, pdfGau2))
    prior_prob = vcol(numpy.array([1/3,1/3,1/3]))
    predictedLables = predictLabels(S,prior_prob)
    #Prior probability for each class
    #SJointSol = numpy.load('Solution/SJoint_MVG.npy')
    #print (numpy.abs(SJointSol - SJoint).max())
    print('Number of erros:', (predictedLables != LTE).sum(), '(out of %d samples)' % (LTE.size))
    print('Error rate: %.1f%%' % ( (predictedLables != LTE).sum() / float(LTE.size) *100 ))
    
    #Start of the second part
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, C0)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, C1)
    pdfGau2Log = logpdf_GAU_ND(DTE, mu2, C2)
    logSJoint = numpy.vstack((pdfGau0Log, pdfGau1Log, pdfGau2Log))
    predictedLables=predictLabelsLog(logSJoint)
    print('Number of erros:', (predictedLables != LTE).sum(), '(out of %d samples)' % (LTE.size))
    print('Error rate: %.1f%%' % ( (predictedLables != LTE).sum() / float(LTE.size) *100 ))
    
    
    #Naive Bayes
    C0diag=C0*numpy.eye(C0.shape[0])
    C1diag=C1*numpy.eye(C1.shape[0])
    C2diag=C2*numpy.eye(C2.shape[0])
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, C0diag)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, C1diag)
    pdfGau2Log = logpdf_GAU_ND(DTE, mu2, C2diag)
    logSJoint = numpy.vstack((pdfGau0Log, pdfGau1Log, pdfGau2Log))
    predictedLables = predictLabelsLog(logSJoint)
    print('Number of erros:', (predictedLables != LTE).sum(), '(out of %d samples)' % (LTE.size))
    print('Error rate: %.1f%%' % ( (predictedLables != LTE).sum() / float(LTE.size) *100 ))

    #Tied Covariance Gaussian Classifier
    Sb,Sw=compute_Sb_Sw(DTR,LTR)
    #Sw is the within class covariance matrix
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, Sw)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, Sw)
    pdfGau2Log = logpdf_GAU_ND(DTE, mu2, Sw)
    logSJoint = numpy.vstack((pdfGau0Log, pdfGau1Log, pdfGau2Log))
    predictedLables = predictLabelsLog(logSJoint)
    print('Number of erros:', (predictedLables != LTE).sum(), '(out of %d samples)' % (LTE.size))
    print('Error rate: %.1f%%' % ( (predictedLables != LTE).sum() / float(LTE.size) *100 ))
    
    # log-likelihood ratios and MVG
    threshold=0
    #we assume that class 2 is true and class 1 is false
    Dclass1and2 = D[:,L != 0]
    Lclass1and2 = L[L != 0]
    (DTR12, LTR12), (DTE12, LTE12) = split_db_2to1(Dclass1and2, Lclass1and2)
    mu1,C1=compute_mu_C(DTR12[:,LTR12==1])
    mu2,C2=compute_mu_C(DTR12[:,LTR12==2])
    pdfGau1Log = logpdf_GAU_ND(DTE12, mu1, C1)
    pdfGau2Log = logpdf_GAU_ND(DTE12, mu2, C2)
    llr = pdfGau2Log - pdfGau1Log
    #llrSol = numpy.load('Solution/llr_MVG.npy')
    #print (numpy.abs(llrSol - llr).max())
    predictions = numpy.where(llr >= 0, 2, 1)
    print('Number of erros:', (predictions != LTE12).sum(), '(out of %d samples)' % (LTE12.size))
    print('Error rate: %.1f%%' % ( (predictions != LTE12).sum() / float(LTE12.size) *100 ))


    





