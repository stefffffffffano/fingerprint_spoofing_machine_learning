import numpy
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets 
import scipy.linalg
import math

def load(fileName):
    Dlist=[]
    L=[]
    with open(fileName) as f:
        for line in f:
            attrs = line.split(',')[0:-1]
            attrs = vcol(numpy.array([float(i) for i in attrs]))
            Dlist.append(attrs)
            label= line.split(',')[-1].strip()
            L.append(int(label))
    return numpy.hstack(Dlist),numpy.array(L)

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D): 
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

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

def predictLabelsLog(logSJoint):
    logSJoint += vcol(numpy.array([math.log(1/2),math.log(1/2)]))
    logSMarginal = scipy.special.logsumexp(logSJoint,0)
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)
    predictedLables = SPost.argmax(0)
    return predictedLables

def MVG(DTR,LTR,DTE,LTE):
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, C0)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, C1)
    llr = pdfGau1Log - pdfGau0Log
    predictions = numpy.where(llr >= 0, 1, 0)
    return predictions

def tiedGaussian(DTR,LTR,DTE,LTE):
    Sb,Sw=compute_Sb_Sw(DTR,LTR)
    #Sw is the within class covariance matrix
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, Sw)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, Sw)
    logSJoint = numpy.vstack((pdfGau0Log, pdfGau1Log))
    return predictLabelsLog(logSJoint)

def naiveBayes(DTR,LTR,DTE,LTE):
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    C0diag=C0*numpy.eye(C0.shape[0])
    C1diag=C1*numpy.eye(C1.shape[0])
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, C0diag)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, C1diag)
    logSJoint = numpy.vstack((pdfGau0Log, pdfGau1Log))
    return predictLabelsLog(logSJoint)

def compute_pca(D, m):
    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D

def calculateErrorRate(predictions,LTE):
    print('Number of erros:', (predictions != LTE).sum(), '(out of %d samples)' % (LTE.size))
    print('Error rate: %.1f%%' % ( (predictions != LTE).sum() / float(LTE.size) *100 ))


if __name__ == '__main__':
    D, L = load('trainData.txt')
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    #MVG
    print("Error rate for MVG: ")
    predictions = MVG(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictions,LTE)

    #tied Gaussian Model
    print("Error rate for tied Gaussian Model: ")
    predictedLables=tiedGaussian(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictedLables,LTE)
    
    #Naive Bayes Gaussian model
    print("Error rate for Naive Bayes classifier: ")
    predictedLables = naiveBayes(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictedLables,LTE)

    #analysis
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    Corr0 = C0 / ( vcol(C0.diagonal()**0.5) * vrow(C0.diagonal()**0.5) )
    Corr1 = C1 / ( vcol(C1.diagonal()**0.5) * vrow(C1.diagonal()**0.5) )
    #print(Corr0)
    #print(Corr1)
    print("Error rate for the classification keeping only the first 4 features: ")
    #keeping only the first 4 features
    D4 = D[0:4,:]
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D4, L)

    #MVG
    print("Error rate for MVG: ")
    predictions = MVG(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictions,LTE)

    #tied Gaussian Model
    print("Error rate for tied Gaussian Model: ")
    predictedLables=tiedGaussian(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictedLables,LTE)
    
    #Naive Bayes Gaussian model
    print("Error rate for Naive Bayes classifier: ")
    predictedLables = naiveBayes(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictedLables,LTE)

    print("Error rate for the classification keeping only feature 1 and 2: ")
    #only features 1-2
    D12 = D[0:2,:]
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D12, L)

    #MVG
    print("Error rate for MVG: ")
    predictions = MVG(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictions,LTE)

    #tied Gaussian Model
    print("Error rate for tied Gaussian Model: ")
    predictedLables=tiedGaussian(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictedLables,LTE)

    print("Error rate for the classification keeping only features 3 and 4: ")
    #only features 3-4
    D34 = D[2:4,:]
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D34, L)

    #MVG
    print("Error rate for MVG: ")
    predictions = MVG(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictions,LTE)

    #tied Gaussian Model
    print("Error rate for tied Gaussian Model: ")
    predictedLables=tiedGaussian(DTR,LTR,DTE,LTE)
    calculateErrorRate(predictedLables,LTE)
    
    #Let's try to apply PCA reducing the feature space from 6 up to 1
    print("PCA applied to reduce the feature space from 6 up to 1: ")
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L) 
    for i in range(6):
        m = 6-(i)
        UPCA = compute_pca(DTR, m = m) 
        DTR_pca = apply_pca(UPCA, DTR)   
        DTE_pca = apply_pca(UPCA, DTE) 
        print("Error rate for MVG with m = %d: " %m)
        predictions = MVG(DTR_pca,LTR,DTE_pca,LTE)
        calculateErrorRate(predictions,LTE)
        #tied Gaussian Model
        print("Error rate for tied Gaussian Model with m = %d: " %m)
        predictedLables=tiedGaussian(DTR_pca,LTR,DTE_pca,LTE)
        calculateErrorRate(predictedLables,LTE)
        #Naive Bayes Gaussian model
        print("Error rate for Naive Bayes classifier with m = %d: " %m)
        predictedLables = naiveBayes(DTR_pca,LTR,DTE_pca,LTE)
        calculateErrorRate(predictedLables,LTE)
       
        

        
      


 