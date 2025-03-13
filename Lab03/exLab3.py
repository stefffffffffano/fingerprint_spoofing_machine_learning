import sys
import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn

def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((v.size,))
def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)



def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx in range(4):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(D2[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Setosa')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Versicolor')
            plt.scatter(D2[dIdx1, :], D2[dIdx2, :], label = 'Virginica')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
    plt.show()
def calculateProjection(C,D,m):
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:,0:m]
    y = numpy.dot(P.T, D)
    return y

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)



if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load(sys.argv[1])
    mu = mcol(D.mean(1))
    DC = D - mu
    C=1/(DC.shape[1])*numpy.dot(DC,DC.T)
    S=numpy.load('solPCA.npy')
    y=calculateProjection(C,D,2)
    #with the PCA we reduce the number of feature, from 4 to any number lower or equal than 4
    #so, basically, we will always have 150 rows, but columns are reduced to the fixed value of m
    D0 = y[:, L==0]
    D1 = y[:, L==1]
    D2 = y[:, L==2]
    #plt.figure
    #plt.hist(D0[0, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
    #plt.hist(D1[0, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
    #plt.hist(D2[0, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
    #plt.figure()
    #plt.xlabel('1')
    #plt.ylabel('2')
    #when we plot, we only put the dimensions we considered 
    #plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    #plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    #plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
    #plt.show()
    #finished the part related to the PCA, start of the LDA
    #I already have D0,D1,D2 as matrixes with features for data belonging to each class respectively
    #I need to calculate the mean for each class, mu0-mu1-mu2
    D0 = D[:,L==0]
    D1 = D[:,L==1]
    D2 = D[:,L==2]
    mu0 = mcol(D0.mean(1))
    mu1 = mcol(D1.mean(1))
    mu2 = mcol(D2.mean(1))
    DC0= D0-mu0
    DC1= D1-mu1
    DC2= D2-mu2
    Sw= 1/D.shape[1]*((DC0 @ DC0.T)+(DC1 @ DC1.T)+(DC2 @ DC2.T))
    #print(Sw)
    Sb= (1/D.shape[1])*(D0.shape[1]* ((mu0-mu) @ (mu0-mu).T) + D1.shape[1]* ((mu1-mu) @ (mu1-mu).T) + D2.shape[1]* ((mu2-mu) @ (mu2-mu).T))
    #print(Sb)
    #Sw and Sb are correct
    m=2
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    y = numpy.dot(W.T, D)
    S0 = y[:, L==0]
    S1 = y[:, L==1]
    S2 = y[:, L==2]
    #plt.scatter(S0[0, :], S0[1, :], label = 'Setosa')
    #plt.scatter(S1[0, :], S1[1, :], label = 'Versicolor')
    #plt.scatter(S2[0, :], S2[1, :], label = 'Virginica')
    #plt.show()
    #Part related to classification
    D1 = D[:, L != 0]
    L1 = L[L != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D1, L1)
    muTR = mcol(DTR.mean(1))
    DTR1 = DTR[:,LTR==1]
    DTR2 = DTR[:,LTR==2]
    muTR1 = mcol(DTR1.mean(1))
    muTR2 = mcol(DTR2.mean(1))
    DCTR1= DTR1-muTR1
    DCTR2= DTR2-muTR2
    Sw= 1/DTR.shape[1]*((DCTR1 @ DCTR1.T)+(DCTR2 @ DCTR2.T))
    Sb= (1/DTR.shape[1])*(DTR1.shape[1]* ((muTR1-muTR) @ (muTR1-muTR).T) + DTR2.shape[1]* ((muTR2-muTR) @ (muTR2-muTR).T))
    #print(Sb)
    #print(Sw)
    sTR, UTR = scipy.linalg.eigh(Sb, Sw)
    WTR = UTR[:, ::-1][:, 0:1]
    yTR = numpy.dot(WTR.T, DTR)
    yVAL= numpy.dot(WTR.T, DVAL)
    disTR1 = yTR[:,LTR==1]
    disTR2 = yTR[:,LTR==2]
    disVAL1 = yVAL[:,LVAL==1]
    disVAL2 = yVAL[:,LVAL==2]
    #plt.figure()
    #plt.hist(disTR1[0, :], bins = 5, density = True, alpha = 0.4, label = 'Versicolor')
    #plt.hist(disTR2[0, :], bins = 5, density = True, alpha = 0.4, label = 'Virginica')
    #plt.legend()
    #plt.show()
    #plt.figure()
    #plt.hist(disVAL1[0, :], bins = 5, density = True, alpha = 0.4, label = 'Versicolor')
    #plt.hist(disVAL2[0, :], bins = 5, density = True, alpha = 0.4, label = 'Virginica')
    #plt.show()
    threshold = (yTR[0, LTR==1].mean() + yTR[0, LTR==2].mean()) / 2.0 #Projected samples have only 1 dimension
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[yVAL[0] >= threshold] = 2
    PVAL[yVAL[0] < threshold] = 1
    diff=0
    for i in range(len(PVAL)):
        if(PVAL[i]!=LVAL[i]):
            diff+=1
    err_rate=diff/len(PVAL)
    print(err_rate)
    #PCA before LDA


    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D1, L1)
    muTR = mcol(DTR.mean(1))
    DCTR = DTR - muTR
    CTR=1/(DCTR.shape[1])*numpy.dot(DCTR,DCTR.T)
    ypcaVAL=calculateProjection(CTR,DVAL,2)
    ypcaTR=calculateProjection(CTR,DTR,2)
    DTR1 = ypcaTR[:,LTR==1]
    DTR2 = ypcaTR[:,LTR==2]
    muTR1 = mcol(DTR1.mean(1))
    muTR2 = mcol(DTR2.mean(1))
    DCTR1= DTR1-muTR1
    DCTR2= DTR2-muTR2
    muTR = mcol(ypcaTR.mean(1))
    Sw= 1/ypcaTR.shape[1]*((DCTR1 @ DCTR1.T)+(DCTR2 @ DCTR2.T))
    Sb= (1/ypcaTR.shape[1])*(DTR1.shape[1]* ((muTR1-muTR) @ (muTR1-muTR).T) + DTR2.shape[1]* ((muTR2-muTR) @ (muTR2-muTR).T))
    sTR, UTR = scipy.linalg.eigh(Sb, Sw)
    WTR = -UTR[:, ::-1][:, 0:1]
    yTR = numpy.dot(WTR.T, ypcaTR)
    yVAL= numpy.dot(WTR.T, ypcaVAL)
    disTR1 = yTR[:,LTR==1]
    disTR2 = yTR[:,LTR==2]
    disVAL1 = yVAL[:,LVAL==1]
    disVAL2 = yVAL[:,LVAL==2]
    plt.figure()
    plt.hist(disTR1[0, :], bins = 5, density = True, alpha = 0.4, label = 'Versicolor')
    plt.hist(disTR2[0, :], bins = 5, density = True, alpha = 0.4, label = 'Virginica')
    plt.legend()
    #plt.show()
    threshold = (yTR[0, LTR==1].mean() + yTR[0, LTR==2].mean()) / 2.0 #Projected samples have only 1 dimension
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[yVAL[0] >= threshold] = 2
    PVAL[yVAL[0] < threshold] = 1
    
    diff=0
    for i in range(len(PVAL)):
        if(PVAL[i]!=LVAL[i]):
            diff+=1
    err_rate=diff/len(PVAL)
    print(err_rate)
   
    
    


