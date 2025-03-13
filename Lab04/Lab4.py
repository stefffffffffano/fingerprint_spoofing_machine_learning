import sys
import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import math

def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((1,v.size))
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

def logpdf_GAU_ND(X, mu, C):
    M = X.shape[0]
    N = X.shape[1]
    Y = numpy.zeros(N)  # Inizializza una matrice per i risultati
    for i in range(N):
        x = X[:, i:i+1]
        pt = (-M / 2) * math.log(2 * math.pi)
        st = (-1 / 2) * numpy.linalg.slogdet(C)[1]
        tt = (-1 / 2) * (x - mu).T @ numpy.linalg.inv(C) @ (x - mu)
        Y[i] = pt + st + tt
    return Y



def calcMuAndSigma(mat):
    mu = mcol(mat.mean(1))
    matC = mat - mu
    C=1/(matC.shape[1])*numpy.dot(matC,matC.T)
    return (mu,C)    



if __name__ == '__main__':
    #C is MxM,x is M and mu is M as well
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    #plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m, C)))
    #plt.show() 

    #Maximum likelihood estimate
    #I have a vector x with N components
    XND = numpy.load('Solution/XND.npy')
    mu = numpy.load('Solution/muND.npy')
    C = numpy.load('Solution/CND.npy')
    pdfSol = numpy.load('Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())
    X1D=numpy.load('Solution/X1D.npy')
    m_ML,C_ML=calcMuAndSigma(X1D)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m_ML, C_ML)))
    plt.show()

   
    
    


