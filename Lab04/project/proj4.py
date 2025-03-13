import sys
import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import math

def mcol(vett):
    return vett.reshape((vett.size,1))
def mrow(v):
    return v.reshape((1,v.size))

def load(fileName):
    Dlist=[]
    L=[]
    with open(fileName) as f:
        for line in f:
            attrs = line.split(',')[0:-1]
            attrs = mcol(numpy.array([float(i) for i in attrs]))
            Dlist.append(attrs)
            label= line.split(',')[-1].strip()
            L.append(int(label))
    return numpy.hstack(Dlist),numpy.array(L)

def plot_hist(D, L):
    D0 = D[:, L==0] #counterfeit
    D1 = D[:, L==1] #genuine
    
    for dIdx in range(6):
        plt.figure()
        plt.xlabel(("Direction %d" % (dIdx+1)))
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Counterfeit')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Genuine')
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('proj_hist_%d.pdf' % dIdx)
    plt.show()


def calculateProjection(C,D,m):
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:,0:m]
    y = numpy.dot(P.T, D)
    return y



def logpdf_GAU_ND(X, mu, C):
    M = X.shape[0]
    N = X.shape[1]
    Y = numpy.zeros(N)  # Inizializza una matrice per i risultati
    print(M)
    print(N)
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

if __name__=='__main__':
    fileName=sys.argv[1]
    D,L=load(fileName)
    # Change default font size as in the lab solution
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    Dc0=D[:,L==0]
    Dc1=D[:,L==1]
    for i in range(D.shape[0]):
        Dx0=Dc0[i:i+1,:]
        Dx1=Dc1[i:i+1,:]
        m_ML0,C_ML0=calcMuAndSigma(Dx0)
        m_ML1,C_ML1=calcMuAndSigma(Dx1)
        plt.figure()
        plt.xlabel(("F%d" % (i+1)))
        plt.hist(Dx0.ravel(), bins=50, density=True,label = 'Counterfeit',alpha=0.5)
        plt.hist(Dx1.ravel(), bins=50, density=True,label = 'Genuine',alpha=0.5)
        XPlot = numpy.linspace(-8, 12, 1000)
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m_ML0, C_ML0)),label = 'Counterfeit',linewidth=2)
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m_ML1, C_ML1)),label = 'Genuine',linewidth=2)
        plt.legend()
        plt.savefig('gaussianFeature%d.pdf' % (i+1))
    plt.show()