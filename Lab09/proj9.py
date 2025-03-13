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

# Optimize SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples

import bayesRisk

    
    
    


if __name__ == '__main__':
    D, L = load('trainData.txt')
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    dcf_values = []
    min_dcf_values = []
    # Calcola il prior effettivo corrispondente a ciascun valore di log-odds del prior
    K=1.0
    for C in numpy.logspace(-5, 0, 11):
        w, b = train_dual_SVM_linear(DTR, LTR, C, K)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        dcf = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_dcf = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min_dcf)
        print ('actDCF - pT = 0.1: %.4f' % dcf)
        dcf_values.append(dcf)
        min_dcf_values.append(min_dcf)
        print ()
    
    plt.figure()
    plt.xscale('log')
    plt.plot(numpy.logspace(-5,0,11), dcf_values, label='actDCF', color='r',marker='o')
    plt.plot(numpy.logspace(-5,0,11), min_dcf_values, label='min DCF', color='b',marker='x')
    plt.ylim([0, 1.1])
    plt.title('DCF and minDCF vs C')
    plt.legend()
    plt.grid(True)
    #I don'twant to plot it, but I want to save it
    plt.savefig('DCF_minDCF_linear')


    #Then we have to repeat the analysis for centered data...
    mean = vcol(DTR.mean(1))
    DTRcentered = DTR - mean
    DVALcentered = DVAL -mean
    dcf_values = []
    min_dcf_values = []
    # Calcola il prior effettivo corrispondente a ciascun valore di log-odds del prior
    K=1.0
    for C in numpy.logspace(-5, 0, 11):
        w, b = train_dual_SVM_linear(DTRcentered, LTR, C, K)
        SVAL = (vrow(w) @ DVALcentered + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        dcf = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_dcf = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min_dcf)
        print ('actDCF - pT = 0.1: %.4f' % dcf)
        dcf_values.append(dcf)
        min_dcf_values.append(min_dcf)
        print ()
    
    plt.figure()
    plt.xscale('log')
    plt.plot(numpy.logspace(-5,0,11), dcf_values, label='actDCF', color='r',marker='o')
    plt.plot(numpy.logspace(-5,0,11), min_dcf_values, label='min DCF', color='b',marker='x')
    plt.ylim([0, 1.1])
    plt.title('DCF and minDCF vs C (centered data)')
    plt.legend()
    plt.grid(True)
    #I don'twant to plot it, but I want to save it
    plt.savefig('DCF_minDCF_linear_centeredData')    
    

    #We consider the plynomial kernel with d=2,c=1, eps=0
    dcf_values = []
    min_dcf_values = []

    for C in numpy.logspace(-5, 0, 11):
        fScore = train_dual_SVM_kernel(DTR, LTR, C, polyKernel(2,1), 0)
        SVAL = fScore(DVAL)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate (polynomial kernel): %.1f' % (err*100))
        dcf = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_dcf = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min_dcf)
        print ('actDCF - pT = 0.1: %.4f' % dcf)
        dcf_values.append(dcf)
        min_dcf_values.append(min_dcf)
        print ()
    
    plt.figure()
    plt.xscale('log')
    plt.plot(numpy.logspace(-5,0,11), dcf_values, label='actDCF', color='r',marker='o')
    plt.plot(numpy.logspace(-5,0,11), min_dcf_values, label='min DCF', color='b',marker='x')
    plt.ylim([0, 1.1])
    plt.title('DCF and minDCF vs C (polynomial kernel)')
    plt.legend()
    plt.grid(True)
    #I don'twant to plot it, but I want to save it
    plt.savefig('DCF_minDCF_polynomialKernel')  


    #We now consider the RBF kernel with eps = 1
    #gamma = [e^(-4), e^(-3), e^(-2), e^(-1)]
    gamma_values = [math.exp(-4), math.exp(-3), math.exp(-2), math.exp(-1)]
    C_values = numpy.logspace(-3, 2, 11)
    # Liste per memorizzare i risultati
    minDCF_values = {g: [] for g in gamma_values}
    actDCF_values = {g: [] for g in gamma_values}

    for g in gamma_values:
        for C in C_values:
            fScore = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(g), 1)
            SVAL = fScore(DVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            print(f'Error rate: {err*100:.1f}% for C={C}, gamma={g}')
            dcf = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            min_dcf = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            minDCF_values[g].append(min_dcf)
            actDCF_values[g].append(dcf)
            print(f'minDCF - pT = 0.1: {min_dcf:.4f}')
            print(f'actDCF - pT = 0.1: {dcf:.4f}')
            print()

    plt.figure()
    plt.xscale('log')
    colors = ['b', 'g', 'r', 'c']   
    for i, g in enumerate(gamma_values):
        plt.plot(C_values, actDCF_values[g], label=f'actDCF - gamma={g:.4e}', color=colors[i], marker='o')
        plt.plot(C_values, minDCF_values[g], label=f'minDCF - gamma={g:.4e}', color=colors[i], marker='x')

    plt.ylim([0, 1.1])
    plt.title('DCF and minDCF vs C (RBK kernel)')
    plt.legend()
    plt.grid(True)
    #I don'twant to plot it, but I want to save it
    plt.savefig('DCF_minDCF_RBFKernel')  

        