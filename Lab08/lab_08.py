import numpy as numpy
from scipy.optimize import fmin_l_bfgs_b
import sklearn.datasets

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))
#takes a 1-D numpy array x of shape (2,) and returns the value of f(y,z)
def objective_function(x):
    y, z = x[0], x[1]
    return (y + 3)**2 + numpy.sin(y) + (z + 1)**2

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

def compute_min_normalized_dcf(llr,LTE,pi1, Cfn, Cfp):
    # Initialize minimum normalized DCF
    min_normalized_dcf = float('inf')

    # Iterate through all possible thresholds among scores
    thresholds = numpy.unique(llr)
    for threshold in thresholds:
        predicted_labels = numpy.where(llr <= threshold, 0, 1)
        DCF_normalized = compute_empirical_Bayes_risk_binary(predicted_labels, LTE, pi1, Cfn, Cfp)
        min_normalized_dcf = min(min_normalized_dcf, DCF_normalized)

    return min_normalized_dcf


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

def trainWeightedLogReg(DTR, LTR, l, pi_t):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        S = (vcol(w).T @ DTR + b).ravel()
        n = DTR.shape[1]
        ZTR = 2 * LTR - 1
        # Compute the weights for the positive and negative samples
        nt = numpy.sum(LTR == 1)
        nf = numpy.sum(LTR == 0)
        weights = numpy.where(LTR == 1, pi_t / nt, (1 - pi_t) / nf)
        G = -ZTR / (1.0 + numpy.exp(ZTR * S))
        weighted_G = weights * G
        deltaJ = numpy.sum(weighted_G)
        grad = l * w + numpy.sum((vrow(weighted_G) * DTR), axis=1)
        vgrad = numpy.hstack([grad, deltaJ])
        loss = (l / 2) * numpy.sum(w ** 2) + numpy.sum(weights * numpy.logaddexp(0, -ZTR * S))
        return (loss, vgrad)
    
    xf = fmin_l_bfgs_b(func=logreg_obj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=False)[0]
    return xf[:-1], xf[-1]


def trainLogReg(DTR, LTR, l):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        S = (vcol(w).T @ DTR + b).ravel()
        n= DTR.shape[1]
        ZTR = 2 * LTR- 1
        G= -ZTR / (1.0 + numpy.exp(ZTR * S))
        deltaJ= (1/n)*numpy.sum(G)
        grad = l*w + (1/n)* numpy.sum((vrow(G) * DTR),axis=1)
        vgrad = numpy.hstack([grad,deltaJ])
        loss = (l / 2) * numpy.sum(w ** 2) + (1 / n) * numpy.sum(numpy.logaddexp(0, -ZTR * S))
        return (loss,vgrad)
    xf = fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1), approx_grad=False)[0]
    return xf[:-1], xf[-1]

def computeEmpiricalPrior(LTR):
    return numpy.mean(LTR)


if __name__ == "__main__":
    x0 = numpy.array([0.0, 0.0])

    # Call the optimizer
    #result = fmin_l_bfgs_b(objective_function, x0, approx_grad=True)

    # Print the result
    #print("Optimal solution:", result[0])
    #print("Function value at the optimal solution:", result[1])
    #print("Information about the optimization process:", result[2])

    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    piEmp= computeEmpiricalPrior(LTR)
    pi_t=0.8
    for l in [0.001, 0.1, 1]:
        w, b = trainLogReg(DTR, LTR, l)
        S = (vcol(w).T @ DVAL + b).ravel() 
        Sllr = S - numpy.log(piEmp / (1 - piEmp))
        LP = (S > 0).astype(int)
        ErrorRate = (numpy.sum(LP != LVAL) / LVAL.size) * 100
        print("Error rate with l = %.3f:" %l, round(ErrorRate, 1))
        print("Actual DCF: %f" %compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr,0.5,1,1),LVAL,0.5,1,1))
        print("Min DCF: %f" %compute_min_normalized_dcf(Sllr,LVAL,0.5,1,1))
        print("Weighted version with a pi_t = %f " %pi_t)
        w, b = trainWeightedLogReg(DTR, LTR, l, pi_t)
        S = (vcol(w).T @ DVAL + b).ravel() 
        Sllr = S - numpy.log(pi_t / (1 - pi_t))
        LP = (S > 0).astype(int)
        ErrorRate = (numpy.sum(LP != LVAL) / LVAL.size) * 100
        print("Error rate with l = %.3f:" %l, round(ErrorRate, 1))
        print("Actual DCF: %f" %compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr,0.8,1,1),LVAL,0.8,1,1))
        print("Min DCF: %f" %compute_min_normalized_dcf(Sllr,LVAL,0.8,1,1))
        print("-"*50)

        
    
