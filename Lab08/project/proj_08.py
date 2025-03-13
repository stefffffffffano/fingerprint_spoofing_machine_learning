import numpy as numpy
from scipy.optimize import fmin_l_bfgs_b
import sklearn.datasets
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

def quadratic_expansion(D):
    n_features, n_samples = D.shape
    expanded_features = []

    # Adding original features
    expanded_features.append(D)

    # Adding quadratic terms
    for i in range(n_features):
        for j in range(i, n_features):
            expanded_features.append(D[i, :] * D[j, :])

    return numpy.vstack(expanded_features)



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
    D, L = load('trainData.txt')
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    piEmp = computeEmpiricalPrior(LTR)
    lambdas = numpy.logspace(-4, 2, 13)
    
    actual_dcf_values = []
    min_dcf_values = []
    
    for l in lambdas:
        w, b = trainLogReg(DTR, LTR, l)
        S = (vcol(w).T @ DTE + b).ravel()
        Sllr = S - numpy.log(piEmp / (1 - piEmp))
        actual_dcf = compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr, 0.1, 1, 1), LTE, 0.1, 1, 1)
        min_dcf = compute_min_normalized_dcf(Sllr, LTE, 0.1, 1, 1)
        actual_dcf_values.append(actual_dcf)
        min_dcf_values.append(min_dcf)
    
    print("Actual DCF values (linear): ", actual_dcf_values)
    print("Min DCF values (linear): ", min_dcf_values)
    plt.figure()
    plt.plot(lambdas, actual_dcf_values, label="Actual DCF", marker='o')
    plt.plot(lambdas, min_dcf_values, label="Min DCF", marker='x')
    plt.xscale('log', base=10)
    plt.xlabel("Lambda")
    plt.ylabel("DCF")
    plt.title("DCF vs Lambda")
    plt.legend()
    plt.savefig('DCF_minDCF_nonWeighted_entireDataSet')
    
    DTR2 = DTR[:, ::50]
    LTR2 = LTR[::50]
    actual_dcf_values = []
    min_dcf_values = []
    for l in lambdas:
        w, b = trainLogReg(DTR2, LTR2, l)
        S = (vcol(w).T @ DTE + b).ravel()
        Sllr = S - numpy.log(piEmp / (1 - piEmp))
        actual_dcf = compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr, 0.1, 1, 1), LTE, 0.1, 1, 1)
        min_dcf = compute_min_normalized_dcf(Sllr, LTE, 0.1, 1, 1)
        actual_dcf_values.append(actual_dcf)
        min_dcf_values.append(min_dcf)
    
    print("Actual DCF values (linear_1outof50): ", actual_dcf_values)
    print("Min DCF values (linear_1outof50): ", min_dcf_values)
    plt.figure()
    plt.plot(lambdas, actual_dcf_values, label="Actual DCF", marker='o')
    plt.plot(lambdas, min_dcf_values, label="Min DCF", marker='x')
    plt.xscale('log', base=10)
    plt.xlabel("Lambda")
    plt.ylabel("DCF")
    plt.title("DCF vs Lambda")
    plt.legend()
    plt.savefig('DCF_minDCF_nonWeighted_1outof50DataSet')
    #weighted version evaluated on all samples
    actual_dcf_values = []
    min_dcf_values = []

    for l in lambdas:
        w, b = trainWeightedLogReg(DTR, LTR, l,0.1)
        S = (vcol(w).T @ DTE + b).ravel()
        Sllr = S - numpy.log(0.1 / (1 - 0.1))
        LP = (S > 0).astype(int)
        #ErrorRate = (numpy.sum(LP != LTE) / LTE.size) * 100 not needed
        actual_dcf = compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr, 0.1, 1, 1), LTE, 0.1, 1, 1)
        min_dcf = compute_min_normalized_dcf(Sllr, LTE, 0.1, 1, 1)
        actual_dcf_values.append(actual_dcf)
        min_dcf_values.append(min_dcf)

    print("Actual DCF values (weighted): ", actual_dcf_values)
    print("Min DCF values (weighted): ", min_dcf_values)   
    plt.figure()
    plt.plot(lambdas, actual_dcf_values, label="Actual DCF", marker='o')
    plt.plot(lambdas, min_dcf_values, label="Min DCF", marker='x')
    plt.xscale('log', base=10)
    plt.xlabel("Lambda")
    plt.ylabel("DCF")
    plt.title("DCF vs Lambda weighted version with pi_t = 0.1")
    plt.legend()
    plt.savefig('DCF_minDCF_Weighted_entireDataSet')
   
    DTR_quad = quadratic_expansion(DTR)
    DTE_quad = quadratic_expansion(DTE)
    lambdas = numpy.logspace(-4, 2, 13)
    actual_dcf_values = []
    min_dcf_values = []

    for l in lambdas:
        w, b = trainLogReg(DTR_quad, LTR, l)
        S = (vcol(w).T @ DTE_quad + b).ravel()
        Sllr = S - numpy.log(piEmp / (1 - piEmp))
        actual_dcf = compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr, 0.1, 1, 1), LTE, 0.1, 1, 1)
        min_dcf = compute_min_normalized_dcf(Sllr, LTE, 0.1, 1, 1)
        actual_dcf_values.append(actual_dcf)
        min_dcf_values.append(min_dcf)

    print("Actual DCF values (quadratic): ", actual_dcf_values)
    print("Min DCF values (quadratic): ", min_dcf_values)
    plt.figure()
    plt.plot(lambdas, actual_dcf_values, label="Actual DCF", marker='o')
    plt.plot(lambdas, min_dcf_values, label="Min DCF", marker='x')
    plt.xscale('log', base=10)
    plt.xlabel("Lambda")
    plt.ylabel("DCF")
    plt.title("Quadratic Logistic Regression: DCF vs Lambda")
    plt.legend()
    plt.savefig('DCF_minDCF_QuadraticLogReg_entireDataSet')
    
    #We need to center the data with respect to the dataset mean and then reapply the standard linear LR model
    #on centered data (full dataset)
    mean = vcol(numpy.mean(DTR, axis=1))
    DTR_centered = DTR - mean
    DTE_centered = DTE - mean
    actual_dcf_values = []
    min_dcf_values = []

    for l in lambdas:
        w, b = trainLogReg(DTR_centered, LTR, l)
        S = (vcol(w).T @ DTE_centered + b).ravel()
        Sllr = S - numpy.log(piEmp / (1 - piEmp))
        actual_dcf = compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr, 0.1, 1, 1), LTE, 0.1, 1, 1)
        min_dcf = compute_min_normalized_dcf(Sllr, LTE, 0.1, 1, 1)
        actual_dcf_values.append(actual_dcf)
        min_dcf_values.append(min_dcf)
    print("Actual DCF values (centered): ", actual_dcf_values)
    print("Min DCF values (centered): ", min_dcf_values)
    plt.figure()
    plt.plot(lambdas, actual_dcf_values, label="Actual DCF", marker='o')
    plt.plot(lambdas, min_dcf_values, label="Min DCF", marker='x')
    plt.xscale('log', base=10)
    plt.xlabel("Lambda")
    plt.ylabel("DCF")
    plt.title("Centered data: DCF vs Lambda")
    plt.legend()
    plt.savefig('DCF_minDCF_centeredData_entireDataSet')

