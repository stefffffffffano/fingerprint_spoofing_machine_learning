from lab_10_profSol import *

import bayesRisk
from proj_08 import *
from proj9 import *

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



if __name__ == '__main__':

    D, L = load('trainData.txt')
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    """
    for covType in ['full', 'diagonal']:
        for numCclass0 in [1,2,4,8,16,32]:
            for numCclass1 in [1,2,4,8,16,32]:
                gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numCclass0, covType = covType, verbose=False, psiEig = 0.01)
                gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numCclass1, covType = covType, verbose=False, psiEig = 0.01)
                SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
                print("%d components for class 0, %d components for class 1. CovType: %s" % (numCclass0, numCclass1,covType))
                #we print minDCF/actDCF
                print ('%.4f / %.4f' % (bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0), bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0)))
                print()
    """
    #train again the most promising models
    #First the most promising LR model
    DTR_quad = quadratic_expansion(DTR)
    DTE_quad = quadratic_expansion(DVAL)
    piEmp = computeEmpiricalPrior(LTR)
    l= 3.16227766e-02
    w, b = trainLogReg(DTR_quad, LTR, l)
    S = (vcol(w).T @ DTE_quad + b).ravel()
    LLRLR = S - numpy.log(piEmp / (1 - piEmp))

    #Then, the most promising SVM
    C = 32.622776601683793
    g = 0.1353352832366127 
    fScore = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(g), 1)
    LLRSVM = fScore(DVAL)

    #Finally, the most promising GMM
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, covType = 'diagonal', verbose=False, psiEig = 0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, covType = 'diagonal', verbose=False, psiEig = 0.01)
    LLRGMM = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    # Bayes error plot
    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCFLR = []
    minDCFLR = []
    actDCFSVM = []
    minDCFSVM = []
    actDCFGMM = []
    minDCFGMM = []
    for effPrior in effPriors:
        minDCFLR.append(bayesRisk.compute_minDCF_binary_fast(LLRLR, LVAL, effPrior, 1.0, 1.0))
        actDCFLR.append(bayesRisk.compute_actDCF_binary_fast(LLRLR, LVAL, effPrior, 1.0, 1.0))
        minDCFSVM.append(bayesRisk.compute_minDCF_binary_fast(LLRSVM, LVAL, effPrior, 1.0, 1.0))
        actDCFSVM.append(bayesRisk.compute_actDCF_binary_fast(LLRSVM, LVAL, effPrior, 1.0, 1.0))
        minDCFGMM.append(bayesRisk.compute_minDCF_binary_fast(LLRGMM, LVAL, effPrior, 1.0, 1.0))
        actDCFGMM.append(bayesRisk.compute_actDCF_binary_fast(LLRGMM, LVAL, effPrior, 1.0, 1.0))
        
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(effPriorLogOdds, actDCFLR, label='actDCF LR', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCFLR, label='minDCF LR', color='b')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCFSVM, label='minDCF SVM', color='g')
    matplotlib.pyplot.plot(effPriorLogOdds, actDCFSVM, label='actDCF SVM', color='c')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCFGMM, label='minDCF GMM', color='m')
    matplotlib.pyplot.plot(effPriorLogOdds, actDCFGMM, label='actDCF GMM', color='y')
    matplotlib.pyplot.ylim([0, 1.1])
    #matplotlib.pyplot.show()
    plt.legend()
    plt.savefig('classifiers_comparison')        
    


    