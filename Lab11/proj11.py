import numpy
import bayesRisk
import logReg
import matplotlib.pyplot as plt

from lab_10_profSol import *
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

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    
    effPriorLogOdds = numpy.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

   
def extract_train_val_folds_from_ary(X, idx):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def shuffle_scores(S, L):
    # Genera una permutazione casuale degli indici degli score
    indices = numpy.random.permutation(S.size)
    # Applica la permutazione agli score e alle etichette
    return S[indices], L[indices]

def shuffle_multiple_scores(scores, labels):
    # Genera una permutazione casuale degli indici
    indices = numpy.random.permutation(labels.size)
    
    # Applica la permutazione agli score
    shuffled_scores = [score[indices] for score in scores]
    # Applica la permutazione alle etichette
    shuffled_labels = labels[indices]
    
    return shuffled_scores, shuffled_labels




if __name__ == '__main__':
    #we now consider the best performing system among the three we selected in the previous lab
    D, L = load('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    #The best model was GMM with 16 components for class 1, 1 for class 0 and diagonal covariance matrices
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, covType = 'diagonal', verbose=False, psiEig = 0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 16, covType = 'diagonal', verbose=False, psiEig = 0.01)
    LLRGMM = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0) #these are the scores on the evaluation set
    #the most promising LR model
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
    KFOLD = 5 #we set the number of folds to 5 as done in the laboratory
 
    #First we evaluate K-fold calibration on GMM scores
    
    LLRGMM, LVALGMM = shuffle_scores(LLRGMM, LVAL)
    print('GMM:')
    print()
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        calibrated_scores = [] 
        labels = [] 
        for foldIdx in range(KFOLD):
            SCAL, SVAL = extract_train_val_folds_from_ary(LLRGMM, foldIdx)
            LCAL, LVAL2 = extract_train_val_folds_from_ary(LVALGMM, foldIdx)
            w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
            calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
            calibrated_scores.append(calibrated_SVAL)
            labels.append(LVAL2)
        
        calibrated_scores = numpy.hstack(calibrated_scores)
        labels = numpy.hstack(labels)
        #evaluation always performed on the target application (0.1,1,1)
        print ('\t\tminDCF(p=%f), cal.   : %.3f' % (pT,bayesRisk.compute_minDCF_binary_fast(calibrated_scores, labels, 0.1, 1.0, 1.0))) 
        print ('\t\tactDCF(p=%f), cal.   : %.3f' % (pT,bayesRisk.compute_actDCF_binary_fast(calibrated_scores, labels, 0.1, 1.0, 1.0)))
    #At this point, we should select the one with the lowest actualDCF
    
    LLRLR, LVALLR = shuffle_scores(LLRLR, LVAL)
    print('LR:')
    print()
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        calibrated_scores = [] 
        labels = [] 
        for foldIdx in range(KFOLD):
            SCAL, SVAL = extract_train_val_folds_from_ary(LLRLR, foldIdx)
            LCAL, LVAL2 = extract_train_val_folds_from_ary(LVALLR, foldIdx)
            w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
            calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
            calibrated_scores.append(calibrated_SVAL)
            labels.append(LVAL2)
        
        calibrated_scores = numpy.hstack(calibrated_scores)
        labels = numpy.hstack(labels)
        #evaluation always performed on the target application (0.1,1,1)
        print ('\t\tminDCF(p=%f), cal.   : %.3f' % (pT,bayesRisk.compute_minDCF_binary_fast(calibrated_scores, labels, 0.1, 1.0, 1.0))) 
        print ('\t\tactDCF(p=%f), cal.   : %.3f' % (pT,bayesRisk.compute_actDCF_binary_fast(calibrated_scores, labels, 0.1, 1.0, 1.0)))
        
    LLRSVM, LVALSVM = shuffle_scores(LLRSVM, LVAL)
    print('SVM:')
    print()
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        calibrated_scores = [] 
        labels = [] 
        for foldIdx in range(KFOLD):
            SCAL, SVAL = extract_train_val_folds_from_ary(LLRSVM, foldIdx)
            LCAL, LVAL2 = extract_train_val_folds_from_ary(LVALSVM, foldIdx)
            w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
            calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
            calibrated_scores.append(calibrated_SVAL)
            labels.append(LVAL2)
        
        calibrated_scores = numpy.hstack(calibrated_scores)
        labels = numpy.hstack(labels)
        #evaluation always performed on the target application (0.1,1,1)
        print ('\t\tminDCF(p=%f), cal.   : %.3f' % (pT,bayesRisk.compute_minDCF_binary_fast(calibrated_scores, labels, 0.1, 1.0, 1.0))) 
        print ('\t\tactDCF(p=%f), cal.   : %.3f' % (pT,bayesRisk.compute_actDCF_binary_fast(calibrated_scores, labels, 0.1, 1.0, 1.0)))
    
    #We have selected pi_T=0.9 for GMM, 0.4 for LR and 0.1 for SVM
    #first GMM
    pT=0.9
    w, b = logReg.trainWeightedLogRegBinary(vrow(LLRGMM), LVAL, 0, pT)
    calibrated_SVAL =  (w.T @ vrow(LLRGMM) + b - numpy.log(pT / (1-pT))).ravel()
    #Then, LR
    pT=0.4
    w, b = logReg.trainWeightedLogRegBinary(vrow(LLRLR), LVAL, 0, pT)
    calibrated_SVAL2 =  (w.T @ vrow(LLRLR) + b - numpy.log(pT / (1-pT))).ravel()
    #Finally, SVM
    pT=0.1
    w, b = logReg.trainWeightedLogRegBinary(vrow(LLRSVM), LVAL, 0, pT)
    calibrated_SVAL3 =  (w.T @ vrow(LLRSVM) + b - numpy.log(pT / (1-pT))).ravel()
    #We now plot the results using Bayes error plots
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
        minDCFLR.append(bayesRisk.compute_minDCF_binary_fast(calibrated_SVAL2, LVAL, effPrior, 1.0, 1.0))
        actDCFLR.append(bayesRisk.compute_actDCF_binary_fast(calibrated_SVAL2, LVAL, effPrior, 1.0, 1.0))
        minDCFSVM.append(bayesRisk.compute_minDCF_binary_fast(calibrated_SVAL3, LVAL, effPrior, 1.0, 1.0))
        actDCFSVM.append(bayesRisk.compute_actDCF_binary_fast(calibrated_SVAL3, LVAL, effPrior, 1.0, 1.0))
        minDCFGMM.append(bayesRisk.compute_minDCF_binary_fast(calibrated_SVAL, LVAL, effPrior, 1.0, 1.0))
        actDCFGMM.append(bayesRisk.compute_actDCF_binary_fast(calibrated_SVAL, LVAL, effPrior, 1.0, 1.0))
        
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
    plt.title('Bayes error plot, calibrated scores')
    plt.savefig('classifiers_comparison_CalibratedScores') 
    
    # Fusion #
    shuffled_scores, LVAL = shuffle_multiple_scores([calibrated_SVAL, calibrated_SVAL2, calibrated_SVAL3], LVAL)
    LLRGMM = shuffled_scores[0]
    LLRLR = shuffled_scores[1]
    LLRSVM = shuffled_scores[2]
    #GMM+LR
    print ('Fusion GMM + quadratic LR')
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            fusedScores = [] # We will add to the list the scores computed for each fold
            fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
            # Train KFOLD times the fusion model
            for foldIdx in range(KFOLD):
                # keep 1 fold for validation, use the remaining ones for training        
                SCAL1, SVAL1 = extract_train_val_folds_from_ary(LLRGMM, foldIdx)
                SCAL2, SVAL2 = extract_train_val_folds_from_ary(LLRLR, foldIdx)
                LCAL, LVAL2 = extract_train_val_folds_from_ary(LVAL, foldIdx)
                # Build the training scores "feature" matrix
                SCAL = numpy.vstack([SCAL1, SCAL2])
                # Train the model on the KFOLD - 1 training folds
                w, b = logReg.trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
                # Build the validation scores "feature" matrix
                SVAL = numpy.vstack([SVAL1, SVAL2])
                # Apply the model to the validation fold
                calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()
                # Add the scores of this validation fold to the cores list
                fusedScores.append(calibrated_SVAL)
                # Add the corresponding labels to preserve alignment between scores and labels
                fusedLabels.append(LVAL2)

            fusedScores = numpy.hstack(fusedScores)
            fusedLabels = numpy.hstack(fusedLabels)
            print ('\t\tminDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0))) 
            print ('\t\tactDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0)))
        
    #GMM+SVM
    print ('Fusion GMM + SVM')
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            fusedScores = [] # We will add to the list the scores computed for each fold
            fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
            
            # Train KFOLD times the fusion model
            for foldIdx in range(KFOLD):
                # keep 1 fold for validation, use the remaining ones for training        
                SCAL1, SVAL1 = extract_train_val_folds_from_ary(LLRGMM, foldIdx)
                SCAL2, SVAL2 = extract_train_val_folds_from_ary(LLRSVM, foldIdx)
                LCAL, LVAL2 = extract_train_val_folds_from_ary(LVAL, foldIdx)
                # Build the training scores "feature" matrix
                SCAL = numpy.vstack([SCAL1, SCAL2])
                # Train the model on the KFOLD - 1 training folds
                w, b = logReg.trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
                # Build the validation scores "feature" matrix
                SVAL = numpy.vstack([SVAL1, SVAL2])
                # Apply the model to the validation fold
                calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()
                # Add the scores of this validation fold to the cores list
                fusedScores.append(calibrated_SVAL)
                # Add the corresponding labels to preserve alignment between scores and labels
                fusedLabels.append(LVAL2)

            fusedScores = numpy.hstack(fusedScores)
            fusedLabels = numpy.hstack(fusedLabels)
            print ('\t\tminDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0))) 
            print ('\t\tactDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0)))
        

    #LR+SVM
    print ('Fusion SVM + quadratic LR')
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            fusedScores = [] # We will add to the list the scores computed for each fold
            fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
            
            # Train KFOLD times the fusion model
            for foldIdx in range(KFOLD):
                # keep 1 fold for validation, use the remaining ones for training        
                SCAL1, SVAL1 = extract_train_val_folds_from_ary(LLRSVM, foldIdx)
                SCAL2, SVAL2 = extract_train_val_folds_from_ary(LLRLR, foldIdx)
                LCAL, LVAL2 = extract_train_val_folds_from_ary(LVAL, foldIdx)
                # Build the training scores "feature" matrix
                SCAL = numpy.vstack([SCAL1, SCAL2])
                # Train the model on the KFOLD - 1 training folds
                w, b = logReg.trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
                # Build the validation scores "feature" matrix
                SVAL = numpy.vstack([SVAL1, SVAL2])
                # Apply the model to the validation fold
                calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()
                # Add the scores of this validation fold to the cores list
                fusedScores.append(calibrated_SVAL)
                # Add the corresponding labels to preserve alignment between scores and labels
                fusedLabels.append(LVAL2)

            fusedScores = numpy.hstack(fusedScores)
            fusedLabels = numpy.hstack(fusedLabels)
            print ('\t\tminDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0))) 
            print ('\t\tactDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0)))

    #GMM+LR+SVM
    print ('Fusion SVM + quadratic LR + GMM')
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            fusedScores = [] # We will add to the list the scores computed for each fold
            fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
            
            # Train KFOLD times the fusion model
            for foldIdx in range(KFOLD):
                # keep 1 fold for validation, use the remaining ones for training        
                SCAL1, SVAL1 = extract_train_val_folds_from_ary(LLRSVM, foldIdx)
                SCAL2, SVAL2 = extract_train_val_folds_from_ary(LLRLR, foldIdx)
                SCAL3, SVAL3 = extract_train_val_folds_from_ary(LLRGMM, foldIdx)
                LCAL, LVAL2 = extract_train_val_folds_from_ary(LVAL, foldIdx)
                # Build the training scores "feature" matrix
                SCAL = numpy.vstack([SCAL1, SCAL2,SCAL3])
                # Train the model on the KFOLD - 1 training folds
                w, b = logReg.trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
                # Build the validation scores "feature" matrix
                SVAL = numpy.vstack([SVAL1, SVAL2,SVAL3])
                # Apply the model to the validation fold
                calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()
                # Add the scores of this validation fold to the cores list
                fusedScores.append(calibrated_SVAL)
                # Add the corresponding labels to preserve alignment between scores and labels
                fusedLabels.append(LVAL2)

            fusedScores = numpy.hstack(fusedScores)
            fusedLabels = numpy.hstack(fusedLabels)
            print ('\t\tminDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0))) 
            print ('\t\tactDCF(p=%f)         : %.3f' % (pT,bayesRisk.compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0)))
    #The best configuration is obteined for GMM+SVM with pT=0.4 for score calibration    
    
    
    D,L = load('evalData.txt')
    #D contains the evaluation set, apply GMM+quadratic LR fusion

    
    LLRGMM_eval = logpdf_GMM(D, gmm1) - logpdf_GMM(D, gmm0)
    D_quad = quadratic_expansion(D)
    DTR_quad = quadratic_expansion(DTR)
    piEmp = computeEmpiricalPrior(LTR)
    l= 3.16227766e-02
    w, b = trainLogReg(DTR_quad, LTR, l)
    S = (vcol(w).T @ D_quad + b).ravel()
    LLRLR_eval = S - numpy.log(piEmp / (1 - piEmp))
    # Fusione dei punteggi
    pT = 0.9  # valore di pT scelto
    w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRGMM_eval), vrow(LLRLR_eval)]), L, 0, pT)
    fused_scores = (w.T @ numpy.vstack([vrow(LLRGMM_eval), vrow(LLRLR_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
    # Valutazione dei punteggi fusi
    min_dcf_fusion = bayesRisk.compute_minDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)
    act_dcf_fusion = bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)

    print(f'Fused model minDCF: {min_dcf_fusion:.3f}')
    print(f'Fused model actDCF: {act_dcf_fusion:.3f}')
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF=[]
    minDCF=[]
    for effPrior in effPriors:
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(fused_scores, L, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(fused_scores, L, effPrior, 1.0, 1.0))
        
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF ', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF ', color='b')
    matplotlib.pyplot.ylim([0, 1.1])
    #matplotlib.pyplot.show()
    plt.legend()
    plt.title('Bayes error plot, chosen model')
    plt.savefig('classifiers_comparison_fusedModel') 
    
    
    #Now I have to consider the three best models and their fusion, assessing their performance on the dataset
    #First, GMM
    LLRGMM_eval = logpdf_GMM(D, gmm1) - logpdf_GMM(D, gmm0)
    #perform calibration with pi_T=0.9 (best value for act_dcf evaluated with K-fold)
    pT=0.9
    w, b = logReg.trainWeightedLogRegBinary(vrow(LLRGMM_eval), L, 0, pT)
    calibrated_S =  (w.T @ vrow(LLRGMM_eval) + b - numpy.log(pT / (1-pT))).ravel()
    minDCFGMM = bayesRisk.compute_minDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)
    actDCFGMM = bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)
    print(f'GMM model minDCF: {minDCFGMM:.3f}')
    print(f'GMM model actDCF: {actDCFGMM:.3f}')
    #Then, LR
    DTR_quad = quadratic_expansion(DTR)
    piEmp = computeEmpiricalPrior(LTR)
    l= 3.16227766e-02
    w, b = trainLogReg(DTR_quad, LTR, l)
    D_quad = quadratic_expansion(D)
    S = (vcol(w).T @ D_quad + b).ravel()
    LLRLR_eval = S - numpy.log(piEmp / (1 - piEmp))
    #perform calibration with pi_T=0.4 (best value for act_dcf evaluated with K-fold)
    pT=0.4
    w, b = logReg.trainWeightedLogRegBinary(vrow(LLRLR_eval), L, 0, pT)
    calibrated_S =  (w.T @ vrow(LLRLR_eval) + b - numpy.log(pT / (1-pT))).ravel()
    minDCFLR = bayesRisk.compute_minDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)
    actDCFLR = bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)
    print(f'LR model minDCF: {minDCFLR:.3f}')
    print(f'LR model actDCF: {actDCFLR:.3f}')
    #Finally, SVM
    LLRSVM_eval = fScore(D)
    #perform calibration with pi_T=0.1 (best value for act_dcf evaluated with K-fold)
    pT=0.1
    w, b = logReg.trainWeightedLogRegBinary(vrow(LLRSVM_eval), L, 0, pT)
    calibrated_S =  (w.T @ vrow(LLRSVM_eval) + b - numpy.log(pT / (1-pT))).ravel()
    minDCFSVM = bayesRisk.compute_minDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)
    actDCFSVM = bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)
    print(f'SVM model minDCF: {minDCFSVM:.3f}')
    print(f'SVM model actDCF: {actDCFSVM:.3f}')
    #Now, fusion of GMM and SVM
    pT=0.9
    w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval)]), L, 0, pT)
    fused_scores = (w.T @ numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
    # Valutazione dei punteggi fusi
    min_dcf_fusion_GMMSVM = bayesRisk.compute_minDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)
    act_dcf_fusion_GMMSVM = bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)
    print(f'Fusion SVM and GMM model minDCF: {min_dcf_fusion_GMMSVM:.3f}')
    print(f'Fusion of SVM and GMM model actDCF: {act_dcf_fusion_GMMSVM:.3f}')
    #Now, fusion of SVM and LR
    pT=0.8
    w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRSVM_eval), vrow(LLRLR_eval)]), L, 0, pT)
    fused_scores = (w.T @ numpy.vstack([vrow(LLRSVM_eval), vrow(LLRLR_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
    min_dcf_fusion_SVMLR = bayesRisk.compute_minDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)
    act_dcf_fusion_SVMLR = bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)
    print(f'Fusion of SVM and LR model minDCF: {min_dcf_fusion_SVMLR:.3f}')
    print(f'Fusion of SVM and LR model actDCF: {act_dcf_fusion_SVMLR:.3f}')
    #Now, fusion of GMM, LR and SVM, pT=0.9
    pT=0.9
    w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval), vrow(LLRLR_eval)]), L, 0, pT)
    fused_scores = (w.T @ numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval), vrow(LLRLR_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
    min_dcf_fusion_GMMSVMLR = bayesRisk.compute_minDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)
    act_dcf_fusion_GMMSVMLR = bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)
    print(f'Fusion of GMM,LR and SVM model minDCF: {min_dcf_fusion_GMMSVMLR:.3f}')
    print(f'Fusion of GMM,LR and SVM model actDCF: {act_dcf_fusion_GMMSVMLR:.3f}')
   
    #DCF error plot
    actDCFGMM = []
    actDCFSVM = []
    actDCFLR = []
    actDCFGMMSVM = []
    actDCFSVMLR = []
    actDCFGMMSVMLR = []
    actDCFGMMLR = []
    for pT in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        #GMM+LR
        LLRGMM_eval = logpdf_GMM(D, gmm1) - logpdf_GMM(D, gmm0)
        D_quad = quadratic_expansion(D)
        DTR_quad = quadratic_expansion(DTR)
        piEmp = computeEmpiricalPrior(LTR)
        l= 3.16227766e-02
        w, b = trainLogReg(DTR_quad, LTR, l)
        S = (vcol(w).T @ D_quad + b).ravel()
        LLRLR_eval = S - numpy.log(piEmp / (1 - piEmp))
        w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRGMM_eval), vrow(LLRLR_eval)]), L, 0, pT)
        fused_scores = (w.T @ numpy.vstack([vrow(LLRGMM_eval), vrow(LLRLR_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
        # Valutazione dei punteggi fusi
        actDCFGMMLR.append(bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0))
        #GMM
        LLRGMM_eval = logpdf_GMM(D, gmm1) - logpdf_GMM(D, gmm0)
        w, b = logReg.trainWeightedLogRegBinary(vrow(LLRGMM_eval), L, 0, pT)
        calibrated_S =  (w.T @ vrow(LLRGMM_eval) + b - numpy.log(pT / (1-pT))).ravel()
        actDCFGMM.append(bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)) 
        #LR
        DTR_quad = quadratic_expansion(DTR)
        piEmp = computeEmpiricalPrior(LTR)
        l= 3.16227766e-02
        w, b = trainLogReg(DTR_quad, LTR, l)
        D_quad = quadratic_expansion(D)
        S = (vcol(w).T @ D_quad + b).ravel()
        LLRLR_eval = S - numpy.log(piEmp / (1 - piEmp))
        w, b = logReg.trainWeightedLogRegBinary(vrow(LLRLR_eval), L, 0, pT)
        calibrated_S =  (w.T @ vrow(LLRLR_eval) + b - numpy.log(pT / (1-pT))).ravel()
        actDCFLR.append(bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0)) 
        #SVM
        LLRSVM_eval = fScore(D)
        w, b = logReg.trainWeightedLogRegBinary(vrow(LLRSVM_eval), L, 0, pT)
        calibrated_S =  (w.T @ vrow(LLRSVM_eval) + b - numpy.log(pT / (1-pT))).ravel()
        actDCFSVM.append(bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, 0.1, 1.0, 1.0))
        #GMM+SVM
        w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval)]), L, 0, pT)
        fused_scores = (w.T @ numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
        actDCFGMMSVM.append(bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0)) 
        #SVM+LR
        w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRSVM_eval), vrow(LLRLR_eval)]), L, 0, pT)
        fused_scores = (w.T @ numpy.vstack([vrow(LLRSVM_eval), vrow(LLRLR_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
        actDCFSVMLR.append(bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0))
        #GMM+SVM+LR
        w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval), vrow(LLRLR_eval)]), L, 0, pT)
        fused_scores = (w.T @ numpy.vstack([vrow(LLRGMM_eval), vrow(LLRSVM_eval), vrow(LLRLR_eval)]) + b - numpy.log(pT / (1 - pT))).ravel()
        actDCFGMMSVMLR.append(bayesRisk.compute_actDCF_binary_fast(fused_scores, L, 0.1, 1.0, 1.0))
    p_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # Creazione del plot
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, actDCFGMM, marker='o', label='GMM')
    plt.plot(p_values, actDCFLR, marker='s', label='LR')
    plt.plot(p_values, actDCFSVM, marker='^', label='SVM')
    plt.plot(p_values, actDCFGMMSVM, marker='d', label='Fusion GMM + SVM')
    plt.plot(p_values, actDCFSVMLR, marker='x', label='Fusion SVM + LR')
    plt.plot(p_values, actDCFGMMSVMLR, marker='*', label='Fusion GMM + SVM + LR')
    plt.plot(p_values, actDCFGMMLR, marker='+', label='Fusion GMM + LR')

    plt.xlabel('pT')
    plt.ylabel('Actual DCF (actDCF)')
    plt.title('DCF Error Plot for Different Models and Fusions')
    plt.legend()
    plt.grid(True)
    plt.savefig('DCFerrorPlot')
    #Bayes error plot for the three best models (without considering their fusion)
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCFGMM = []
    minDCFGMM = []
    actDCFSVM = []
    minDCFSVM = []
    actDCFLR = []
    minDCFLR = []
    for effPrior in effPriors:
        #GMM
        pT=0.9
        w, b = logReg.trainWeightedLogRegBinary(vrow(LLRGMM_eval), L, 0, pT)
        calibrated_S =  (w.T @ vrow(LLRGMM_eval) + b - numpy.log(pT / (1-pT))).ravel()
        minDCFGMM.append(bayesRisk.compute_minDCF_binary_fast(calibrated_S, L, effPrior, 1.0, 1.0))
        actDCFGMM.append(bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, effPrior, 1.0, 1.0))
        #LR
        pT=0.4
        w, b = logReg.trainWeightedLogRegBinary(vrow(LLRLR_eval), L, 0, pT)
        calibrated_S =  (w.T @ vrow(LLRLR_eval) + b - numpy.log(pT / (1-pT))).ravel()
        minDCFLR.append(bayesRisk.compute_minDCF_binary_fast(calibrated_S, L, effPrior, 1.0, 1.0))
        actDCFLR.append(bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, effPrior, 1.0, 1.0))
        #SVM
        pT=0.1
        w, b = logReg.trainWeightedLogRegBinary(vrow(LLRSVM_eval), L, 0, pT)
        calibrated_S =  (w.T @ vrow(LLRSVM_eval) + b - numpy.log(pT / (1-pT))).ravel()
        minDCFSVM.append(bayesRisk.compute_minDCF_binary_fast(calibrated_S, L, effPrior, 1.0, 1.0))
        actDCFSVM.append(bayesRisk.compute_actDCF_binary_fast(calibrated_S, L, effPrior, 1.0, 1.0))
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(effPriorLogOdds, actDCFGMM, label='actDCF GMM', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCFGMM, label='minDCF GMM', color='b')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCFSVM, label='minDCF SVM', color='g')
    matplotlib.pyplot.plot(effPriorLogOdds, actDCFSVM, label='actDCF SVM', color='c')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCFLR, label='minDCF LR', color='m')
    matplotlib.pyplot.plot(effPriorLogOdds, actDCFLR, label='actDCF LR', color='y')
    matplotlib.pyplot.ylim([0, 1.1])
    #matplotlib.pyplot.show()
    plt.legend()
    plt.title('Bayes error plot, best models (not their fusion)')
    plt.savefig('BayesErrorPlot')

    #Last point of the analysis
    print('Starting point of the last part\n')
    print()
    print()
    for covType in ['full', 'diagonal']:
        for numCclass0 in [1,2,4,8,16,32]:
            for numCclass1 in [1,2,4,8,16,32]:
                gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numCclass0, covType = covType, verbose=False, psiEig = 0.01)
                gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numCclass1, covType = covType, verbose=False, psiEig = 0.01)
                SLLR = logpdf_GMM(D, gmm1) - logpdf_GMM(D, gmm0)
                min_dcf = bayesRisk.compute_minDCF_binary_fast(SLLR, L, 0.1, 1.0, 1.0) #always on the target application
                print("%d components for class 0, %d components for class 1. CovType: %s. Min dcf: %.4f" % (numCclass0, numCclass1, covType, min_dcf))
                print()
    
