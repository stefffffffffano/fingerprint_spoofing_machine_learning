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

def MVG(DTR,LTR,DTE,LTE,effective_prior):
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, C0)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, C1)
    llr = pdfGau1Log - pdfGau0Log
    predictions = numpy.where(llr >= -numpy.log(effective_prior/(1-effective_prior)), 1, 0)
    return (predictions,llr)

def tiedGaussian(DTR,LTR,DTE,LTE,effective_prior):
    Sb,Sw=compute_Sb_Sw(DTR,LTR)
    #Sw is the within class covariance matrix
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, Sw)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, Sw)
    llr = pdfGau1Log - pdfGau0Log
    predictions = numpy.where(llr >= -numpy.log(effective_prior/(1-effective_prior)), 1, 0)
    return (predictions,llr)

def naiveBayes(DTR,LTR,DTE,LTE,effective_prior):
    mu0,C0=compute_mu_C(DTR[:,LTR==0])
    mu1,C1=compute_mu_C(DTR[:,LTR==1])
    C0diag=C0*numpy.eye(C0.shape[0])
    C1diag=C1*numpy.eye(C1.shape[0])
    pdfGau0Log = logpdf_GAU_ND(DTE, mu0, C0diag)
    pdfGau1Log = logpdf_GAU_ND(DTE, mu1, C1diag)
    llr = pdfGau1Log - pdfGau0Log
    predictions = numpy.where(llr >= -numpy.log(effective_prior/(1-effective_prior)), 1, 0)
    return (predictions,llr)

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


def compute_effective_prior(pi1, Cfn, Cfp):
    effective_prior = (pi1*Cfn) / (pi1*Cfn + (1 - pi1) * Cfp)
    return effective_prior



def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M




def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

def compute_prior_eff(log_odds_prior):
    return 1.0 / (1.0 + numpy.exp(-log_odds_prior))

def plot_prior_log_odds(predictions,llr,LTE,classifier):
    effPriorLogOdds = numpy.linspace(-4, 4, 30)
    # Inizializza gli array per contenere i valori di DCF e minimo DCF
    dcf_values = []
    min_dcf_values = []
    # Calcola il prior effettivo corrispondente a ciascun valore di log-odds del prior
    pi_eff = compute_prior_eff(effPriorLogOdds)

    for effPrior in pi_eff:
        predictions = compute_optimal_Bayes_binary_llr(llr, effPrior, 1.0, 1.0)
        dcf_values.append(compute_empirical_Bayes_risk_binary(predictions, LTE, effPrior, 1.0, 1.0))
        min_dcf_values.append(compute_min_normalized_dcf(llr, LTE, effPrior, 1.0, 1.0))
    plt.figure()
    plt.plot(effPriorLogOdds, dcf_values, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, min_dcf_values, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.title('Bayes Error Plot for %s classifier' %classifier)
    plt.legend()
    plt.grid(True)
    #I don'twant to plot it, but I want to save it
    plt.savefig('BayesErrorPlot_%s.png' %classifier)
    
    
    


if __name__ == '__main__':
    D, L = load('trainData.txt')
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    pi1_values = [0.5, 0.9, 0.1, 0.5,0.5]
    Cfn_values = [1.0, 1.0, 1.0, 1.0,9.0]
    Cfp_values = [1.0, 1.0, 1.0, 9.0,1.0]
    best_results = {}  
    classifiers = {"MVG": MVG, "Tied Gaussian": tiedGaussian, "Naive Bayes": naiveBayes}
    for pi1, Cfn, Cfp in zip(pi1_values, Cfn_values, Cfp_values):
        best_results[pi1] = {}
        #Represent each application in terms of the effective prior
        effective_prior=compute_effective_prior(pi1,Cfn,Cfp)
        #MVG
        (predictionsMVG,llrMVG) = MVG(DTR,LTR,DTE,LTE,effective_prior)

        #tied Gaussian Model
        (predictionsTied,llrTied)=tiedGaussian(DTR,LTR,DTE,LTE,effective_prior)

        #Naive Bayes Gaussian model
        (predictionsBayes,llrBayes) = naiveBayes(DTR,LTR,DTE,LTE,effective_prior)
        print("Application with effective prior %f" %effective_prior)
        print("MVG classifier:")
        print(compute_confusion_matrix(predictionsMVG,LTE))
        DCF = compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(llrMVG,pi1,Cfn,Cfp),LTE,pi1,Cfn,Cfp)
        minDCF=compute_min_normalized_dcf(llrMVG,LTE,pi1,Cfn,Cfp)
        print("Actual normalized DCF: %f" %DCF)
        print("Minimum normalized DCF: %f" %minDCF)
        best_results[pi1]["MVG"] = {"best_m": None, "best_min_dcf": float('inf')}
        best_results[pi1]["MVG"]["best_m"] = 0
        best_results[pi1]["MVG"]["best_min_dcf"] = minDCF
        print("Tied Gaussian classifier:")
        print(compute_confusion_matrix(predictionsTied,LTE))
        DCF=compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(llrTied,pi1,Cfn,Cfp),LTE,pi1,Cfn,Cfp)
        minDCF=compute_min_normalized_dcf(llrTied,LTE,pi1,Cfn,Cfp)
        best_results[pi1]["Tied Gaussian"] = {"best_m": None, "best_min_dcf": float('inf')}
        print("Actual normalized DCF: %f" %DCF)
        print("Minimum normalized DCF: %f" %minDCF)
        best_results[pi1]["Tied Gaussian"]["best_m"] = 0
        best_results[pi1]["Tied Gaussian"]["best_min_dcf"] = minDCF
        print("Tied Gaussian classifier:")
        print("Naive Bayes classifier:")
        print(compute_confusion_matrix(predictionsBayes,LTE))
        DCF=compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(llrBayes,pi1,Cfn,Cfp),LTE,pi1,Cfn,Cfp)
        minDCF=compute_min_normalized_dcf(llrBayes,LTE,pi1,Cfn,Cfp)
        print("Actual normalized DCF: %f" %DCF)
        print("Minimum normalized DCF: %f" %minDCF)
        best_results[pi1]["Naive Bayes"] = {"best_m": None, "best_min_dcf": float('inf')}
        best_results[pi1]["Naive Bayes"]["best_m"] = 0
        best_results[pi1]["Naive Bayes"]["best_min_dcf"] = minDCF

    """
    Let's now apply PCA to reduce the feature subspace from 6 to 5, 4, 3, 2 and 1 to evaluate which is the best configuration
    for the three applications taken into considerations
    """
    print("-"*80)
    
    for pi_tilde in [0.1, 0.5, 0.9]:
        print("Application with effective prior %f" % pi_tilde)
        for classifier_name, classifier_func in classifiers.items():
            print("Classifier: %s" % classifier_name)
            # Dictionary to store best values for each classifier
           
            for i in range(6):
                m = 6 - i
                print("PCA applied to reduce the feature subspace from 6 to %d" % m)
                UPCA = compute_pca(DTR, m=m) 
                DTR_pca = apply_pca(UPCA, DTR)   
                DTE_pca = apply_pca(UPCA, DTE) 
                (predictions, llr) = classifier_func(DTR_pca, LTR, DTE_pca, LTE,pi_tilde)
                
                # Calculate actual DCF and min normalized DCF
                actual_dcf = compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(llr, pi_tilde, 1, 1), LTE, pi_tilde, 1, 1)
                min_dcf = compute_min_normalized_dcf(llr, LTE, pi_tilde, 1.0, 1.0)
                print("Actual normalized DCF: %f" % actual_dcf)
                print("Minimum normalized DCF: %f" % min_dcf)
                
                # Update results if necessary
                if min_dcf < best_results[pi_tilde][classifier_name]["best_min_dcf"]:
                    best_results[pi_tilde][classifier_name]["best_m"] = m
                    best_results[pi_tilde][classifier_name]["best_min_dcf"] = min_dcf

    # Print results for each pi_tilde and for each classifier
    for pi_tilde, pi_tilde_results in best_results.items():
        print("For pi_tilde = %f:" % pi_tilde)
        for classifier_name, results in pi_tilde_results.items():
            print("Classifier: %s" % classifier_name)
            print("Best m: %d" % results["best_m"])
            print("Best minimum normalized DCF: %f" % results["best_min_dcf"])

    pi = 0.1
    for classifier_name, classifier_func in classifiers.items():
        best_m = best_results[pi][classifier_name]["best_m"]
        print(f"Applying {classifier_name} with best m = {best_m} for pi = {pi}")
        if best_m == 0:
            (predictions, llr) = classifier_func(DTR, LTR, DTE, LTE,0.1)
        else:
            UPCA = compute_pca(DTR, m=best_m)
            DTR_pca = apply_pca(UPCA, DTR)
            DTE_pca = apply_pca(UPCA, DTE)
            (predictions, llr) = classifier_func(DTR_pca, LTR, DTE_pca, LTE,0.1)
        plot_prior_log_odds(predictions,llr,LTE,classifier_name)

            

   
    
    
    
    
    
    
    
    
    

    

