import sklearn.datasets
import scipy.linalg
import numpy
import matplotlib
import matplotlib.pyplot as plt
import math
import itertools

def mcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))


def compute_mu_C(D):
    mu = mcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def load_data():

    lInf = []

    f=open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f=open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f=open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()
    
    return lInf, lPur, lPar

def split_data(l, n):

    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
            
    return lTrain, lTest


def S2_buildDictionary(lTercets):

    '''
    Create a dictionary of all words contained in the list of tercets lTercets
    The dictionary allows storing the words, and mapping each word to an index i (the corresponding index in the array of occurrencies)

    lTercets is a list of tercets (list of strings)
    '''

    hDict = {}
    nWords = 0
    for tercet in lTercets:
        words = tercet.split()
        for w in words:
            if w not in hDict:
                hDict[w] = nWords
                nWords += 1
    return hDict

def S2_estimateModel(hlTercets, eps = 0.1):

    '''
    Build word log-probability vectors for all classes

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: tuple (h_clsLogProb, h_wordDict). h_clsLogProb is a dictionary whose keys are the classes. For each class, h_clsLogProb[cls] is an array containing, in position i, the log-frequency of the word whose index is i. h_wordDict is a dictionary that maps each word to its corresponding index.
    '''

    # Since the dictionary also includes mappings from word to indices it's more practical to build a single dict directly from the complete set of tercets, rather than doing it incrementally as we did in Solution S1
    lTercetsAll = list(itertools.chain(*hlTercets.values())) 
    hWordDict = S2_buildDictionary(lTercetsAll)
    nWords = len(hWordDict) # Total number of words

    h_clsLogProb = {}
    for cls in hlTercets:
        h_clsLogProb[cls] = numpy.zeros(nWords) + eps # In this case we use 1-dimensional vectors for the model parameters. We will reshape them later.
    
    # Estimate counts
    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets: # Loop over all tercets of the class
            words = tercet.split()
            for w in words: # Loop over words of the given tercet
                wordIdx = hWordDict[w]
                h_clsLogProb[cls][wordIdx] += 1 # h_clsLogProb[cls] ius a 1-D array, h_clsLogProb[cls][wordIdx] is the element in position wordIdx

    # Compute frequencies
    for cls in h_clsLogProb.keys(): # Loop over class labels
        vOccurrencies = h_clsLogProb[cls]
        vFrequencies = vOccurrencies / vOccurrencies.sum()
        vLogProbabilities = numpy.log(vFrequencies)
        h_clsLogProb[cls] = vLogProbabilities

    return h_clsLogProb, hWordDict
    
def S2_tercet2occurrencies(tercet, hWordDict):
    
    '''
    Convert a tercet in a (column) vector of word occurrencies. Word indices are given by hWordDict
    '''
    v = numpy.zeros(len(hWordDict))
    for w in tercet.split():
        if w in hWordDict: # We discard words that are not in the dictionary
            v[hWordDict[w]] += 1
    return mcol(v)

def S2_compute_logLikelihoodMatrix(h_clsLogProb, hWordDict, lTercets, hCls2Idx = None):

    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb and hWordDict are the dictionary of model parameters and word indices as returned by S2_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used
   
    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''

    if hCls2Idx is None:
        hCls2Idx = {cls:idx for idx, cls in enumerate(sorted(h_clsLogProb))}
    
    numClasses = len(h_clsLogProb)
    numWords = len(hWordDict)

    # We build the matrix of model parameters. Each row contains the model parameters for a class (the row index is given from hCls2Idx)
    MParameters = numpy.zeros((numClasses, numWords)) 
    for cls in h_clsLogProb:
        clsIdx = hCls2Idx[cls]
        MParameters[clsIdx, :] = h_clsLogProb[cls] # MParameters[clsIdx, :] is a 1-dimensional view that corresponds to the row clsIdx, we can assign to the row directly the values of another 1-dimensional array

    SList = []
    for tercet in lTercets:
        v = S2_tercet2occurrencies(tercet, hWordDict)
        STercet = numpy.dot(MParameters, v) # The log-lieklihoods for the tercets can be computed as a matrix-vector product. Each row of the resulting column vector corresponds to M_c v = sum_j v_j log p_c,j
        SList.append(numpy.dot(MParameters, v))

    S = numpy.hstack(SList)
    return S


def compute_classPosteriors(S, logPrior = None):

    '''
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    '''

    if logPrior is None:
        logPrior = numpy.log( numpy.ones(S.shape[0]) / float(S.shape[0]) )
    J = S + mcol(logPrior) # Compute joint probability
    ll = scipy.special.logsumexp(J, axis = 0) # Compute marginal likelihood log f(x)
    P = J - ll # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return numpy.exp(P)


def print_confusion_matrix(true_labels, predicted_labels, num_classes):
    # Initialize confusion matrix with zeros
    conf_matrix = numpy.zeros((num_classes, num_classes), dtype=int)
    
    # Fill confusion matrix
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        conf_matrix[predicted_label, true_label] += 1
    
    # Print confusion matrix
    #print("Confusion Matrix:")
    #print("   |", end="")
    #for i in range(num_classes):
        #print(f" {i} |", end="")
    #print()
    #print("-" * (4 * (num_classes + 1)))

    #for i in range(num_classes):
        #print(f" {i} |", end="")
        #for j in range(num_classes):
            #print(f" {conf_matrix[i][j]:2} |", end="")
        #print()
    return conf_matrix

def compute_optimal_bayes_decisions(pi1, Cfn, Cfp):
    
    # Load LLRs 
    llr = numpy.load("data/commedia_llr_infpar.npy")

    # Compute log prior ratio
    threshold = -numpy.log((pi1 / (1 - pi1))*(Cfn/Cfp))
    # Compute predicted labels based on log-likelihood ratios and threshold
    predicted_labels = numpy.where(llr <= threshold, 0, 1)
    return predicted_labels

def compute_dcf(conf_matrix, pi1, Cfn, Cfp):
    # Pfn and Pfp
    Pfn = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1, 1])
    Pfp = conf_matrix[1,0] / (conf_matrix[1,0] + conf_matrix[0, 0])

    # Calculate DCF
    DCF = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
    return DCF

def compute_normalized_dcf(conf_matrix, pi1, Cfn, Cfp):
    # Pfn and Pfp
    Pfn = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1, 1])
    Pfp = conf_matrix[1,0] / (conf_matrix[1,0] + conf_matrix[0, 0])

    # Calculate DCF
    DCF = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
    Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
    DCF_normalized = DCF / Bdummy
    return DCF_normalized

def compute_min_normalized_dcf(pi1, Cfn, Cfp):
    # Load LLRs 
    llr = numpy.load("data/commedia_llr_infpar.npy")

    # Calculate Bdummy
    Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)

    # Initialize minimum normalized DCF
    min_normalized_dcf = float('inf')

    # Iterate through all possible thresholds
    thresholds = numpy.unique(llr)
    for threshold in thresholds:
        predicted_labels = numpy.where(llr <= threshold, 0, 1)
        confusion_matrix = print_confusion_matrix(labels, predicted_labels.tolist(), 2)
        DCF = compute_dcf(confusion_matrix, pi1, Cfn, Cfp)
        DCF_normalized = DCF / Bdummy
        min_normalized_dcf = min(min_normalized_dcf, DCF_normalized)

    return min_normalized_dcf

def compute_roc_curve():
    # Load LLRs and labels
    llr = numpy.load("data/commedia_llr_infpar.npy")
    labels = numpy.load("data/commedia_labels_infpar.npy")

    # Initialize lists to store TPR and FPR
    tpr_list = []
    fpr_list = []

    # Iterate through all possible thresholds
    thresholds = numpy.unique(llr)
    for threshold in thresholds:
        predicted_labels = numpy.where(llr <= threshold, 0, 1)
        confusion_matrix = print_confusion_matrix(labels, predicted_labels.tolist(), 2)
        tn = confusion_matrix[0][0]
        fn = confusion_matrix[0][1]
        fp = confusion_matrix[1][0]
        tp = confusion_matrix[1][1]

        
        tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
        fpr = fp / (fp + tn)  # False Positive Rate

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Plot ROC curve
    plt.plot(fpr_list, tpr_list)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def compute_prior_eff(log_odds_prior):
    return 1 / (1 + numpy.exp(-log_odds_prior))

def prior_log_odds(labels):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)

    # Calcola il prior effettivo corrispondente a ciascun valore di log-odds del prior
    pi_eff = compute_prior_eff(effPriorLogOdds)

    # Inizializza gli array per contenere i valori di DCF e minimo DCF
    dcf_values = []
    min_dcf_values = []

    # Calcola il DCF e il minimo DCF per ciascun prior effettivo
    for pi in pi_eff:
        predictions = compute_optimal_bayes_decisions(pi, 1, 1)
        confusion_matrix = print_confusion_matrix(labels, predictions.tolist(), 2)
        dcf = compute_normalized_dcf(confusion_matrix,pi, 1, 1)  # Assumendo Cfn = 1 e Cfp = 1
        min_dcf = compute_min_normalized_dcf(pi, 1, 1)  # Assumendo Cfn = 1 e Cfp = 1
        dcf_values.append(dcf)
        min_dcf_values.append(min_dcf)

    # Plot della DCF e del minimo DCF rispetto al log-odds del prior
    plt.plot(effPriorLogOdds, dcf_values, label='DCF', color='r')
    plt.plot(effPriorLogOdds, min_dcf_values, label='min DCF', color='b')
    plt.xlabel('prior Log-odds')
    plt.ylabel('DCF')
    plt.title('Bayes Error Plot')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ =='__main__':
    labels = numpy.load('data/commedia_labels_infpar.npy')
    pi1_values = [0.5, 0.8, 0.5, 0.8]
    Cfn_values = [1, 1, 10, 1]
    Cfp_values = [1, 1, 1, 10]
    for pi1, Cfn, Cfp in zip(pi1_values, Cfn_values, Cfp_values):
        Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
        predictions = compute_optimal_bayes_decisions(pi1, Cfn, Cfp)
        print(f"π1 = {pi1}, Cfn = {Cfn}, Cfp = {Cfp}")
        confusion_matrix = print_confusion_matrix(labels, predictions.tolist(), 2)
        DCF= compute_dcf(confusion_matrix,pi1,Cfn,Cfp)
        min_normalized_DCF =compute_min_normalized_dcf(pi1,Cfn,Cfp)
        DCFNormalized = DCF/Bdummy
        print(f"{DCFNormalized:.3f}")
        print(f"{min_normalized_DCF:.3f}")
    #compute_roc_curve()
    prior_log_odds(labels)
    
