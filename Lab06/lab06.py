import sys
import numpy
import string
import scipy.special
import itertools

def mcol(v):
    return v.reshape((v.size, 1))


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

def S1_buildDictionary(lTercets):

    '''
    Create a set of all words contained in the list of tercets lTercets
    lTercets is a list of tercets (list of strings)
    '''

    sDict = set([])
    for s in lTercets:
        words = s.split()
        for w in words:
            sDict.add(w)
    return sDict

def S1_estimateModel(hlTercets, eps = 0.1):

    '''
    Build frequency dictionaries for each class.

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: dictionary h_clsLogProb whose keys are the classes. For each class, h_clsLogProb[cls] is a dictionary whose keys are words and values are the corresponding log-frequencies (model parameters for class cls)
    '''

    # Build the set of all words appearing at least once in each class
    sDictCommon = set([])

    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        sDictCls = S1_buildDictionary(lTercets)
        sDictCommon = sDictCommon.union(sDictCls)

    # Initialize the counts of words for each class with eps
    h_clsLogProb = {}
    for cls in hlTercets: # Loop over class labels
        h_clsLogProb[cls] = {w: eps for w in sDictCommon} # Create a dictionary for each class that contains all words as keys and the pseudo-count as initial values

    # Estimate counts
    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets: # Loop over all tercets of the class
            words = tercet.split()
            for w in words: # Loop over words of the given tercet
                h_clsLogProb[cls][w] += 1
            
    # Compute frequencies
    for cls in hlTercets: # Loop over class labels
        nWordsCls = sum(h_clsLogProb[cls].values()) # Get all occurrencies of words in cls and sum them. this is the number of words (including pseudo-counts)
        for w in h_clsLogProb[cls]: # Loop over all words
            h_clsLogProb[cls][w] = numpy.log(h_clsLogProb[cls][w]) - numpy.log(nWordsCls) # Compute log N_{cls,w} / N

    return h_clsLogProb

def S1_compute_logLikelihoods(h_clsLogProb, text):

    '''
    Compute the array of log-likelihoods for each class for the given text
    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    The function returns a dictionary of class-conditional log-likelihoods
    '''
    
    logLikelihoodCls = {cls: 0 for cls in h_clsLogProb}
    for cls in h_clsLogProb: # Loop over classes
        for word in text.split(): # Loop over words
            if word in h_clsLogProb[cls]:
                logLikelihoodCls[cls] += h_clsLogProb[cls][word]
    return logLikelihoodCls

def S1_compute_logLikelihoodMatrix(h_clsLogProb, lTercets, hCls2Idx = None):

    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used
   
    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''
    
    if hCls2Idx is None:
        hCls2Idx = {cls:idx for idx, cls in enumerate(sorted(h_clsLogProb))}

    S = numpy.zeros((len(h_clsLogProb), len(lTercets)))
    for tIdx, tercet in enumerate(lTercets):
        hScores = S1_compute_logLikelihoods(h_clsLogProb, tercet)
        for cls in h_clsLogProb: # We sort the class labels so that rows are ordered according to alphabetical order of labels
            clsIdx = hCls2Idx[cls]
            S[clsIdx, tIdx] = hScores[cls]

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

def compute_accuracy(P, L):

    '''
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label (in alphabetical order)
    '''

    PredictedLabel = numpy.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect)/float(NTotal)

if __name__ == '__main__':

    lInf, lPur, lPar = load_data()

    lInfTrain, lInfEval = split_data(lInf, 4)
    lPurTrain, lPurEval = split_data(lPur, 4)
    lParTrain, lParEval = split_data(lPar, 4)


    ### Solution 1 ###
    ### Multiclass ###

    hCls2Idx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'purgatorio': lPurTrain,
        'paradiso': lParTrain
        }

    lTercetsEval = lInfEval + lPurEval + lParEval

    S1_model = S1_estimateModel(hlTercetsTrain, eps = 0.001)

    S1_predictions = compute_classPosteriors(
        S1_compute_logLikelihoodMatrix(
            S1_model,
            lTercetsEval,
            hCls2Idx,
            ),
        numpy.log(numpy.array([1./3., 1./3., 1./3.]))
        )

    labelsInf = numpy.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = numpy.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsPur = numpy.zeros(len(lPurEval))
    labelsPur[:] = hCls2Idx['purgatorio']

    labelsEval = numpy.hstack([labelsInf, labelsPur, labelsPar])

    # Per-class accuracy
    print('Multiclass - S1 - Inferno - Accuracy: %.2f%%' % (compute_accuracy(S1_predictions[:, labelsEval==hCls2Idx['inferno']], labelsEval[labelsEval==hCls2Idx['inferno']])*100))
    print('Multiclass - S1 - Purgatorio - Accuracy: %.2f%%' % (compute_accuracy(S1_predictions[:, labelsEval==hCls2Idx['purgatorio']], labelsEval[labelsEval==hCls2Idx['purgatorio']])*100))
    print('Multiclass - S1 - Paradiso - Accuracy: %.2f%%' % (compute_accuracy(S1_predictions[:, labelsEval==hCls2Idx['paradiso']], labelsEval[labelsEval==hCls2Idx['paradiso']])*100))

    # Overall accuracy
    print('Multiclass - S1 - Accuracy: %.2f%%' % (compute_accuracy(S1_predictions, labelsEval)*100))




    ### Binary from multiclass scores [Optional, for the standard binary case see below] ###
    ### Only inferno vs paradiso, the other pairs are similar ###

    lTercetsEval = lInfEval + lParEval
    S = S1_compute_logLikelihoodMatrix(S1_model, lTercetsEval, hCls2Idx = hCls2Idx)

    SBinary = numpy.vstack([S[0:1, :], S[2:3, :]])
    P = compute_classPosteriors(SBinary)
    labelsEval = numpy.hstack([labelsInf, labelsPar])
    # Since labelsPar == 2, but the row of Paradiso in SBinary has become row 1 (row 0 is Inferno), we have to modify the labels for paradise, otherwise the function compute_accuracy will not work
    labelsEval[labelsEval == 2] = 1

    print('Binary (From multiclass) - S1 - Accuracy: %.2f%%' % (compute_accuracy(P, labelsEval)*100))

    ### Binary ###
    ### Only inferno vs paradiso, the other pairs are similar ###

    hCls2Idx = {'inferno': 0, 'paradiso': 1}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'paradiso': lParTrain
        }

    lTercetsEval = lInfEval + lParEval

    S1_model = S1_estimateModel(hlTercetsTrain, eps = 0.001)

    S1_predictions = compute_classPosteriors(
        S1_compute_logLikelihoodMatrix(
            S1_model,
            lTercetsEval,
            hCls2Idx,
            ),
        numpy.log(numpy.array([1./2., 1./2.]))
        )

    labelsInf = numpy.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = numpy.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsEval = numpy.hstack([labelsInf, labelsPar])

    print('Binary [inferno vs paradiso] - S1 - Accuracy: %.2f%%' % (compute_accuracy(S1_predictions, labelsEval)*100))