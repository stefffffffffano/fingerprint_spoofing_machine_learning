import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import sklearn.datasets

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # Remove setosa
    L = L[L != 0]    # Remove setosa
    L[L == 2] = 0    # Assign label 0 to virginica
    return D, L

def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    return np.int32(llr > th)

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1-prior) * Cfp)
    return bayesError

def compute_min_normalized_dcf(llr, LTE, pi1, Cfn, Cfp):
    min_normalized_dcf = float('inf')
    thresholds = np.unique(llr)
    for threshold in thresholds:
        predicted_labels = np.where(llr <= threshold, 0, 1)
        DCF_normalized = compute_empirical_Bayes_risk_binary(predicted_labels, LTE, pi1, Cfn, Cfp)
        min_normalized_dcf = min(min_normalized_dcf, DCF_normalized)
    return min_normalized_dcf

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def trainSVM(DTR, LTR, C, K):
    # Estende la matrice dei dati
    D_ext = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    ZTR = 2 * LTR - 1  # Converte le etichette in +1, -1
    
    # Calcola la matrice H
    H = np.outer(ZTR, ZTR) * np.dot(D_ext.T, D_ext)

    # Definisce la funzione obiettivo e il suo gradiente
    def svm_dual_obj(alpha):
        alpha = alpha.reshape(-1, 1)
        term1 = 0.5 * np.dot(alpha.T, np.dot(H, alpha))[0, 0]
        term2 = np.sum(alpha)
        loss = term1 - term2
        grad = np.dot(H, alpha).ravel() - 1
        return loss, grad

    # Definisce i vincoli
    bounds = [(0, C) for _ in range(DTR.shape[1])]
    alpha0 = np.zeros(DTR.shape[1])

    # Minimizza la funzione obiettivo
    alpha_opt, _, _ = fmin_l_bfgs_b(func=svm_dual_obj, x0=alpha0, bounds=bounds, approx_grad=False, factr=1.0)

    # Calcola il vettore dei pesi primale e il termine di bias
    w_star = np.sum((alpha_opt * ZTR) * D_ext,axis=1)
    b = w_star[-1]
    w= w_star[:-1]
    return w, b, alpha_opt

def compute_primal_objective(w, b, DVAL, LVAL, C):
    # Calcolo delle etichette di classe scalate a {-1, +1}
    ZTR = 2 * LVAL - 1
    # Calcolo del punteggio
    S = vcol(w).T @ DVAL + b
    # Calcolo della regolarizzazione
    regularization_term = 0.5 * np.sum(w ** 2)
    
    # Calcolo della hinge loss
    hinge_loss_term = C * np.sum(np.maximum(0, 1 - ZTR * S))
    # Loss primale totale
    primal_loss = regularization_term + hinge_loss_term
    return S, primal_loss

def compute_dual_objective(alpha, H):
    term1 = 0.5 * np.dot(alpha.T, np.dot(H, alpha))
    term2 = np.sum(alpha)
    return term2 - term1

def evaluate_accuracy(DTE, LTE, w, b):
    scores = np.dot(w.T, DTE) + b
    predictions = (scores > 0).astype(int)
    accuracy = np.mean(predictions == LTE)
    return accuracy

def computeEmpiricalPrior(LTR):
    return np.sum(LTR) / LTR.size

if __name__ == "__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    empirical_prior = computeEmpiricalPrior(LTR)
    K = [1, 1, 1, 10, 10, 10]
    C = [0.1, 1.0, 10, 0.1, 1.0, 10]
    for k, c in zip(K, C):
        w, b, alpha_opt = trainSVM(DTR, LTR, c, k)
        S, loss = compute_primal_objective(w, b, DVAL, LVAL, c)
        Sllr = S - np.log(empirical_prior / (1 - empirical_prior))
        D_ext = np.vstack([DTR, np.ones(DTR.shape[1]) * k])
        ZTR = 2 * LTR - 1
        G = np.dot(D_ext.T, D_ext)
        H = np.outer(ZTR, ZTR) * G
        dual_obj = compute_dual_objective(alpha_opt, H)
        print("Primal objective with K = %d and C = %.1f: %f" % (k, c, loss))
        print("Dual objective with K = %d and C = %.1f: %f" % (k, c, dual_obj))
        print("Duality gap with K = %d and C = %.1f: %f" % (k, c, np.abs(loss - dual_obj)))
        print("Error rate with K = %d and C = %.1f: %f" % (k, c, (1 - evaluate_accuracy(DVAL, LVAL, w, b)) * 100))
        print("Actual DCF: %f" % compute_empirical_Bayes_risk_binary(compute_optimal_Bayes_binary_llr(Sllr, 0.5, 1, 1), LVAL, 0.5, 1, 1))
        print("Min DCF: %f" % compute_min_normalized_dcf(Sllr, LVAL, 0.5, 1, 1))
        print("-" * 80)
