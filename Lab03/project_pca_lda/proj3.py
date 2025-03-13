import numpy
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets 
import scipy.linalg

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
        plt.savefig('proj_hist_Direction%d.pdf' % dIdx)
    #plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    for dIdx1 in range(6):
        for dIdx2 in range(6):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(("Dimension %d" % (dIdx1+1)))
            plt.ylabel(("Dimension %d" % (dIdx2+1)))
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Counterfeit')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Genuine')
        
            plt.legend()
            plt.tight_layout() 
            plt.savefig('scatter_pca_dim%d_%d.pdf' % (dIdx1, dIdx2))
    #plt.show()


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D): 
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C


def compute_pca(D, m):
    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D

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

def compute_lda_geig(D, L, m):
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def apply_lda(U, D):
    return U.T @ D

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

if __name__ == '__main__':
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    D, L = load('trainData.txt')
    P = compute_pca(D, 6)
    ypca = apply_pca(P,D) #y is the reconstruction, projected with the six directions of PCA

    #showing results
    
    
    #plot_hist(ypca,L)
    #plot_scatter(ypca,L)
    #part related to PCA finished 
    

    #start of part related to LDA
    U= compute_lda_geig(D,L,1)
    ylda=apply_lda(U,D)
    
    #showing results
    
    c0 = ylda[:, L==0]
    c1 = ylda[:, L==1]
    plt.figure()
    plt.xlabel("LDA 1 dimension")
    plt.hist(c0[0, :], bins = 10, density = True, alpha = 0.4, label = 'Counterfeit')
    plt.hist(c1[0, :], bins = 10, density = True, alpha = 0.4, label = 'Genuine')
    plt.legend()
    plt.tight_layout() 
    plt.savefig('proj_hist_LDA.pdf')
    #plt.show()
    
    #part related to LDA finished

    #start of part related to classification

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    Uclass= compute_lda_geig(DTR,LTR,1)
    DTR_lda = apply_lda(Uclass, DTR)
    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        Uclass = -Uclass
        DTR_lda = apply_lda(Uclass, DTR)
    
    DVAL_lda  = apply_lda(Uclass, DVAL)

    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean())/2.0 
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    print('Error rate applying LDA: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
    #Try varying the threshold to see if the error rate improves
    min_error_rate = (PVAL != LVAL).sum() / float(LVAL.size) *100 
    min_threshold = threshold
    for new_threshold in numpy.linspace(-2, 2, 100000):
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 0].mean():
            PVAL[DVAL_lda[0] >= new_threshold] = 1
            PVAL[DVAL_lda[0] < new_threshold] = 0
        else:
            PVAL[DVAL_lda[0] >= new_threshold] = 0
            PVAL[DVAL_lda[0] < new_threshold] = 1

        error_rate = 0
        for i in range(LVAL.size):
            if PVAL[i] != LVAL[i]:
                error_rate += 1
        error_rate = error_rate / LVAL.size
        if(error_rate < min_error_rate):
            min_error_rate = error_rate
            min_threshold = threshold
        #print("Treshold = %f" % new_threshold)
        #print('Error rate for LDA = %f' % (error_rate * 100))

    print("Min error rate = " + str(min_error_rate*100))
    print("Threshold for min error rate = " + str(min_threshold))

    #PCA applied before LDA
    print("PCA applied before LDA to reduce the feature size from 6 to different values of m:")
    # Solution with PCA pre-processing with dimension m.

    for i in range(4):
        m = 6-(i+1)
        UPCA = compute_pca(DTR, m = m) 
        DTR_pca = apply_pca(UPCA, DTR)   
        DVAL_pca = apply_pca(UPCA, DVAL) 
        ULDA = compute_lda_geig(DTR_pca, LTR, m = 1) 

        DTR_lda = apply_lda(ULDA, DTR_pca)  
        if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
            ULDA = -ULDA
            DTR_lda = apply_lda(ULDA, DTR_pca)

        DVAL_lda = apply_lda(ULDA, DVAL_pca)
        threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean())/ 2.0
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        print('Number of erros with m = %d:'%m,(PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
        print('Error rate with m = %d: %.1f%%' % (m, (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
      