import sys
import numpy
import matplotlib.pyplot as plt

def mcol(vett):
    return vett.reshape((vett.size,1))

def load(fileName):
    Dlist=[]
    L=[]
    with open(fileName) as f:
        for line in f:
            attrs = line.split(',')[0:-1]
            attrs = mcol(numpy.array([float(i) for i in attrs]))
            Dlist.append(attrs)
            label= line.split(',')[-1].strip()
            L.append(int(label))
    return numpy.hstack(Dlist),numpy.array(L)

def plot_hist(D, L):
    D0 = D[:, L==0] #counterfeit
    D1 = D[:, L==1] #genuine
    
    for dIdx in range(6):
        plt.figure()
        plt.xlabel(("Feature %d" % (dIdx+1)))
        plt.hist(D0[dIdx, :], bins = 10, density = False, alpha = 0.4, label = 'Counterfeit')
        plt.hist(D1[dIdx, :], bins = 10, density = False, alpha = 0.4, label = 'Genuine')
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    for dIdx1 in range(6):
        for dIdx2 in range(6):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(("Feature %d" % (dIdx1+1)))
            plt.ylabel(("Feature %d" % (dIdx2+1)))
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Counterfeit')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Genuine')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
    plt.show()


if __name__=='__main__':
    fileName=sys.argv[1]
    D,L=load(fileName)
    # Change default font size as in the lab solution
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plot_hist(D,L)
    #plot_scatter(D,L)

    #PRINT THE MEAN FOR EACH FEATURE
    mu = D.mean(1).reshape((D.shape[0], 1))
    print('Mean:')
    print(mu) #this way I print the mean for all the classes
    print()

    #PRINT THE COVARIANCE FOR EACH FEATURE
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    print('Covariance:')
    print(C)
    print()

    #PRINT VARIANCE AND STANDARD DEVIATION FOR EACH FEATURE
    var = D.var(1)
    std = D.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()
    
    for cls in [0,1]:
        print('Class', 'Counterfeit' if cls==0 else 'Genuine')
        DCls = D[:, L==cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean:')
        print(mu)
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print('Covariance:')
        print(C)
        var = DCls.var(1)
        std = DCls.std(1)
        print('Variance:', var)
        print('Std. dev.:', std)
        print()