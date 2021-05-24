import numpy as np


# Wasserstein distance on the real line
def wasserstein(x, y, p, a=None, b=None):	
    n = len(x)
    m = len(y)
    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m
    #  cumulative distirbutions
    ca = np.cumsum(a) 
    cb = np.cumsum(b) 

    # points on which we need to evaluate the quantile functions
    cba = np.sort(np.hstack([ca, cb])) 

    # construction of first quantile function
    bins = ca + 1e-10 # bins need some small tollerance to avoid numerical rounding errors
    bins = np.hstack([-np.Inf, bins , np.Inf]) 
    index_qx = np.digitize(cba, bins, right=True) - 1    # right=True becouse quantile function is right continuous.
    x = np.sort(x)
    qx = x[index_qx]

    # constuction of second quantile function
    bins = cb + 1e-10
    bins = np.hstack([-np.Inf, bins , np.Inf]) 
    index_qy = np.digitize(cba, bins, right=True) - 1    # right=True becouse quantile function is right continuous.
    y = np.sort(y)
    qy = y[index_qy]

    # weights for the inegral
    h = np.diff(np.hstack([0, cba]))
    # evaluation of integral
    res = np.sum(np.abs(qy - qx)**p * h )**(1/p)
    return res

# quick test
if __name__=="__main__":
    import ot

    np.random.seed(0)
    n = 60
    m = 10
    p = 2
    x = np.random.random(size=n)
    y = np.random.random(size=m)

    res1 = wasserstein(x, y, p, a=None, b=None)

    # Just as a general check we can compute the same distance with the pot package
    # building cost matrix
    u = x.reshape((-1, 1))
    v = y.reshape((1, -1))
    cost= np.abs(u-v)**p
    cost  = cost.copy(order='C') # for some reason the emd2 function requires C contingency..

    res2 = ot.emd2(np.ones(n)/n, np.ones(m)/m, cost)**(1/p)
    print(res1, res2)
