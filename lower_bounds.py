import numpy as np
import ot
from wasserstein_1d import wasserstein
from scipy.spatial.distance import pdist, squareform
def flb(x, y, d_x, d_y, p, a=None, b=None):
    n = len(x)
    m = len(y)
    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m
    # computing eccentricities
    e_x = np.sum(d_x**p * a, axis=1)**(1/p)
    e_y = np.sum(d_y**p * b, axis=1)**(1/p)
    res = wasserstein(e_x, e_y, p, a=a, b=b)
    return res


def slb(x, y, d_x, d_y, p, a=None, b=None):
    n = len(x)
    m = len(y)
    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m
    # product measures are created
    aa = a.reshape((1, -1)) * a.reshape((-1, 1))
    bb = b.reshape((1, -1)) * b.reshape((-1, 1))

    # now everything gets flattened..
    d_x = d_x.reshape((-1))
    d_y = d_y.reshape((-1))
    aa = aa.reshape((-1))
    bb = bb.reshape((-1))
    res = wasserstein(d_x, d_y, p, a=aa, b=bb)
    return res

def tlb(x, y, d_x, d_y, p, a=None, b=None):
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

    # constuction of second quantile function
    bins = cb + 1e-10
    bins = np.hstack([-np.Inf, bins , np.Inf]) 
    index_qy = np.digitize(cba, bins, right=True) - 1    # right=True becouse quantile function is right continuous.

    d_x = np.sort(d_x, axis=1)
    d_y = np.sort(d_y, axis=1)
    
    # quantiles of the r.v. d_x(X, x_i) evaluated in the points cba
    qx = d_x[:, index_qx]
    qy = d_y[:, index_qy]

    # weights for the inegral
    h = np.diff(np.hstack([0, cba]))

    # Evaluation of integral in different locations
    # I use a loop to avoid potential memory problems that could arise if qy - qx is vectorized
    omega = np.empty((n, m)) 
    for i in range(n):
        # if i%50==0:
        #     print("iteration ", i)
        integrand = np.abs(qy - qx[i, :])**p * h # notice that p-root is omitted
        omega_i = np.sum(integrand, axis=1)
        omega[i, :] = omega_i 
    # omega = np.sum(np.abs(qy.reshape((1, n, m, -1)) - qx.reshape((n, 1, n, m))**p * h )**(1/p)
    plan = ot.emd(a, b, omega)
    cost = np.sum(plan*omega)**(1/p)
    return cost, plan, omega

# test
if __name__=="__main__":
    np.random.seed(0)
    n = 200
    m = 200
    p = 2
    distance = "euclidean"
    x = np.random.random(size=(n, 3))
    y = np.random.normal(size=(m, 3))
    d_x = squareform(pdist(x, distance))
    d_y = squareform(pdist(y, distance))
    res_1 = flb(x, y, d_x, d_y, p, a=None, b=None)
    res_2 = slb(x, y, d_x, d_y, p, a=None, b=None)
    res_3 = tlb(x, y, d_x, d_y, p, a=None, b=None)[0]
    print(res_1)
    print(res_2)
    print(res_3)
#     print(ot.gromov.gromov_wasserstein(d_x, d_y, np.ones(n)/n, np.ones(m)/m, 'square_loss', verbose=False, log=True)[1]["gw_dist"]**(1/p))


