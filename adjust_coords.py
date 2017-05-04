import numpy as np
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class coordinates(object):
    def __init__(self, f_h5):
        self.h5 = h5py.File(f_h5,'r+')

    def __getitem__(self, N):
        return self.h5[str(N)]['coordinates'][:]

# Quaternion code adapted from
# http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion

def axisangle_to_q(v, theta):
    v /= np.linalg.norm(v)
    theta /= 2
    
    return np.array([
        np.cos(theta),
        v[0] * np.sin(theta),
        v[1] * np.sin(theta),
        v[2] * np.sin(theta),
    ])

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def q_conjugate(q):
    w, x, y, z = q
    return np.array((w, -x, -y, -z))

def qv_mult(q1, v1):
    q2 = np.hstack([(0.0,), v1])
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def qV_mult(q, X):
    # Normalize the quaternion
    q /= np.linalg.norm(q)
    return np.array([qv_mult(q,x) for x in X])

def orient_function(q, X, direction):
    qX = qV_mult(q, X)
    err  = np.linalg.norm(qX[0] - direction)
    return err

def orient(X,direction=[0,0,1]):
    print "Orienting", len(X)
    q0 = np.array([1,0,0,0])
    res = minimize(orient_function, q0,args=(X,direction),
                   method = 'Nelder-Mead')
    q = res.x
    return qV_mult(q, X)
    
def align_function(q, Y, X):
    qY = qV_mult(q, Y)
    err_orient = np.linalg.norm(qY - [0,0,1], axis=1).min()
    err_align = cdist(qY,X).sum()
    return err_align + 10*err_orient

def align(Y,X):
    print "Aligning", len(Y), len(X)
    q0 = np.array([1,0,0,0])
    res = minimize(align_function,q0,args=(Y,X),method = 'Nelder-Mead')
    q = res.x
    return qV_mult(q, Y)


if __name__ == "__main__":
    C = coordinates('configurations_test.h5')
    
    #X = C[10]
    #Y = C[11]
    #X = orient(X)
    #print align(Y,X)

    #pts = C[12]
    #print (cdist(pts,pts)**2)*2*3*5*7*11
    #exit()

    X = C[3]

    from scipy.spatial.distance import cdist
    print cdist(X,X,metric='cosine')

    #print cdist([[1,0,0]],[[-1,0,0]],metric='cosine')

        

