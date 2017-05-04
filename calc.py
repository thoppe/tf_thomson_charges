import os

# Uncomment this to hide all the ugly status messages
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# Uncomment this to run on the CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

import tensorflow as tf
import numpy as np
import h5py
import sys
from tqdm import tqdm
from scipy.spatial.distance import pdist
import pandas as pd

'''
Runs the tf model for the Thomson problem. Starts at N=2 and minimizes
in order. Saves best configurations to "configurations.h5". If a known 
solution exists, problem continues to run until it is found.
'''

FLAG_TRY_TO_MATCH_KNOWN = True

f_h5 = 'configurations.h5'


def thompson_model(N):
    tf.reset_default_graph()

    # Start with random coordinates from a normal dist
    r0 = np.random.normal(size=[N,3])
    coord = tf.Variable(r0, name='coordinates')

    # Normalize the coordinates onto the unit sphere
    coord = coord/tf.reshape(tf.norm(coord,axis=1),(-1,1))

    def squared_diff(A):
        r = tf.reduce_sum(A*A, 1)
        r = tf.reshape(r, [-1, 1])
        return r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

    RR = squared_diff(coord)

    # We don't need to compute the gradient over the symmetric distance
    # matrix, only the upper right half
    mask = np.triu(np.ones((N, N), dtype=np.bool_), 1)

    R = tf.sqrt(tf.boolean_mask(RR, mask))

    # Electostatic potential up to a constant, 1/r
    U = tf.reduce_sum(1/R)

    return U, coord

def minimize_thompson(N, reported_U=None, limit=10**10):
    
    U, coord = thompson_model(N)

    # Choose a high energy to start with
    previous_u = N**2

    learning_rate = 0.1
    LR = tf.placeholder(tf.float64, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=LR).minimize(U)
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for n in xrange(limit):
            for _ in tqdm(range(100)):
                sess.run(opt, feed_dict={LR:learning_rate})

            u = sess.run(U)
            delta_u = np.abs(previous_u - u)
            previous_u = u

            msg = "  {} {} {:0.14f} {:0.14f} {:0.10f}"

            if reported_U is None:
                print msg.format(n, N, u, delta_u, learning_rate)
            else:
                print msg.format(n, N, u-reported_U, delta_u, learning_rate)
            
            if np.isclose(delta_u,0,atol=1e-16):
                break

            # Even ADAM gets stuck, slowly decay the learning rate
            learning_rate *= 0.96

        u,c = sess.run([U,coord])
        return u, c



if not os.path.exists(f_h5):
    with h5py.File(f_h5):
        pass

   
with h5py.File(f_h5,'r+') as h5:

    # Load the wikipedia dataset
    df = pd.read_csv("data/wikipedia.csv").set_index("N")    

    for N in range(2, 4000):
        #if N==78: continue

        if str(N) not in h5.keys():
            h5.create_group(str(N))

        g = h5[str(N)]

        if 'coordinates' in g.keys():

            if N not in df.index:
                print "Already solved for", N
                continue

            wiki_u = df.ix[N].U_min

            while True:
                c = g['coordinates'][:]
                model_u = (1/pdist(c)).sum()
                delta_u = (model_u - wiki_u)

                if delta_u < 0 or not FLAG_TRY_TO_MATCH_KNOWN:
                    print "Current energy is lower than known", N, delta_u
                    break

                if np.isclose(delta_u,0):
                    print "Current energy is close to known", N
                    break

                print "Energies between known", wiki_u, model_u
            
                u, c = minimize_thompson(N, wiki_u, limit=500)

                dipole_strength = np.linalg.norm(c.sum(axis=0))
                print "Final energy", u, dipole_strength
                g.attrs['energy'] = u

                del g['coordinates']
                g['coordinates'] = c
        
            continue

        else:
            u, c = minimize_thompson(N)
            g.attrs.create('energy', u)
            g['coordinates'] = c
