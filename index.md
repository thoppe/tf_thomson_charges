---- .aligncenter .bg-black .slide-top
@unsplash(RdmLSJR-tq8) .light

@h1 Stupid TensorFlow Tricks
@h3 A new take on an old (Thomson) problem
  
@footer @div .wrap @div .span
 @button(href="https://github.com/thoppe/tf_thomson_charges") .alignleft .ghost
   ::github:: Project repo
 @button(href="https://medium.com/towards-data-science/stupid-tensorflow-tricks-3a837194b7a0") .alignleft .ghost
   ::medium:: Medium post
 @button(href="https://twitter.com/metasemantic") .ghost .alignright
   ::twitter:: @metasemantic 

----  .bg-white .slide-top

@h1 What is TensorFlow?

@h4 .grid 
     | Open source software library for machine learning, released by Google. Synonymous with "deep-learning". Computation lives on a graph and analytic derivatives are possible.
     | @figure(src="images/TensorFlowLogo.png")
     | @figure(src="http://riosyvalles.com/wp/wp-content/uploads/2016/10/graph2.png")

----  .bg-black .slide-top

@h1 What is the Thomson problem?
@h2 $U = \sum_{i}^N \sum_{i>j}^N \frac{1}{r_{ij}}$

@h4 .grid 
     | The objective of the [Thomson problem](https://en.wikipedia.org/wiki/Thomson_problem) is to determine the minimum electrostatic potential energy configuration of N electrons constrained to the surface of a unit sphere that repel each other with a force given by Coulomb's law.
     | @figure(src="images/N_2_to_5_ThomsonSolutions.png" height=400px)

---- .wrap

@h2 
     + Can we use TensorFlow to
     + solve the Thomson problem?
     
Spoiler alert: Yes!

---- .wrap

@h3 Build a model
```
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

    # Electrostatic potential up to a constant, 1/r
    U = tf.reduce_sum(1/R)

    return U, coord
```

---- .wrap
@h3 Minimize the energy

```
def minimize_thompson(N):
    
    U, coord = thompson_model(N)

    learning_rate = 0.1
    LR = tf.placeholder(tf.float64, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=LR).minimize(U)
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for n in xrange(50):
            for _ in range(100):
                sess.run(opt, feed_dict={LR:learning_rate})

            # Even ADAM gets stuck, slowly decay the learning rate
            learning_rate *= 0.96

```
---- .wrap .slide-top

@h3 Make pictures
.grid
	| @figure(src="images/charges2.gif") 100 charges
	| @figure(src="images/charges1.gif") 625 charges
	
---- .bg-black

.wrap 
   @h3 Does it always work?
   **NO!** When N gets large there are an exponentially large amount of stable configurations that aren't the minima.
   <br><br>
   @h3 Does TensorFlow make it faster?
   **YES!** TensorFlow on a GPU is 10x faster than the 4-cores on a CPU (tested at N=4000)
   <br><br>
   @h3 Is it useful?
   **EH.** The solution to the Thomson problem isn't exciting _per se_ but new solutions often give insight into minimization problems. This is more of an example of a novel use of TensorFlow.





----- .bg-apple

# .text-data Thanks, you!
<br><br>

@h2
  + Thoughts? Ideas? 
  + [@metasemantic](href="https://twitter.com/metasemantic")
<br><br>
@h2
  + Want to read more? Read on medium.com:
  + [Stupid TensorFlow Tricks](https://medium.com/towards-data-science/stupid-tensorflow-tricks-3a837194b7a0)

  

