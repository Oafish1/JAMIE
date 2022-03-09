import tensorflow as tf
import numpy as np
import argparse

tf.compat.v1.disable_eager_execution()

def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  """Computes a Guassian Radial Basis Kernel between the samples of x and y.
  We create a sum of multiple gaussian kernels each having a width sigma_i.
  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix, bandwidth=1.0):
  """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x, tf.constant([bandwidth])))
    cost += tf.reduce_mean(kernel(y, y, tf.constant([bandwidth])))
    cost -= 2 * tf.reduce_mean(kernel(x, y, tf.constant([bandwidth])))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost

## parse arguments
parser = argparse.ArgumentParser(description='''MMD-MA implementation.\n
Input: (n by n) similarity matrices in tsv format\n
Output: (n by p) mapping matrices''')
parser.add_argument('mat1', type=str, help='input similarity matrix1 in tsv format')
parser.add_argument('mat2', type=str, help='input similarity matrix2 in tsv format')
parser.add_argument('--l1', type=float, help='lambda1 in loss function, 0.01 by default')
parser.add_argument('--l2', type=float, help='lambda2 in loss function, 0.001 by default')
parser.add_argument('--p', type=int, help='embedded dimensions, 2 by default')
parser.add_argument('--bandwidth', type=float, help='Gaussian kernel bandwidth in MMD, 1 by default')
parser.add_argument('--training_rate', type=float, help='training rate for MMD-MA, 0.00005 by default')
parser.add_argument('--seed', type=int, help='random seed, default is 0')
args = parser.parse_args()

f1, f2 = args.mat1, args.mat2

if args.l1:
  tradeoff2 = args.l1
else:
  tradeoff2 = 0.01

if args.l2:
  tradeoff3 = args.l2
else:
  tradeoff3 = 0.001

if args.p:
  p = args.p
else:
  p=2

if args.bandwidth:
  bandwidth = args.bandwidth
else:
  bandwidth = 1.0

if args.training_rate:
  training_rate = args.training_rate
else:
  training_rate = 0.00005

if args.seed:
  k = args.seed
else:
  k = 0

f = open(f1, 'r')
k1_matrix = np.array([[float(num) for num in line.split('\t')] for line in f ])
f = open(f2, 'r')
k2_matrix = np.array([[float(num) for num in line.split('\t')] for line in f ])
I_p=tf.eye(p)
record = open('loss.txt', 'w')
n1 = k1_matrix.shape[0]
n2 = k2_matrix.shape[0]
K1 = tf.constant(k1_matrix, dtype=tf.float32)
K2 = tf.constant(k2_matrix, dtype=tf.float32)
alpha = tf.Variable(tf.random.uniform([n1,p],minval=0.0,maxval=0.1,seed=k))
beta = tf.Variable(tf.random.uniform([n2,p],minval=0.0,maxval=0.1,seed=k))

# myFunction = tradeoff1*maximum_mean_discrepancy(tf.matmul(K1,alpha), tf.matmul(K2,beta)) + tradeoff2*(tf.norm(tf.subtract(tf.matmul(tf.transpose(alpha),tf.matmul(K1,alpha)),I_p),ord=2) + tf.norm(tf.subtract(tf.matmul(tf.transpose(beta),tf.matmul(K2,beta)),I_p),ord=2)) + tradeoff3*(tf.norm(tf.subtract(tf.matmul(tf.matmul(K1,alpha),tf.matmul(tf.transpose(alpha),tf.transpose(K1))),K1),ord=2)+tf.norm(tf.subtract(tf.matmul(tf.matmul(K2,beta),tf.matmul(tf.transpose(beta),tf.transpose(K2))),K2),ord=2))
def myFunction():
    mmd_part = maximum_mean_discrepancy(tf.matmul(K1,alpha), tf.matmul(K2,beta), bandwidth=bandwidth)
    penalty_part = tradeoff2*(tf.norm(tf.subtract(tf.matmul(tf.transpose(alpha),tf.matmul(K1,alpha)),I_p),ord=2) + tf.norm(tf.subtract(tf.matmul(tf.transpose(beta),tf.matmul(K2,beta)),I_p),ord=2))
    distortion_part = tradeoff3*(tf.norm(tf.subtract(tf.matmul(tf.matmul(K1,alpha),tf.matmul(tf.transpose(alpha),tf.transpose(K1))),K1),ord=2)+tf.norm(tf.subtract(tf.matmul(tf.matmul(K2,beta),tf.matmul(tf.transpose(beta),tf.transpose(K2))),K2),ord=2))
    return mmd_part + penalty_part + distortion_part
train_step = tf.compat.v1.train.AdamOptimizer(training_rate).minimize(myFunction)

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

