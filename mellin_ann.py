import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import binom

"""
The class PDFANN implements the ANN that parametrizes the PDF.
The ANN has only one hidden layer, one input feature x and one output variable y = pdf(x)
"""
class PDFANN:
    def __init__(self, neurons, moments, mesh=100, drop=0.01, alpha=0.01, genetic=False, num_gen=500, sol_per_pop=100, mut_per=0.1):
        """
        :param neurons: number of neurons in the hidden layer
        :param moments: vector containing the first even n Mellin moments corresponding to an even function defined in [-0.5, 0.5]
        :param mesh: number of intervals for the pdf domain partition
        :param drop: rate for the dropout regularization
        :param alpha: parameter for the L2 regualarization
        :param genetic: if True an optimization using a genetic algorithm is implemented (work in progress). If False
                        gradient descent is implemented (currently only genetic=False is working).
        :param num_gen: number of generations for the genetic algorithm
        :param sol_per_pop: population size for the genetic algorithm
        :param mut_per: mutation percentage rate for the genetic algorithm
        """
        self.neurons = neurons
        self.L = 3 # total number of layers (input + hidden + output)
        self.n_moments = len(moments)
        if mesh % 2 == 0:
            mesh += 1
        self.mesh = mesh
        self.drop = drop
        self.alpha = alpha

        if genetic:
            self.W = []
            self.b = []
            self.vec_wb = []
            self.mat_neuron = []
            self.sol_per_pop = sol_per_pop
            self.num_gen = num_gen
            self.mut_per = mut_per
        else:
            self.W = {} # dictionary containing the weights. W[1] contains the weights connected to the input and W[2]
                        # the weights connected to the output
            self.b = {} # dictionary containing the biases. b[1] contains the biases corresponding to the hidden neurons
            self.dW = {} # dictionary containing the weight gradients
            self.db = {} # dictionary containing the biases gradients

        self.moments_weights = np.zeros(self.n_moments) # weights coefficients that multiply the momentum residuals in the
                                                        # loss function
        for n in range(self.n_moments):
            self.moments_weights[n] = (1+n)/4**n#10**(2*n)

        self.domain = np.linspace(-0.5, 0.5, mesh)
        self.integrand_power = self.build_integrand_power() # matrix (n_moments x mesh) that contains the nth-power of x
                                                            # for the moments calculation, evaluated in the nodes of the mesh
        self.domain = tf.convert_to_tensor(self.domain, dtype=tf.float64)
        self.moments = tf.convert_to_tensor(moments, dtype=tf.float64)
        self.moments_weights = tf.convert_to_tensor(self.moments_weights, dtype=tf.float64)
        self.h = self.domain[1]-self.domain[0] # mesh spacing
        if genetic:
            self.setup_genetic()
        else:
            self.setup()

    def build_integrand_power(self):
        matrix = np.zeros((self.n_moments, self.mesh))
        for n in range(self.n_moments):
            matrix[n] = self.domain**(2*n)

        return tf.convert_to_tensor(matrix, dtype=tf.float64)

    def setup(self):
        """
        It initializes the weights and biases with random values
        """
        self.W[1] = tf.Variable(tf.random.uniform(minval=-5, maxval=5, shape=(1, self.neurons), dtype=tf.float64))
        self.b[1] = tf.Variable(tf.random.uniform(minval=-5, maxval=5, shape=(1, self.neurons), dtype=tf.float64))
        self.W[2] = tf.Variable(tf.random.uniform(minval=-5, maxval=5, shape=(self.neurons, 1), dtype=tf.float64))

    def setup_genetic(self):
        W_pop = []
        b_pop = []
        for n in range(self.sol_per_pop):
            W_curr = {}
            b_curr = {}
            W_curr[1] = (np.random.uniform(low=-5, high=5, size=(1, self.neurons)))
            b_curr[1] = (np.random.uniform(low=-5, high=5, size=(1, self.neurons)))
            W_curr[2] = (np.random.uniform(low=-5, high=5, size=(self.neurons, 1)))
            W_pop.append(W_curr)
            b_pop.append(b_curr)

        self.W = np.array(W_pop)
        self.b = np.array(b_pop)
        self.vec_wb = self.mat_to_vector(self.W, self.b)
        self.mat_neuron = self.mat_to_mat_neuron()

    def mat_to_mat_neuron(self):
        pop_mat_neuron = []
        for sol_idx in range(self.sol_per_pop):
            curr_mat_neuron = []
            for neuron in range(self.neurons):
                curr_mat_neuron.append([self.W[sol_idx][1][0, neuron], self.b[sol_idx][1][0, neuron],
                                        self.W[sol_idx][2][neuron, 0]])
            pop_mat_neuron.append(curr_mat_neuron)
        return np.array(pop_mat_neuron)

    def mat_neuron_to_mat(self, pop_mat_neuron):
        for sol_idx in range(self.sol_per_pop):
            self.W[sol_idx][1][0,:] = pop_mat_neuron[sol_idx][:,0]
            self.b[sol_idx][1][0,:] = pop_mat_neuron[sol_idx][:,1]
            self.W[sol_idx][2][:,0] = pop_mat_neuron[sol_idx][:,2]


    def mat_to_vector(self, mat_pop_weights, mat_pop_biases):
        pop_weights_vector = []
        for sol_idx in range(self.sol_per_pop):
            curr_vector = []
            for layer_idx in range(1, self.L):
                vector_weights = np.reshape(mat_pop_weights[sol_idx][layer_idx],
                                               newshape=(mat_pop_weights[sol_idx][layer_idx].size))
                curr_vector.extend(vector_weights)
                #if layer_idx != self.L-1:
                vector_biases = np.reshape(mat_pop_biases[sol_idx][layer_idx],
                                               newshape=(mat_pop_biases[sol_idx][layer_idx].size))
                curr_vector.extend(vector_biases)
            pop_weights_vector.append(curr_vector)
        return np.array(pop_weights_vector)

    def normal_factor(self):
        """
        :return: the zeroth Mellin moment of the pdf, which constitutes the normalization factor for the ANN output.
                 The integral is approximated using the Boole's rule
        """
        coeff = np.ones(self.mesh)
        #coeff[1:-1:2] = 4
        #coeff[2:-2:2] = 2
        coeff[0] = 7.
        coeff[-1] = 7.
        coeff[1:-1:2] = 32.
        coeff[2:-2:4] = 12.
        coeff[4:-4:4] = 14.
        coeff = tf.convert_to_tensor(coeff, dtype=tf.float64)
        forward = self.forward_pass()
        #print(forward)
        normal = tf.tensordot(forward, coeff, axes=1)
        normal = tf.math.multiply(normal, self.h*2./45.)
        #print(normal)
        return normal

    def pdf(self):
        result = self.forward_pass()
        return tf.math.divide(result, self.normal_factor())

    def forward_pass(self):
        """
        :return: the output of the ANN after passing through the hidden layer. The different terms are combined in order
                 to implement the vanishing of the pdf at the boundaries of the domain and the parity under x <--> -x
        """
        x = tf.expand_dims(self.domain, axis=1)
        hidden1 = tf.nn.sigmoid(tf.matmul(x, self.W[1]) + self.b[1])
        hidden2 = tf.nn.sigmoid(tf.matmul(-x, self.W[1]) + self.b[1])
        hidden3 = tf.nn.sigmoid(self.W[1]/2. + self.b[1])
        hidden4 = tf.nn.sigmoid(-self.W[1]/2. + self.b[1])
        hidden = hidden1 + hidden2 - hidden3 - hidden4
        hidden = tf.nn.dropout(hidden, rate=self.drop)
        forward = tf.matmul(hidden, self.W[2])
        return tf.math.abs(tf.squeeze(forward))

    def compute_mellin(self):
        """
        :return: the first n_moments Mellin moments calculated from the ANN output, using the Boole's rule for the integration
        """
        coeff = np.ones(self.mesh)
        #coeff[1:-1:2] = 4
        #coeff[2:-2:2] = 2
        coeff[0] = 7.
        coeff[-1] = 7.
        coeff[1:-1:2] = 32.
        coeff[2:-2:4] = 12.
        coeff[4:-4:4] = 14.
        coeff = tf.convert_to_tensor(coeff, dtype=tf.float64)
        moment0 = tf.tensordot(self.forward_pass(), coeff, axes=1)
        moment0 = tf.math.multiply(moment0, self.h*2/45.)
        integrand = tf.math.multiply(self.integrand_power, self.forward_pass())
        moments = tf.tensordot(integrand, coeff, axes=1)
        moments = tf.math.multiply(moments, self.h*2/45.)
        moments = tf.math.divide(moments, moment0)
        return moments

    def compute_loss(self, y_pred, y_true):
        """
        :param y_pred: vector containing the Mellin moments predicted by the ANN
        :param y_true: vector containing the true Mellin moments
        :return: the loss function to be minimized by the optimization algorithm
        """
        l2 = tf.Variable(0., dtype=tf.float64)
        l2.assign(tf.math.reduce_sum(self.W[1]**2) + tf.math.reduce_sum(self.W[2]**2) + tf.math.reduce_sum(self.b[1]**2))
        # l2 is the L2 norm of the weights to be added as a regularization term (not actually needed)
        loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.multiply(tf.math.square((y_pred-y_true)/y_true),self.moments_weights))
                            + self.alpha*l2)
               #+ tf.math.square(self.normal_factor() - 1) + self.alpha*l2)
        return loss


    def back_prop(self, lr):
        """
        computes the gradient of the loss function and updates the weights and biases
        :param lr: learning rate for the gradient descent optimization algorithm
        :return: the loss function to be displayed
        """
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        with tf.GradientTape(persistent=True) as tape:
            moments_pred = self.compute_mellin()
            loss = self.compute_loss(moments_pred, self.moments)
        gradients = []
        params = []
        self.dW[1] = tape.gradient(loss, self.W[1])
        gradients.append(self.dW[1])
        params.append(self.W[1])
        self.dW[2] = tape.gradient(loss, self.W[2])
        gradients.append(self.dW[2])
        params.append(self.W[2])
        self.db[1] = tape.gradient(loss, self.b[1])
        gradients.append(self.db[1])
        params.append(self.b[1])
        opt.apply_gradients(zip(gradients, params))
        del tape
        return loss.numpy()


    def fit(self, epochs=100, lr=0.001):
        for epoch in range(epochs):
            loss = self.back_prop(lr)
            print('Epoch: ',  epoch, '\t loss: ', loss)


    def fit_genetic(self):
        pass


def mellin_moment(f, n, *params):
    """
    :param f: function for which the n-th Mellin moment is calculated (defined in [0, 1] and even under x <--> 1-x)
    :param n: integer order of the Mellin moment
    :param params: optional parameters belonging to the function f
    :return: the n-th Mellin moment of the function f
    """
    if len(params):
        return quad(lambda x: (x**n + (1-x)**n)*f(x, *params), 0, 1/2)[0]
    else:
        return quad(lambda x: (x**n + (1-x)**n)*f(x), 0, 1/2)[0]


def mellin_moment_shift(f, n, *params):
    """
    the same function as the previous one, but the moments are calculated for the shifted function centered in zero.
    The even moment 2n is calculated
    """
    if len(params):
        return 2*quad(lambda x: (x-0.5)**(2*n)*f(x, *params), 0, 1/2)[0]
    else:
        return 2*quad(lambda x: (x-0.5)**(2*n)*f(x), 0, 1/2)[0]


def train_func1(x, *params):
    """
    trial function used to generate the moments to train the ANN
    :param x: x value
    :param params: optional parameters as part of the functional form
    :return: function value
    """
    norm = 2*quad(lambda x: np.log(1 + x**2*(1-x)**2/params[0]**2), 0, 1/2)[0]
    return np.log(1 + x**2*(1-x)**2/params[0]**2)/norm


def train_func2(x, *params):
    norm = 2*quad(lambda x: x**params[0]*(1-x)**params[0], 0, 1/2)[0]
    return x**params[0]*(1-x)**params[0]/norm


def get_moments_from_func(n_moments, func, *params, shift):
    """
    Generate the first n_moments Mellin moments of the function f
    :param n_moments: number of the first Mellin moments
    :param func: function for which the Mellin moments are calculated
    :param params: optional parameters as part of the function definition
    :param shift: if True the shifted moments, corresponding to the function centered in zero, are calculated
    :return: a vector containing the values of the moments
    """
    moments = np.zeros(n_moments)
    if shift:
        for n in range(n_moments):
            moments[n] = mellin_moment_shift(func, n, *params)
    else:
        for n in range(n_moments):
            moments[n] = mellin_moment(func, n, *params)
    return moments


def transform_moments(moments):
    """
    :param moments: vector containing the Mellin moments corresponding to the function defined in [0,1]
    :return: a vector containing the shifted moments corresponding to the shifted function in [-0.5, 0.5]
    """
    transform_matrix = np.zeros((len(moments), len(moments)))
    for n in range(len(moments)):
        for i in range(len(moments)):
            if n >= i:
                transform_matrix[n, i] = (-0.5)**(n-i)*binom(n,i)

    return np.matmul(transform_matrix, moments)[::2]


def func_points(func, n_points, params):
    """
    :return: a vector with the values of the function func, evaluated in equidistant n_points in [0, 1] and for a list of
             different parameters
    """
    x = np.linspace(0, 1, n_points)
    y = []
    for param in params:
        y_param = []
        for point in x:
            y_param.append(func(point, *param))
        y.append(y_param)
    y = np.array(y)
    return y


def visual_test(func, net, *params, n_interp=200):
    """
    It plots the ANN pdf vs. the known function that generated the Mellin moments used to train it
    :param func: function that generated the Mellin moments
    :param net: ANN that parametrizes the pdf
    :param params: optional parameters as part of the function definition
    :param n_interp: number of points for the numerical interpolation
    """
    params_list = []
    params_sublist = []
    for param in params:
        params_sublist.append(param)
    params_list.append(params_sublist)
    x = np.linspace(0,1,net.mesh)
    y_true = func_points(func, n_interp, params_list).reshape(-1)
    y = net.pdf()
    interp = interp1d(x, y, kind='linear')
    x_interp = np.linspace(0,1,n_interp)
    y_interp = np.zeros_like(x_interp)
    for n in range(n_interp):
        y_interp[n] = interp(x_interp[n])

    plt.plot(x_interp, y_true, linewidth=1.5, label='Exact')
    plt.plot(x_interp, y_interp, linewidth=1.5, label='ANN')
    plt.grid(alpha=0.5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()


param = 3.1
moments = get_moments_from_func(5, train_func2, param, shift=True)
#moments = get_moments_from_func(10, train_func2, param, shift=False)
#moments = transform_moments(moments)
net = PDFANN(350, moments, mesh=350, drop=0.0, alpha=0.0)
learning_rates = [0.001, 0.0001, 0.001]
learning_rate_boundaries = [700, 2200]
learning_rates_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)
net.fit(epochs=3000, lr=0.001)
moments_pred = net.compute_mellin().numpy()
print(moments_pred, moments)
print(np.abs((moments_pred-moments)/moments))
print(net.normal_factor())
visual_test(train_func2, net, param, n_interp=1000)

