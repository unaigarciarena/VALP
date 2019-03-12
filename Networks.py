import numpy as np
import tensorflow as tf
from descriptor import EncoderDescriptor, DecoderDescriptor, GeneratorDescriptor, GenericDescriptor, DiscreteDescriptor, init_functions, act_functions, xavier_init
import math

class Network:
    def __init__(self, network_descriptor, ident):
        self.id = ident
        self.descriptor = network_descriptor
        self.List_layers = []
        self.List_weights = []
        self.List_bias = []

    def reset_network(self):
        self.List_layers = []
        self.List_weights = []
        self.List_bias = []

    @staticmethod
    def create_hidden_layer(in_size, out_size, init_w_function, layer_name):
        if "uniform" in init_w_function.__name__:
            w = tf.Variable(init_w_function(shape=[in_size, out_size], minval=-0.1, maxval=0.1), name="W"+layer_name)
        elif "normal" in init_w_function.__name__:
            w = tf.Variable(init_w_function(shape=[in_size, out_size], mean=0, stddev=0.03), name="W"+layer_name)
        else:
            w = tf.Variable(init_w_function(shape=[in_size, out_size]), name="W"+layer_name)
        b = tf.Variable(tf.zeros(shape=[out_size]), name="b"+layer_name)
        return w, b

    def network_initialization(self):
        for lay in range(self.descriptor.number_hidden_layers+1):
            init_w_function = self.descriptor.List_init_functions[lay]
            if lay == 0:
                in_size = self.descriptor.input_dim
                out_size = self.descriptor.List_dims[lay]
            elif lay == self.descriptor.number_hidden_layers:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.output_dim
            else:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.List_dims[lay]

            w, b = self.create_hidden_layer(in_size, out_size, init_w_function, str(lay))

            self.List_weights.append(w)
            self.List_bias.append(b)

    def network_evaluation(self, layer):

        for lay in range(self.descriptor.number_hidden_layers+1):
            w = self.List_weights[lay]
            b = self.List_bias[lay]
            act = self.descriptor.List_act_functions[lay]
            if act is None:
                layer = tf.add(tf.matmul(layer, w), b)
            else:
                if lay == self.descriptor.number_hidden_layers:
                    layer = tf.matmul(layer, w) + b
                else:
                    layer = act(tf.add(tf.matmul(layer, w), b))

            self.List_layers.append(layer)

        return layer

    def variables(self):

        tensors = {}
        for ind, var in enumerate(self.List_bias):
            tensors[self.id + "-B-" + str(ind)] = var
        for ind, var in enumerate(self.List_weights):
            tensors[self.id + "-B-" + str(ind)] = var
        return tensors


class Encoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    This implementation uses probabilistic-multi-layer-perceptron encoders and decoders using Gaussian
    distributions. The VAE can be learned end-to-end.
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, descriptor, inp):
        prior_networks = {"Normal": self._create_normal_network, "Kumaraswamy": self._create_stick_breaking_network}

        self.network_architecture = descriptor

        self.x = inp

        # Create autoencoder network
        prior_networks[self.network_architecture.prior]()

    def _create_normal_network(self, ident):
        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z = self._recognition_network(ident)

        eps = tf.random_normal(shape=tf.shape(self.z[0]), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.result = tf.add(self.z[0], tf.multiply(tf.sqrt(tf.exp(self.z[1])), eps))

    def _create_stick_breaking_network(self, ident):
        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z = self._recognition_network(ident)

        eps = tf.random_uniform(shape=tf.shape(self.z[0]), minval=0.01, maxval=0.99, dtype=tf.float32)
        self.result = (1-(eps**(1/self.z[0])))**(1/self.z[1])

    def _recognition_network(self, ident):
        # Generate the probabilistic encoder (recognition network), which
        # maps inputs to the latent space.
        # The transformation is parametrized and can be learned.

        recog_descriptor = self.network_architecture.Enc_network

        w0 = tf.Variable(xavier_init(recog_descriptor.output_dim, self.network_architecture.z_dim))
        w1 = tf.Variable(xavier_init(recog_descriptor.output_dim, self.network_architecture.z_dim))
        b0 = tf.Variable(tf.zeros([self.network_architecture.z_dim], dtype=tf.float32)),
        b1 = tf.Variable(tf.zeros([self.network_architecture.z_dim], dtype=tf.float32))

        self.recog_network = Network(recog_descriptor, ident)
        self.recog_network.network_initialization()

        layer = self.recog_network.network_evaluation(self.x)
        z0 = tf.add(tf.matmul(layer, w0), b0)
        z1 = tf.add(tf.matmul(layer, w1), b1)
        return z0, z1

    # ########## Extracted from Eric Nalisnick's VAE implementation:
    # https://github.com/enalisnick/stick-breaking_dgms/blob/master/models/variational_coders/encoders.py

    @staticmethod
    def beta_fn(a, b):
        """
        :return: Beta(a, b)
        """
        return tf.exp(tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a+b))

    def calc_kl_divergence(self, prior_alpha, prior_beta):

        # compute taylor expansion for E[log (1-v)] term
        # hard-code so we don't have to use Scan()
        kl = 1./(1 + self.z[0]*self.z[1]) * self.beta_fn(1. / self.z[0], self.z[1])
        kl += 1./(2 + self.z[0]*self.z[1]) * self.beta_fn(2. / self.z[0], self.z[1])
        kl += 1./(3 + self.z[0]*self.z[1]) * self.beta_fn(3. / self.z[0], self.z[1])
        kl += 1./(4 + self.z[0]*self.z[1]) * self.beta_fn(4. / self.z[0], self.z[1])
        kl += 1./(5 + self.z[0]*self.z[1]) * self.beta_fn(5. / self.z[0], self.z[1])
        kl += 1./(6 + self.z[0]*self.z[1]) * self.beta_fn(6. / self.z[0], self.z[1])
        kl += 1./(7 + self.z[0]*self.z[1]) * self.beta_fn(7. / self.z[0], self.z[1])
        kl += 1./(8 + self.z[0]*self.z[1]) * self.beta_fn(8. / self.z[0], self.z[1])
        kl += 1./(9 + self.z[0]*self.z[1]) * self.beta_fn(9. / self.z[0], self.z[1])
        kl += 1./(10 + self.z[0]*self.z[1]) * self.beta_fn(10. / self.z[0], self.z[1])
        kl *= (prior_beta-1)*self.z[1]

        # use another taylor approx for Digamma function
        psi_b_taylor_approx = tf.log(self.z[1]) - 1./(2 * self.z[1]) - 1./(12 * self.z[1]**2)
        kl += (self.z[0]-prior_alpha)/self.z[0] * (-0.57721 - psi_b_taylor_approx - 1/self.z[1])  # T.psi(self.posterior_b)

        # add normalization constants
        kl += tf.log(self.z[0]*self.z[1]) + tf.log(self.beta_fn(prior_alpha, prior_beta))

        # final term
        kl += -(self.z[1]-1)/self.z[1]

        return 0.1*tf.reduce_sum(kl, axis=1)

    ###################################################


class Decoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    This implementation uses probabilistic-multi-layer-perceptron encoders and decoders using Gaussian
    distributions. The VAE can be learned end-to-end.
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, descriptor, inp, ident):

        self.descriptor = descriptor

        self.network = None

        self.input = inp
        self.cond = []
        self.z = []

        for i in range(len(inp)):
            if i in self.descriptor.rands:
                self.z += [inp[i]]
            else:
                self.cond += [inp[i]]

        self.z = tf.concat(self.z, axis=1)

        self.result = self._generator_network(ident)

    def _generator_network(self, ident):

        self.z_mean = self.z[:, :math.floor(self.z.shape[1].value/2)]  # tf.add(tf.matmul(self.z, self.w_mean), self.b_mean)
        self.z_log_sigma_sq = self.z[:, math.ceil(self.z.shape[1].value/2):]  # tf.add(tf.matmul(self.z, self.w_log_sigma), self.b_log_sigma)
        eps = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z_samples = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.z_samples_cond = tf.concat([self.z_samples] + self.cond, axis=1)
        self.descriptor.network.input_dim = self.z_samples_cond.shape[1].value
        self.network = Network(self.descriptor.network, ident)

        self.network.network_initialization()

        result = self.network.network_evaluation(self.z_samples_cond)

        return result

    def variables(self):
        tensors = self.network.variables()

        return tensors


class Generic(object):
    def __init__(self, descriptor, inp, ident):
        self.descriptor = descriptor
        self.network = None
        self.input = inp
        self.x = tf.concat(inp, axis=1)
        self.result = self.create_network(ident)

    def create_network(self, ident):
        self.network = Network(self.descriptor.network, ident)
        self.network.network_initialization()

        return self.network.network_evaluation(self.x)

    def variables(self):
        return self.network.variables()


class Discrete(object):
    def __init__(self, descriptor, inp, ident):
        self.descriptor = descriptor
        self.network = None
        self.input = inp
        self.x = tf.concat(inp, axis=1)
        self.result = self.create_network(ident)

    def create_network(self, ident):
        self.network = Network(self.descriptor.network, ident)
        self.network.network_initialization()
        return self.network.network_evaluation(self.x)

    def variables(self):
        return self.network.variables()


def random_batch(x, y, size):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param y: Fitness scores of the population, intended to be fed to the net in the output
    :param size: Size of the batch desired
    :return: A random batch of the data (x, y) of the selected size
    """
    indices = np.random.randint(x.shape[0], size=size)
    return x[indices, :], y[indices]


def batch(x, size, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array
        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((x[i:, :], x[:index, :]))
    else:  # Easy case
        index = i+size
        return x[i:index, :]


def encoder_descriptor(x_dim, z_dim):

    priors = ["Normal", "Kumaraswamy"]

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)  # np.random.randint(nlayers)+1                             # Number of hidden layers

    encoder_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for encoder

    encoder_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    encoder_act_functions[n_hidden] = None

    prior = np.random.choice(priors)

    my_descriptor = EncoderDescriptor(x_dim, z_dim, prior)

    my_descriptor.initialization(n_hidden, dim_list, encoder_init_functions,  encoder_act_functions)

    return my_descriptor


def decoder_descriptor(z_dim, x_dim, n_inputs):

    # tf.reset_default_graph()

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)

    decoder_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for decoder

    decoder_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    decoder_act_functions[n_hidden] = None

    my_vae_descriptor = DecoderDescriptor(z_dim, x_dim, n_inputs)

    my_vae_descriptor.initialization(n_hidden, dim_list, decoder_init_functions,  decoder_act_functions)

    return my_vae_descriptor


def generator_descriptor(x_dim, z_dim):

    # tf.reset_default_graph()

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)

    measures = ["Standard_Divergence", "Total_Variation", "Forward_KL", "Reverse_KL", "Pearson_Chi_squared", "Squared_Hellinger", "Least_squared"]
    fmeasure = np.random.choice(measures)

    generator_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for encoder

    generator_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    generator_act_functions[n_hidden] = None

    my_vae_descriptor = GeneratorDescriptor(x_dim, z_dim, fmeasure)

    my_vae_descriptor.initialization(n_hidden, dim_list, generator_init_functions,  generator_act_functions)

    return my_vae_descriptor


def generic_descriptor(input_dim, output_dim):

    # tf.reset_default_graph()

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)

    generic_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for encoder
    generic_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    generic_act_functions[n_hidden] = None

    my_descriptor = GenericDescriptor(input_dim, output_dim, 0)

    my_descriptor.initialization(n_hidden, dim_list, generic_init_functions,  generic_act_functions)

    return my_descriptor


def discrete_descriptor(input_dim, output_dim):

    # tf.reset_default_graph()

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)

    generic_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for encoder
    generic_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    generic_act_functions[n_hidden] = tf.nn.softmax

    my_descriptor = DiscreteDescriptor(input_dim, output_dim, 0)

    my_descriptor.initialization(n_hidden, dim_list, generic_init_functions,  generic_act_functions)

    return my_descriptor


def create_descriptor(tp, in_size, out_size):
    types = {"Encoder": encoder_descriptor, "Decoder": decoder_descriptor, "Generator": generator_descriptor, "Generic": generic_descriptor}
    desc = types[tp]

    return desc(in_size, out_size)
