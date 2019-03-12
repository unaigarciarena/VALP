from Networks import Encoder, Decoder, Generic, Discrete, batch
from ModelDescriptor import recursive_creator
from descriptor import MNMDescriptor
from classes import ModelComponent, InOut
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MNM(object):
    def __init__(self, descriptor, batch_size, inputs, outputs, loss_func_weights, load=None):

        if descriptor.constructed or load:
            self.descriptor = descriptor
        else:
            self.descriptor = recursive_creator(descriptor, 0)

        self.inputs = {}
        self.outputs = {}
        self.components = {}
        self.predictions = {}
        self.initialized = []
        self.input_data = {}
        self.output_data = {}
        self.sess = tf.Session()
        self.optimizer = None
        self.loss_function = 0
        self.loss_function_sample = 0
        self.batch_size = batch_size
        self.example_num = inputs[random.choice(list(inputs.keys()))].shape[0]
        self.loss_weights = loss_func_weights

        for model_input in descriptor.inputs:
            self.add_input(inputs[model_input], model_input)

        for outp in descriptor.outputs:
            self.add_output(outputs[outp], outp)

        if load is not None:
            if "str" in type(load).__name__:
                self.load(load)
            elif load is True:
                self.load()
        else:
            self.initialize()

    def add_input(self, data, inp_id):

        self.inputs[inp_id] = tf.placeholder(tf.float32, [None] + list(data.shape[1:]), name=inp_id)
        self.input_data[inp_id] = data

    def add_output(self, data, outp_id):

        self.outputs[outp_id] = tf.placeholder(tf.float32, [None] + list(data.shape[1:]), name=outp_id)
        self.output_data[outp_id] = data

    def add_component(self, comp, comp_id=None):
        if not comp_id:
            comp_id = str(len(self.components.keys()))
        self.components[comp_id] = comp

    def component_output_by_id(self, ident):
        if ident in self.components:
            return self.components[ident].result
        elif ident in self.inputs:
            return self.inputs[ident]
        elif ident in self.outputs:
            return self.outputs[ident]
        else:
            return None

    def initialize(self):
        aux_pred = {}

        self.recursive_init(self.descriptor.comp_by_input(self.descriptor.outputs), aux_pred)

        self.loss_function = 0
        self.loss_function_sample = 0

        for pred in self.predictions.keys():
            if self.descriptor.outputs[pred].taking.type == "discrete":
                self.loss_function += self.loss_weights[pred] * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions[pred], labels=self.outputs[pred]))
                self.predictions[pred] = tf.reshape(tf.argmax(self.predictions[pred], axis=1), (-1, 1))  # tf.sigmoid(self.predictions[pred])
            elif self.descriptor.outputs[pred].taking.type == "values":
                self.loss_function += self.loss_weights[pred] * tf.losses.mean_squared_error(self.predictions[pred], self.outputs[pred])
            elif self.descriptor.outputs[pred].taking.type == "samples":
                self.loss_function_sample += self.loss_weights[pred] * tf.losses.mean_squared_error(predictions=self.predictions[pred], labels=self.outputs[pred])
                self.loss_function += self.loss_weights[pred] * tf.losses.mean_squared_error(predictions=self.predictions[pred], labels=self.outputs[pred])
                self.predictions[pred] = tf.sigmoid(self.predictions[pred])

        for network in self.descriptor.networks:
            if "Decoder" in type(self.descriptor.comp_by_ind(network).descriptor).__name__:
                self.loss_function += -0.00005 * tf.reduce_sum(1 + self.components[network].z_log_sigma_sq - tf.square(self.components[network].z_mean) - tf.exp(self.components[network].z_log_sigma_sq))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_function)

        self.sess.run(tf.global_variables_initializer())

    def train(self, batch_size, epochs, sync, display_step=1):
        aux_ind = 0
        partial_loss = 0
        for epoch in range(epochs):
            feed_dict = {}

            for inp in self.inputs:
                feed_dict[self.inputs[inp]] = batch(self.input_data[inp], batch_size, aux_ind)
            for output in self.outputs:
                feed_dict[self.outputs[output]] = batch(self.output_data[output], batch_size, aux_ind)

            aux_ind = (aux_ind + batch_size) % self.example_num
            if epoch % sync == 1:
                _, partial_loss = self.sess.run([self.optimizer, self.loss_function_sample], feed_dict=feed_dict)
            else:
                _, partial_loss = self.sess.run([self.optimizer, self.loss_function], feed_dict=feed_dict)
            if epoch % display_step == 1:
                print(epoch, partial_loss)
        return partial_loss

    def predict(self, inputs, intra_preds, new):

        feed_dict = {}
        intra = []
        examples = 1
        for intra_pred in intra_preds:
            intra += [self.components[intra_pred].z_log_sigma_sq, self.components[intra_pred].z_mean]

        for inp in inputs.keys():
            feed_dict[self.inputs[inp]] = inputs[inp]
            examples = inputs[inp].shape[0]
        if new:
            for net in self.components:
                if "Decoder" in type(self.components[net]).__name__:

                    feed_dict[self.components[net].z_log_sigma_sq] = np.reshape(np.random.normal(0, 1, examples*self.components[net].z_log_sigma_sq.shape[1].value), (examples, self.components[net].z_log_sigma_sq.shape[1].value))
                    feed_dict[self.components[net].z_mean] = np.reshape(np.random.normal(0, 1, examples*self.components[net].z_mean.shape[1].value), (examples, self.components[net].z_mean.shape[1].value))

        return self.sess.run([self.predictions] + intra, feed_dict=feed_dict)

    def recursive_init(self, comps, aux_pred):

        for comp in comps:

            if comp not in self.initialized and "i" not in comp:

                self.initialized += [comp]
                comps_below = self.descriptor.comp_by_input(comp)
                self.recursive_init(comps_below, aux_pred)
                net = self.descriptor.comp_by_ind(comp)

                aux_input = []

                for comp_below in comps_below:
                    aux_input += [self.component_output_by_id(comp_below)]
                self.components[comp] = network_types[type(net.descriptor).__name__[:-10]](net.descriptor, aux_input, comp)
                outs = self.descriptor.comp_by_output(comp)
                for out in outs:
                    if "o" in out:
                        if out not in self.predictions.keys():
                            self.predictions[out] = tf.reshape(self.components[comp].result[:, :self.descriptor.outputs[out].taking.size], (-1, self.descriptor.outputs[out].taking.size))
                        else:

                            self.predictions[out] = self.predictions[out] + tf.reshape(self.components[comp].result[:, :self.descriptor.outputs[out].taking.size], (-1, self.descriptor.outputs[out].taking.size))

    def save(self, path="/home/unai/Escritorio/MultiNetwork/model/model"):
        saver = tf.train.Saver()

        saver.save(self.sess, path)
        self.descriptor.save(path + ".txt")

    def load(self, path="/home/unai/Escritorio/MultiNetwork/model/model"):

        self.descriptor.load(path + ".txt")
        self.initialize()
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


def histogram(x):
    # the histogram of the data
    plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    plt.grid(True)

    plt.show()


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

x_train = np.reshape(fashion_mnist[0][0], (-1, 784))/256
c_train = fashion_mnist[0][1]
y_train = np.array([np.histogram(obs, bins=32)[0] for obs in x_train])/784

x_test = np.reshape(fashion_mnist[1][0], (-1, 784))/256
c_test = fashion_mnist[1][1]
y_test = np.array([np.histogram(obs, bins=32)[0] for obs in x_test])/784

# dataset = datasets.fetch_mldata('MNIST original')

# data_y = np.reshape(np.sin(data.data[:, 1]) + data.data[:, 2] * data.data[:, 0] - np.cos(data.data[:, 3]), (-1, 1))  # Manual function
# data_y = np.array([np.histogram(obs, bins=32)[0] for obs in dataset.data])/784
# dataset.data = dataset.data/256

# x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(dataset.data, dataset.target, data_y, random_state=1)

network_types = {"Encoder": Encoder, "Decoder": Decoder, "Generic": Generic, "Discrete": Discrete}

model_inputs = {}

output_placeholders = {}

data_inputs = {}
inp_dict = {}
# Separated inputs
"""
for i in range(iris.data.shape[1]):
    data_inputs["i" + str(i)] = np.reshape(iris.data[:, i], (-1, 1))
    inp_dict["i" + str(i)] = ModelComponent(None, InOut(size=1, type="values"))
"""

# Merged inputs

data_inputs["i0"] = x_train
inp_dict["i0"] = ModelComponent(None, InOut(size=x_train.shape[1], data_type="values"), -1)

OHEnc = OneHotEncoder()

a = OHEnc.fit_transform(np.reshape(c_train, (-1, 1))).toarray()

data_outputs = {"o0": y_train}

outp_dict = {"o0": ModelComponent(InOut(size=y_train.shape[1], data_type="values"), None, 0)}

# Separated one hot encoding
"""
for i in range(a.shape[1]):
    data_outputs["o" + str(i+1)] = np.reshape(a[:, i], [-1, 1])
    outp_dict["o" + str(i+1)] = ModelComponent(InOut(size=1, type="values"), None)
"""
# Merged one hot encoding
data_outputs["o1"] = a
outp_dict["o1"] = ModelComponent(InOut(size=a.shape[1], data_type="discrete"), None, 0)

# Samples

data_outputs["o2"] = x_train
outp_dict["o2"] = ModelComponent(InOut(size=x_train.data.shape[1], data_type="samples"), None, 0)

btch_sz = 50
loss_weights = {"o0": 1, "o1": 1, "o2": 1}

accs = []
mses = []
images = []
save = True
iters = 40000

for seed in range(475, 501):

    print("Seed:" + str(seed))
    print("plz lern sumthin")
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    tf.reset_default_graph()

    model_descriptor = MNMDescriptor(10, inp_dict, outp_dict)

    model = MNM(model_descriptor, btch_sz, data_inputs, data_outputs, loss_weights)
    model.descriptor.print_model_graph("huehue" + str(seed))
    loss = model.train(btch_sz, iters, 5)
    # model.save()

    # a, = model.predict({"i0": x_test}, [], new=False)
    a, = model.predict({"i0": x_test}, [], new=True)
    if save:
        mses += [mean_squared_error(a["o0"], y_test)]
        accs += [accuracy_score(a["o1"], c_test)]
        images += [np.reshape(a["o2"], (10000, 784))]

        if seed % 1 == 0:
            np.save("accuraciesTopAcc" + str(seed) + ".npy", accs)
            np.save("MSEsTopAcc" + str(seed) + ".npy", mses)
            np.save("imagesTopAcc" + str(seed) + ".npy", images)
            np.save("condsTopAcc" + str(seed) + ".npy", c_test)
            accs = []
            mses = []
            images = []

    model.sess.close()
