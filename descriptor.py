import numpy as np
import copy
from classes import Connection, InOut, NetworkComp, ModelComponent
import pygraphviz
import tensorflow as tf
import os


def xavier_init(fan_in=None, fan_out=None, shape=None, constant=1):
    """ Xavier initialization of network weights"""
    if fan_in is None:
        fan_in = shape[0]
        fan_out = shape[1]
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


init_functions = np.array([xavier_init, tf.random_uniform, tf.random_normal])
act_functions = np.array([None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh])


class MNMDescriptor(object):
    def __init__(self, max_comp, model_inputs, model_outputs, load=None):
        """
        :param max_comp: Number of components allowed in the model
        :param model_inputs: List of inputs of the model (InOut objects)
        :param model_outputs: List of outputs of the model (InOut objects)
        """
        if load is not None:
            self.load(load)
        else:
            if type(max_comp) is not int:
                raise Exception("The number of components should be an integer")
            if type(model_inputs) is not dict:
                raise Exception("Inputs of the model should be given in a dictionary; id in the key and size in the content")
            if type(model_outputs) is not dict:
                raise Exception("Outputs of the model should be given in a dictionary; id in the key and size in the content")
            self.constructed = False
            self.max_comp = max_comp
            self.networks = {}
            self.connections = []
            self.inputs = model_inputs
            self.active_outputs = list(model_outputs.keys())
            self.outputs = model_outputs
            self.reachable = {}
            for i in list(model_outputs.keys()) + list(model_inputs.keys()):
                self.reachable[i] = []

    def print(self):
        print("Inputs of the model")

        for i in self.inputs:
            print("  " + i + " " + self.inputs[i].print())
        print("##################################")
        print("")
        print("Outputs of the model")
        for i in self.outputs:
            print("  " + i + " " + self.outputs[i].print())
        print("##################################")
        print("")
        print("Networks of the model")
        for i in self.networks:
            print("  " + i + " " + self.networks[i].print())
        print("##################################")
        print("")
        print("Connections of the model")
        for i in self.connections:
            print(i.print())

    def print_model_graph(self, name=None):

        dot = pygraphviz.AGraph(directed="True")
        for outp in list(self.outputs.keys()):
            dot.add_node(outp, pos=(outp[1:] + ",10"), color="red", label=outp + ", " + str(self.outputs[outp].taking.size) + "-" + self.outputs[outp].taking.type)
        for inp in list(self.inputs.keys()):
            dot.add_node(inp, pos=(inp[1:] + ",0"), color="blue", label=inp + ", " + str(self.inputs[inp].producing.size) + "-" + self.inputs[inp].producing.type)
        for comp in list(self.networks.keys()):
            dot.add_node(comp, label=comp + "-" + str(type(self.networks[comp].descriptor).__name__)[:-14] + ":" + str(self.networks[comp].taking.size) + "-" + str(self.networks[comp].producing.size))

        for con in self.connections:
            dot.add_edge(con.input, con.output, label=str(con.info.size) + " " + self.comp_by_ind(con.input).producing.type)
        dot.layout('dot')
        if not name:
            name = str(hash(self))
        dot.draw(name + '.pdf')

    def add_net(self, net, index=None):

        if not type(net) is NetworkComp:
            raise Exception("Don't introduce a plain descriptor, introduce a NetworkComp")

        if not index:
            index = "n" + str(len(self.networks))
        self.networks[index] = net
        return index

    def comp_ids(self):
        return list(self.networks.keys())

    def net_exists(self, net):
        return net.index in self.networks.keys()

    def comp_by_ind(self, i):
        if "i" in i:
            return self.inputs[i]
        if "o" in i:
            return self.outputs[i]
        return self.networks[i]

    def comp_number(self):
        return len(self.networks)

    def conn_number(self):
        return len(self.connections)

    def random_output(self):
        return np.random.choice(list(self.networks.keys())+list(self.outputs.keys()))

    def random_input(self, output):
        try:
            if "o" in output or ("Network" in type(self.comp_by_ind(output)).__name__ and "Decoder" in type(self.comp_by_ind(output).descriptor).__name__):

                aux = np.random.choice([i for i in list(self.networks.keys()) if self.networks[i].producing == self.comp_by_ind(output).taking])
                return aux
            else:

                comps = {**self.networks, **self.inputs}
                aux = np.random.choice([i for i in list(comps.keys()) if i not in self.reachable[output] and not self.conn_exists(i, output) and self.networks[output].taking.type in comps[i].producing.type])
                return aux
        except:
            return -1

    def conn_exists(self, i0, i1):
        for conn in self.connections:
            if conn.input == i0 and conn.output == i1:
                return True
        return False

    def get_depth(self, index):
        return self.comp_by_ind(index).depth

    def random_model_input(self):
        return np.random.choice(list(self.inputs.keys()))

    def add_connection(self, connection):
        self.connections += [connection]

    def active_indices(self):
        return self.active_outputs

    def delete_active_by_index(self, index):
        self.active_outputs = [i for i in self.active_outputs if i != index]

    def connect(self, index1, index2):

        inp = self.comp_by_ind(index1)
        con = Connection(index1, index2, InOut(data_type=inp.producing.type, size=np.random.randint(inp.producing.size)))
        self.add_connection(con)

    def comp_by_input(self, comp):
        ins = []

        for con in self.connections:
            if (type(comp) is dict and con.output in comp) or ("str" in type(comp).__name__ and comp == con.output):
                ins += [con.input]

        return ins

    def comp_by_output(self, comp):
        outs = []
        for con in self.connections:
            if con.input == comp:
                outs += [con.output]
        return outs

    def save(self, name):

        if os.path.isfile(name):
            os.remove(name)
        f = open(name, "w+")
        for ident in self.networks:
            f.write(ident + "_" + self.networks[ident].descriptor.codify_components() + "_" + str(self.networks[ident].taking.size) + "," + self.networks[ident].taking.type + "_" + str(self.networks[ident].producing.size) + "," + self.networks[ident].producing.type + "_" + str(self.networks[ident].depth) + "\n")
        f.write("\n")

        for ident in self.inputs:
            f.write(ident + "_" + str(self.inputs[ident].producing.size) + "_" + self.inputs[ident].producing.type + "_" + str(self.inputs[ident].depth) + "\n")
        f.write("\n")

        for ident in self.outputs:
            f.write(ident + "_" + str(self.outputs[ident].taking.size) + "_" + self.outputs[ident].taking.type + "_" + str(self.outputs[ident].depth) + "\n")
        f.write("\n")
        for con in self.connections:
            f.write(con.codify() + "\n")
        f.close()

    def load(self, name):

        if name == "":
            name = "/home/unai/Escritorio/MultiNetwork/model/model"

        network_descriptors = {"Generic": GenericDescriptor, "Decoder": DecoderDescriptor, "Discrete": DiscreteDescriptor}

        if not os.path.isfile(name):
            return None

        f = open(name, "r+")

        lines = f.readlines()

        i = 0
        while lines[i] != "\n":
            ident, kind, inp, outp, layers, init, act, taking, producing, depth = lines[i].split("_")

            desc = network_descriptors[kind](int(inp), int(outp))

            desc.initialization(len([int(x) for x in layers.split(",")]), [int(x) for x in layers.split(",")], init_functions[[int(x) for x in init.split(",")]],  act_functions[[int(x) for x in act.split(",")]])

            net = NetworkComp(desc, InOut(size=int(taking.split(",")[0]), data_type=taking.split(",")[1]), InOut(data_type=producing.split(",")[1], size=int(producing.split(",")[0])), int(depth))

            self.add_net(net, ident)
            i += 1

        i += 1

        while lines[i] != "\n":

            ident, size, kind, depth = lines[i].split("_")

            self.inputs[ident] = ModelComponent(None, InOut(size=int(size), data_type=kind), int(depth))
            i += 1

        i += 1

        while lines[i] != "\n":

            ident, size, kind, depth = lines[i].split("_")

            self.outputs[ident] = ModelComponent(InOut(size=int(size), data_type=kind), None, int(depth))
            i += 1

        i += 1

        while i < len(lines):

            inp, outp, kind, size = lines[i].split("_")

            self.connections += [Connection(inp, outp, InOut(kind, int(size)))]
            i += 1


class NetworkDescriptor:
    def __init__(self, number_hidden_layers, input_dim, output_dim,  list_dims, list_init_functions, list_act_functions, number_loop_train):
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.List_dims = list_dims
        self.List_init_functions = list_init_functions
        self.List_act_functions = list_act_functions
        self.number_loop_train = number_loop_train

    def copy_from_other_network(self, other_network):
        self.number_hidden_layers = other_network.number_hidden_layers
        self.input_dim = other_network.input_dim
        self.output_dim = other_network.output_dim
        self.List_dims = copy.deepcopy(other_network.List_dims)
        self.List_init_functions = copy.deepcopy(other_network.List_init_functions)
        self.List_act_functions = copy.deepcopy(other_network.List_act_functions)
        self.number_loop_train = other_network.number_loop_train

    def network_add_layer(self, layer_pos, lay_dims, init_w_function, init_a_function):
        """
        Function: network_add_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.List_dims.insert(layer_pos, lay_dims)
        self.List_init_functions.insert(layer_pos, init_w_function)
        self.List_act_functions.insert(layer_pos, init_a_function)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers + 1

    """
    Function: network_remove_layer()
    Adds a layer at a specified position, with a given  number of units, init weight
    function, activation function.
    If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
    other layers are shifted.
    If the layer is inserted in position number_hidden_layers+1, then it is just appended
    to previous layer and it will output output_dim variables.
    If the position for the layer to be added is not within feasible bounds
    in the current architecture, the function silently returns
    """

    def network_remove_layer(self, layer_pos):

        # If not within feasible bounds, return
        if layer_pos <= 1 or layer_pos > self.number_hidden_layers:
            return

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it

        # We delete the layer in pos layer_pos
        self.List_dims.pop(layer_pos)
        self.List_init_functions.pop(layer_pos)
        self.List_act_functions.pop(layer_pos)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers - 1

    def network_remove_random_layer(self):
        layer_pos = np.random.randint(self.number_hidden_layers)
        self.network_remove_layer(layer_pos)

    def change_activation_fn_in_layer(self, layer_pos, new_act_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.List_act_functions[layer_pos] = new_act_fn

    def change_weight_init_fn_in_layer(self, layer_pos, new_weight_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.List_init_functions[layer_pos] = new_weight_fn

    def change_all_weight_init_fns(self, new_weight_fn):
        # If not within feasible bounds, return
        for layer_pos in range(self.number_hidden_layers):
            self.List_init_functions[layer_pos] = new_weight_fn

    def change_dimensions_in_layer(self, layer_pos, new_dim):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        # If the dimension of the layer is identical to the existing one, return
        self.List_dims[layer_pos] = new_dim

    def change_dimensions_in_random_layer(self, max_layer_size):
        layer_pos = np.random.randint(self.number_hidden_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos, new_dim)

    def print_components(self, identifier):
        print(identifier, ' n_hid:', self.number_hidden_layers)
        print(identifier, ' Dims:', self.List_dims)
        print(identifier, ' Init:', self.List_init_functions)
        print(identifier, ' Act:', self.List_act_functions)
        print(identifier, ' Loop:', self.number_loop_train)

    def codify_components(self):

        layers = [str(x) for x in self.List_dims]

        init_funcs = [str(np.where(init_functions == x)[0][0]) for x in self.List_init_functions]

        act_funcs = [str(np.where(act_functions == x)[0][0]) if len(np.where(act_functions == x)[0]) else "0" for x in self.List_act_functions]

        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(layers) + "_" + ",".join(init_funcs) + "_" + ",".join(act_funcs)


class GeneratorDescriptor:
    def __init__(self, x_dim, z_dim, latent_distribution_function=np.random.uniform, fmeasure="Standard_Divergence"):
        self.X_dim = x_dim
        self.z_dim = z_dim
        self.latent_distribution_function = latent_distribution_function
        self.fmeasure = fmeasure
        self.network = None

    def copy_from_other(self, other):
        self.X_dim = other.X_dim
        self.z_dim = other.z_dim
        self.latent_distribution_function = other.latent_distribution_function

        self.fmeasure = other.fmeasure

        self.network = copy.deepcopy(other.Gen_network)

    def initialization(self, generator_n_hidden, generator_dim_list, generator_init_functions, generator_act_functions, generator_number_loop_train=1):

        self.network = NetworkDescriptor(generator_n_hidden, self.z_dim, self.X_dim, generator_dim_list, generator_init_functions,
                                         generator_act_functions, generator_number_loop_train)

    def print_components(self):
        self.network.print_components("Gen")

        print('Latent:',  self.latent_distribution_function)
        print('Divergence_Measure:', self.fmeasure)

    def codify_components(self):

        return self.network.codify_components()

    def input(self):
        return self.z_dim

    def output(self):
        return self.X_dim


class GenericDescriptor:
    def __init__(self, inp, outp, n_inputs):
        self.in_dim = inp
        self.out_dim = outp
        self.network = None
        self.n_inputs = n_inputs

    def copy_from_other(self, other):
        self.in_dim = other.in_dim
        self.out_dim = other.out_dim

        self.network = copy.deepcopy(other.Gen_network)

    def initialization(self, generic_n_hidden, generic_dim_list, generic_init_functions, generic_act_functions):

        self.network = NetworkDescriptor(generic_n_hidden, self.in_dim, self.out_dim, generic_dim_list, generic_init_functions,
                                         generic_act_functions, None)

    def print_components(self):
        self.network.print_components("Gen")

    def codify_components(self):
        return type(self).__name__[:-10] + "_" + self.network.codify_components()

    def input(self):
        return self.in_dim

    def output(self):
        return self.out_dim


class DiscreteDescriptor:
    def __init__(self, inp, outp, n_inputs):
        self.in_dim = inp
        self.out_dim = outp
        self.network = None
        self.n_inputs = n_inputs

    def copy_from_other(self, other):
        self.in_dim = other.in_dim
        self.out_dim = other.out_dim

        self.network = copy.deepcopy(other.Gen_network)

    def initialization(self, generic_n_hidden, generic_dim_list, generic_init_functions, generic_act_functions):

        self.network = NetworkDescriptor(generic_n_hidden, self.in_dim, self.out_dim, generic_dim_list, generic_init_functions,
                                         generic_act_functions, None)

    def print_components(self):
        self.network.print_components("Gen")

    def codify_components(self):

        return type(self).__name__[:-10] + "_" + self.network.codify_components()

    def input(self):
        return self.in_dim

    def output(self):
        return self.out_dim


class EncoderDescriptor:
    def __init__(self, x_dim, z_dim, prior):
        self.X_dim = x_dim
        self.z_dim = z_dim
        self.network = None
        self.prior = prior
        self.prior_params = [1., 1.]

    def copy_from_other(self, other):
        self.X_dim = other.X_dim
        self.z_dim = other.z_dim
        self.prior = other.prior

        self.network = other.Gen_network     # These are  Network_Descriptor structures

    def initialization(self, encoder_n_hidden, encoder_dim_list, encoder_init_functions, encoder_act_functions, encoder_number_loop_train=1):

        input_dim = self.X_dim
        output_dim = self.z_dim
        self.network = NetworkDescriptor(encoder_n_hidden, input_dim, output_dim, encoder_dim_list,
                                         encoder_init_functions, encoder_act_functions, encoder_number_loop_train)

    def print_components(self):
        self.network.print_components("Enc")

    def codify_components(self):

        return self.network.codify_components()

    def input(self):
        return self.X_dim

    def output(self):
        return self.z_dim


class DecoderDescriptor:
    def __init__(self, z_dim, x_dim, n_inputs):
        self.in_dim = x_dim
        self.out_dim = z_dim
        self.n_inputs = n_inputs
        self.network = None
        self.conds = []
        self.rands = []

    def copy_from_other(self, other):
        self.in_dim = other.X_dim
        self.out_dim = other.z_dim

        self.network = other.Dec_network

    def initialization(self, decoder_n_hidden, decoder_dim_list, decoder_init_functions, decoder_act_functions, decoder_number_loop_train=1):

        self.network = NetworkDescriptor(decoder_n_hidden, self.out_dim, self.in_dim, decoder_dim_list, decoder_init_functions,
                                         decoder_act_functions, decoder_number_loop_train)

    def print_components(self):
        self.network.print_components("Dec")

    def codify_components(self):

        return type(self).__name__[:-10] + "_" + self.network.codify_components()

    def input(self):
        return self.out_dim

    def output(self):
        return self.in_dim
