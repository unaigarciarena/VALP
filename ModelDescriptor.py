import numpy as np
from Networks import generic_descriptor, discrete_descriptor, decoder_descriptor
from classes import InOut, NetworkComp
import copy
from random import shuffle

"""
distributions = ["Normal", "StickBreaking"]
dist_params = {"Normal": [[0, 1]], "StickBreaking": [[1, 1], [1, 3], [1, 5], [1, 7]]}
networks = ["Encoder", "Decoder", "Generator", "Generic"]
data_types = ["samples", "values"]

i1 = ModelComponent(None, InOut(size=5, type="samples"))
i2 = ModelComponent(None, InOut(size=2, type="values"))

o1 = ModelComponent(InOut(size=5, type="values"), None)
o2 = ModelComponent(InOut(size=10, type="samples"), None)
o3 = ModelComponent(InOut(size=2, type="values"), None)

model_inputs = [i1, i2]
model_outputs = [o1, o2, o3]

active_outputs = copy.copy(model_outputs)
active_inputs = copy.copy(model_inputs)

output_list = copy.copy(model_outputs)
input_list = copy.copy(model_inputs)

comp_limit = 3

model = MNMDescriptor(comp_limit, model_inputs, model_outputs)

"""


def complete_model(model):

    while len(model.active_outputs) > 0:
        output = model.active_outputs[0]

        inp = model.random_input(output)  # Look if there is a network that can serve the output

        if inp != -1:
            model.connect(inp, output)

            model.delete_active_by_index(output)

            model.reachable[inp] = model.reachable[inp] + model.reachable[output]
            for comp in list(model.reachable.keys()):
                if inp in model.reachable[comp]:
                    model.reachable[comp] += model.reachable[output]

        else:  # In case there is not, create a new one
            aux = np.random.randint(2, 50)
            inp = None
            if model.comp_by_ind(output).taking.type in "discrete":
                d = discrete_descriptor(0, aux)
                inp = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="discrete", size=aux), model.get_depth(output)+1)

            elif model.comp_by_ind(output).taking.type in "values":
                d = generic_descriptor(0, aux)
                inp = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="values", size=aux), model.get_depth(output)+1)

            elif model.comp_by_ind(output).taking.type in "samples":
                d = decoder_descriptor(0, aux, 0)
                inp = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="samples", size=aux), model.get_depth(output)+1)

            inp = model.add_net(inp)
            model.connect(inp, output)

            if "o" in output:
                model.reachable[inp] = [output, inp]
            else:
                model.reachable[inp] = model.reachable[output] + [output, inp]

            model.delete_active_by_index(output)
            model.active_outputs += [inp]

    return model


def recursive_creator(model, depth):

    model.constructed = True

    model = recursive_function(model, depth)

    fix_in_out_sizes(model)
    return model


def recursive_function(model, depth):
    if (model.max_comp - model.comp_number()) <= len([x for x in model.active_outputs if x in model.outputs or ("Network" in type(model.comp_by_ind(x)).__name__ and "Decoder" in type(model.comp_by_ind(x).descriptor).__name__)]):
        return complete_model(model)

    elements = model.networks.copy()

    elements.update(model.outputs)

    a = list(elements.keys())

    shuffle(a)

    #for con_output in a:  # For each output and network in the model
    #while len(model.active_outputs) > 0:
    con_output = np.random.choice(a)
    m = copy.deepcopy(model)  # Create a copy of the model

    index = m.random_input(con_output)

    if np.random.rand() < 0.5 or index == -1:  # Create new component as input of the connection

        aux = np.random.randint(2, 50)
        con_input = None
        if m.comp_by_ind(con_output).taking.type in "discrete" or (np.random.rand() < 0.5 and "o" not in con_output):
            d = discrete_descriptor(0, aux)
            con_input = NetworkComp(d, InOut(data_type="", size=0), InOut(data_type="discrete", size=aux), m.get_depth(con_output)+1)

        if m.comp_by_ind(con_output).taking.type in "values":
            if con_output not in m.active_outputs and np.random.rand() < 0.1 and "o" not in con_output:
                d = discrete_descriptor(0, aux)
                con_input = NetworkComp(d, InOut(data_type="", size=0), InOut(data_type="discrete", size=aux), m.get_depth(con_output)+1)
            else:
                d = generic_descriptor(0, aux)
                con_input = NetworkComp(d, InOut(data_type="", size=0), InOut(data_type="values", size=aux), m.get_depth(con_output)+1)

        elif "samples" in m.comp_by_ind(con_output).taking.type:

            if np.random.rand() > 0.5:
                d = decoder_descriptor(0, aux, 0)
                con_input = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="samples", size=aux), m.get_depth(con_output)+1)
            else:
                d = generic_descriptor(0, aux)
                con_input = NetworkComp(d, InOut(data_type="samples", size=0), InOut(data_type="samples", size=aux), m.get_depth(con_output)+1)

        index = m.add_net(con_input)

        if "o" in con_output:
            m.reachable[index] = [con_output, index]
        else:
            m.reachable[index] = m.reachable[con_output] + [con_output, index]

        m.networks[index] = con_input

        m.active_outputs += [index]

    else:  # Use an existing component as the input for the new connection
        m.reachable[index] = m.reachable[index] + m.reachable[con_output]
        for comp in list(m.reachable.keys()):
            if index in m.reachable[comp]:
                m.reachable[comp] += m.reachable[con_output]

    m.connect(index, con_output)

    m.delete_active_by_index(con_output)

    a = recursive_function(m, depth+1)

    return a


def fix_in_out_sizes(model):

    for out in model.outputs:  # Change the output of the networks in the last layer to the maximum required size
        for comp in model.comp_by_input(out):
            model.comp_by_ind(comp).update_output(model.comp_by_ind(out).taking.size)

    for inp in model.networks:  # Increase the input size of the networks to fit all the incomes they have
        for comp in model.comp_by_input(inp):
            model.comp_by_ind(inp).increase_input(model.comp_by_ind(comp).producing.size)

    for con in model.connections:
        if "o" not in con.output:
            con.info.size = model.comp_by_ind(con.input).producing.size
            model.comp_by_ind(con.output).descriptor.n_inputs += 1
        else:
            con.info.size = model.comp_by_ind(con.output).taking.size

    for n in model.networks:
        if "Decoder" in type(model.networks[n].descriptor).__name__:
            for i in range(model.networks[n].descriptor.n_inputs):
                if np.random.rand() < 0.3 and len(model.networks[n].descriptor.rands) > 0:
                    model.networks[n].descriptor.conds += [i]
                else:
                    model.networks[n].descriptor.rands += [i]


def cell_descriptor(model, inherit_prob, dupl_prob):

    # ########################## Additive construction ######################### #
    inputs = {"n0": np.array(list(model.inputs.keys()))}
    for i in model.outputs.keys():
        inputs[i] = ["n0"]
    outputs = {"n0": np.array(list(model.outputs.keys()))}
    for i in model.inputs.keys():
        outputs[i] = ["n0"]
    in_aux = 0
    out_aux = np.random.randint(1, 10)
    d = NetworkComp(generic_descriptor(in_aux, out_aux), InOut("Values", in_aux), InOut("Values", out_aux), 0)
    model.add_net(d, "n0")

    for i in range(1, model.max_comp):

        new_net = "n" + str(i)

        ins = np.random.choice(inputs["n0"], size=np.random.randint(1, max(2, int(len(inputs["n0"])*inherit_prob))), replace=False)  # Select what inputs are passed to the new network
        for j in ins:  # This loop deletes some of the inputs from the network (some are duplicated others are just passed)
            outputs[j] = np.append(outputs[j], [new_net])
            if np.random.random() < 1-dupl_prob and len(inputs["n0"]) > 1:
                inputs["n0"] = inputs["n0"][inputs["n0"] != j]
                outputs[j] = outputs[j][outputs[j] != "n0"]

        outs = np.random.choice(outputs["n0"], size=np.random.randint(1, max(2, int(len(outputs["n0"])*inherit_prob))), replace=False)  # Select what outputs are passed to the new network
        for j in outs:  # This loop deletes some of the outputs from the network (some are duplicated others are just passed)
            inputs[j] = np.append(inputs[j], [new_net])
            if np.random.random() < 1-dupl_prob and len(outputs["n0"]) > 1:
                outputs["n0"] = outputs["n0"][outputs["n0"] != j]
                inputs[j] = inputs[j][inputs[j] != "n0"]

        action = np.random.random()  # Define whether the new net is placed before, after or in parallel to n0

        if action < 1/3:  # After
            ins = np.append(ins, ["n0"])
            outputs["n0"] = np.append(outputs["n0"], new_net)

        elif action < 2/3:  # Before
            outs = np.append(outs, ["n0"])
            inputs["n0"] = np.append(inputs["n0"], new_net)

        inputs[new_net] = ins
        outputs[new_net] = outs
        aux_out = np.random.randint(1, 10)
        d = NetworkComp(generic_descriptor(0, aux_out), InOut("Values", 0), InOut("Values", aux_out), 0)

        model.add_net(d, new_net)

    # ############################# Network type decision ############################ #

    # ############################# Descriptive transformation ####################### #

    fix_in_out_sizes(model)

    for comp in inputs:
        for inp in inputs[comp]:
            model.connect(inp, comp)


"""
def main():
    for seed in range(0, 50):
        np.random.seed(seed)
        random.seed(seed)

        inp_dict = {}
        outp_dict = {}

        for i in range(10):
            inp_dict["i" + str(i)] = ModelComponent(None, InOut(size=np.random.randint(1, 10), type="values"))

        for o in range(10):
            outp_dict["o" + str(o)] = ModelComponent(InOut(size=np.random.randint(1, 10), type="values"), None)

        model = MNMDescriptor(5, inp_dict, outp_dict)

        cell_descriptor(model, .5, .2)
        model.print()
        model.print_model_graph("hueuhu" + str(seed))

"""
