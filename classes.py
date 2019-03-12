class Connection(object):
    def __init__(self, input_component, output_component, info):
        """
        :param input_component: Input of the connection (the component providing the data)
        :param output_component: Output of the connection (the component receiving the data)
        Both parameters need to be indices
        """
        self.input = input_component
        self.info = info
        self.output = output_component

    def print(self):
        return "From " + self.input + " to " + self.output + ", " + self.info.print()

    def codify(self):
        return self.input + "_" + self.output + "_" + self.info.type + "_" + str(self.info.size)


class InOut(object):
    def __init__(self, data_type, size):
        """
        :param data_type: Type of the data
        :param size: Size of the data
        :return:
        """
        self.type = data_type
        self.size = size

    def print(self):
        return "Type: " + self.type + ", Size: " + str(self.size)


class Component(object):
    def __init__(self, taking, producing, depth):

        if (type(taking) is not InOut and taking is not None) or (type(producing) is not InOut and producing is not None):
            raise Exception("Both the input and output of the model components must be InOut Objects")

        self.taking = taking
        self.producing = producing
        self.depth = depth

    def print(self):
        if self.taking is not None:
            string = "Taking:\n\t" + self.taking.print()
        else:
            string = ""
        if self.producing is not None:
            string += "\n\t Producing: \n\t" + self.producing.print()
        return string


class NetworkComp(Component):
    def __init__(self, descriptor, taking, producing, depth):
        """
        :param descriptor: Parameters of the network, NetworkDescriptor object
        """
        self.descriptor = descriptor  # Network descriptor
        super().__init__(taking, producing, depth)

    def change_input(self, inp):
        self.taking = inp
        self.descriptor.in_dim = inp
        self.descriptor.network.input_dim = max(inp, self.descriptor.network.input_dim)

    def update_output(self, inp):
        self.producing.size = max(inp, self.producing.size)
        self.descriptor.out_dim = max(inp, self.descriptor.out_dim)
        self.descriptor.network.output_dim = max(inp, self.descriptor.network.output_dim)

    def increase_input(self, size):
        self.taking.size += size
        self.descriptor.in_dim += size
        self.descriptor.network.input_dim += size

    def increase_output(self, size):
        self.producing += size
        self.descriptor.out_dim += size
        self.descriptor.network.output_dim += size


class DistributionComp(Component):
    def __init__(self, distribution, parameters, taking, producing, depth):
        """
        :param parameters: Parameters of the distribution
        """
        self.type = distribution  # Distribution to which the values will be approximated
        self.parameters = parameters  # Parameters to which the distribution will be approximated
        super().__init__(taking, producing, depth)


class ModelComponent(Component):
    def __init__(self, taking, producing, depth):
        super().__init__(taking, producing, depth)
