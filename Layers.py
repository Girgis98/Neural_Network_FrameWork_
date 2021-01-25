from Activations import *
from Function import Function


class Layer(Function):

    def __init__(self, Input_Matrix, Weight_Matrix, Bias_Matrix, Activation_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_layer = False
        self.inputs =np.array(Input_Matrix)
        #self.inputs = np.array(Input_Matrix).reshape(len(Input_Matrix), -1)
        self.weights = Weight_Matrix
        self.bias = Bias_Matrix
        self.weights_gradients = {}
        self.bias_gradients = {}
        self.act_fn_name = Activation_fn
        self.activation_Fn = get_activation(Activation_fn)
        self()

    def update_weights(self, Weights, Bias):  # bn callha f el optization 3la kol layer
        self.weights = Weights
        self.bias = Bias
        self()  # deh bt call el layer nafsha fa bt7sb el formward w el grad w kda
        # m7tag hena a recalcuate el output w el grad w el backprop bta3 kol el layers # mosh kol layer el layer el na feha bas w el optimization t loop 3la kol el layers

    def output(self):
        return self.cache['A'].cache['A']
        #return self.cache['A'].cache['A']

    def initialize_layers(self, dim_list, input_x):
        w = {}
        b = {}
        layers = {}
        for i in range(0, len(dim_list)):
            w[i] = np.random.rand(dim_list[i])
            b[i] = np.random.rand((1, 1))

        layers[0] = Layer(input_x, w[0], b[0])
        for i in range(1, len(dim_list)):
            layers[i] = Layer(layers[i - 1].output(), w[i], b[i])

        return layers


class Linear(Layer):

    def forward_pass(self):
        Z = np.dot(self.weights, self.inputs) + self.bias
        A = self.activation_Fn(Z)

        self.cache['X'] = self.inputs  # Z1 = W1 . X + b1
        self.cache['Z'] = Z  # A1 = f1(Z1)
        self.cache['A'] = A
        return A

    def local_gradient(self):
        gradient_X_L = self.weights
        gradient_W_L = self.inputs
        gradient_B_L = np.ones(self.bias.shape, dtype='float16')
        self.grad = {'dZ': gradient_X_L, 'dW': gradient_W_L, 'db': gradient_B_L}
        return self.grad

    def backward_pass(self, dG):
        Z = self.cache['Z']
        dA = np.multiply(dG, self.activation_Fn.grad['dA'])
        dX = np.dot(self.grad['dZ'].T, dA)  # m7tag a3dl henaa
        #if self.last_layer:
        dW = np.dot(dA, self.grad['dW'].T)

            # self.last_layer = False
        #elif not self.last_layer:
            #dW = np.dot(dX, self.grad['dW'].T)
        db = np.sum(dA, axis=1, keepdims=True)  # changed here none to 0

        self.weights_gradients = {'W': dW}
        self.bias_gradients = {'b': db}
        return dX

