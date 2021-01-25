import numpy as np
from Function import Function


class Activation(Function):
    def __init__(self, Z, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Z = Z
        self.Activation_Output = self.forward_pass(Z)
        self.grad = self.local_gradient(Z)

    def backward_pass(self, dY):
        # dY is the global gradient from the nec=xt layer propagated back to this layer
        return dY * self.grad['dA']

    def Sigmoid(Z):
        Sigmoid_Act = Sigmoid(Z)
        return Sigmoid_Act

    def Relu(Z):
        Relu_Act = Relu(Z)
        return Relu_Act

    def BinaryStep(Z):
        BinaryStep_Act = BinaryStep(Z)
        return BinaryStep_Act

    def Linear_act(Z):
        Linear_Act = Linear_act(Z)
        return Linear_Act

    def Softmax(Z):
        Softmax_Act = Softmax(Z)
        return Softmax_Act

    def LeakyReLU(Z):
        LeakyReLU_Act = LeakyReLU(Z)
        return LeakyReLU_Act

    def Tanh(Z):
        Tanh_Act = Tanh(Z)
        return Tanh_Act



class Sigmoid(Activation):
    def forward_pass(self, Z):
        o = 1 / (1 + np.exp(-Z))  # o is an intermediate variable
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o

    def local_gradient(self, Z):
        sigmoid = 1 / (1 + np.exp(-Z))

        self.grad = {'dA': np.multiply(sigmoid, (1 - sigmoid))}
        return self.grad


class Relu(Activation):
    def forward_pass(self, Z):
        o = np.maximum(0, Z)  # o is an intermediate variable
        self.cache['Z'] = Z
        self.cache['A'] = o

        return o

    def local_gradient(self, Z):
        Z = (Z > 0)
        self.grad = {'dA': Z}
        return self.grad


class BinaryStep(Activation):
    def forward_pass(self, Z):
        if Z >= 0:  # o is an intermediate variable
            o = 1
        elif Z < 0:
            o = -1
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o

    def local_gradient(self, Z):
        self.grad = {'dA': 0}
        return self.grad


class Linear_act(Activation):
    def forward_pass(self, Z):
        self.cache['Z'] = Z
        self.cache['A'] = Z
        return Z

    def local_gradient(self, Z):
        self.grad = {'dA': np.ones_like(Z)}
        return self.grad


class Softmax(Activation):
    def forward_pass(self, Z):
        exp_x = np.exp(Z)
        o = exp_x / np.sum(exp_x, axis=0, keepdims=True)  # o is an intermediate variable
        #print("\n dy el o b3d ma et7sbt: ",o)
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o

    def local_gradient(self, Z):
        '''
        Sz = self.cache['A']
        N = Z.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = Sz[i, 0] * (np.float64(i == j) - Sz[j, 0])


        D = -np.outer(N,N) + np.diag(Sz.flatten())
        '''
        g = np.ones_like(Z)
        self.grad= {'dA': g}
        return self.grad



'''
    def local_gradient(self, Z):
        jacobian_m = np.diag(Z)
        if (jacobian_m.shape[0]==1):
            jacobian_m = np.identity(jacobian_m.shape[0])
        for i in range(len(jacobian_m.shape[0])):
            for j in range(len(jacobian_m.shape[1])):
                if i == j:
                    jacobian_m[i][j] = Z[i] * (1 - Z[i])
                else:
                    jacobian_m[i][j] = -Z[i] * Z[j]
        self.grad = {'dA': jacobian_m}
        return self.grad
'''
class LeakyReLU(Activation):
    def forward_pass(self, Z, LRP=0.5):
        o = Z * (Z > 0) + LRP * Z * (Z <= 0)  # o is an intermediate variable
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o
        # Leaky Relu Parameter is LRP determines slope of negative part of the leaky relu function

    def local_gradient(self, Z, alpha=0.5):
        self.grad = {'dA': (1 * (Z > 0) + alpha * (Z <= 0))}
        return self.grad

class Tanh(Activation):
    def forward_pass(self, Z):
        o = np.tanh(Z)
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o

    def local_gradient(self, Z):
        self.grad = {'dA': (1-np.square(np.tanh(Z)))}
        return self.grad




def get_activation(activation):
    if (activation == 'sigmoid'):
        return Sigmoid
    elif (activation == 'relu'):
        return Relu
    elif (activation == 'identity'):
        return Linear_act
    elif (activation == 'softmax'):
        return Softmax
    elif (activation == 'leakyrelu'):
        return LeakyReLU
    elif (activation == 'binarystep'):
        return BinaryStep
    elif (activation == 'tanh'):
        return Tanh
    else:
        raise Exception ("Error in Activation Function type : Please choose a valid type")

