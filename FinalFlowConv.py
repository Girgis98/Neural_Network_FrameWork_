
cuda = False


try:
    import cupy as np
    cuda = True
except:
    import numpy as np


from math import sqrt
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd

class Function:

    def __init__(self, *args, **kwargs):  # initialize cache for inputs, outputs and the local grads
        self.cache = {}
        self.grad = {}

    def __call__(self, *args, **kwargs):
        self.out = self.forward_pass(*args, **kwargs)
        self.grad = self.local_gradient(*args, **kwargs)

        return self.out

    def forward_pass(self, *args, **kwargs):  # implemented specifically in other classes
        pass

    def local_gradient(self, *args, **kwargs):
        pass

    def backward_pass(self, *args, **kwargs):
        pass


class Activation(Function):
    def __init__(self, Z, *args, **kwargs):  # Z input ely da5li
        super().__init__(*args, **kwargs)
        self.Z = Z
        self.Activation_Output = self.forward_pass(Z)
        self.grad = self.local_gradient(Z)

    def backward_pass(self, dY):
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
        o = 1 / (1 + np.exp(-Z))  # o is an intermediate variable for output
        self.cache['Z'] = Z  # cache input
        self.cache['A'] = o  # cache output of act fn
        return o

    def local_gradient(self, Z):
        sigmoid = 1 / (1 + np.exp(-Z))
        self.grad = {'dA': np.multiply(sigmoid, (1 - sigmoid))}
        return self.grad


class Relu(Activation):
    def forward_pass(self, Z):
        o = np.maximum(0, Z)
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o

    def local_gradient(self, Z):
        Z = (Z > 0)
        self.grad = {'dA': Z}
        return self.grad


class BinaryStep(Activation):
    def forward_pass(self, Z):  # lw akbr mn 0 -> 1 lw as8r -> -1
        if Z >= 0:
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
        o = exp_x / np.sum(exp_x, axis=0, keepdims=True)  # e /sum e
        # print("\n dy el o b3d ma et7sbt: ",o)
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
        g = np.ones_like(Z)  ######### with respect to l
        self.grad = {'dA': g}
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
    def forward_pass(self, Z, LRP=0.2):
        o = Z * (Z > 0) + LRP * Z * (Z <= 0)  # o is an intermediate variable
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o
        # Leaky Relu Parameter is LRP determines slope of negative part of the leaky relu function

    def local_gradient(self, Z, alpha=0.2):
        self.grad = {'dA': (1 * (Z > 0) + alpha * (Z <= 0))}
        return self.grad


class Tanh(Activation):
    def forward_pass(self, Z):
        o = np.tanh(Z)
        self.cache['Z'] = Z
        self.cache['A'] = o
        return o

    def local_gradient(self, Z):
        self.grad = {'dA': (1 - np.square(np.tanh(Z)))}
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
    else:  # in case ay input 8lt
        raise Exception("Error in Activation Function type : Please choose a valid type")


class Model_Utils():
    def __init__(self, save_path='saved_models/'):
        self.save_path = save_path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

    def save(self, model, name):
        save_file = open(self.save_path + name, "wb")
        pickle.dump(model, save_file)
        save_file.close()

    def load(self, name):
        save_file = open(self.save_path + name, "rb")
        model = pickle.load(save_file)
        save_file.close()
        return model


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.
        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.
        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """

    m, n_C, n_H, n_W = X_shape

    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # da awl level
    level1 = np.repeat(np.arange(HF), WF)
    # copy mn elchannels
    level1 = np.tile(level1, n_C)

    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # copy el channels
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.
        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad),
                          (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform our matrix back to the input image.
        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    # Index matrices, necessary to transform our input image into a matrix.
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    if cuda:
        np.scatter_add(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    else:
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[:, :, pad:-pad, pad:-pad]


class Layer(Function):

    def __init__(self, Input_Matrix, Output_Dimension, Activation_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_layer = False
        self.inputs = np.array(Input_Matrix)
        self.in_dimension = self.inputs.shape[0]
        self.out_dimension = Output_Dimension
        self._init_weights(self.in_dimension, self.out_dimension)
        #self.inputs = np.array(Input_Matrix).reshape(len(Input_Matrix), -1)
        self.weights_gradients = {}
        self.bias_gradients = {}
        self.act_fn_name = Activation_fn
        self.activation_Fn = get_activation(Activation_fn)
        self()

    def update_weights(self, Weights, Bias):  # bn callha f el optimization 3la kol layer
        self.weights = Weights
        self.bias = Bias
        self()  # deh bt call el layer nafsha fa bt7sb el formward w el grad w kda
        # m7tag hena a recalcuate el output w el grad w el backprop bta3 kol el layers # mosh kol layer el layer el na feha bas w el optimization t loop 3la kol el layers

    def output(self):
        return self.cache['A'].cache['A']

    def _init_weights(self, in_dimension, out_dimension):
        scale = 1 / np.sqrt(self.in_dimension)

        self.weights = scale * \
                       np.random.randn(self.out_dimension, self.in_dimension)
        self.bias = scale * np.random.randn(self.out_dimension, 1)

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


class Conv(Function):

    def __init__(self, X, in_channels, out_channels, kernel_size=3, stride=1, padding=0, Activation_fn='relu'):
        super().__init__()
        self.last_layer = False
        self.cache['X'] = X
        self.inputs = X
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self._init_weights(in_channels, out_channels, self.kernel_size)
        self.act_fn_name = Activation_fn
        self.activation_Fn = get_activation(Activation_fn)
        self.weights_gradients = {}
        self.bias_gradients = {}
        self.grad = {}

    def _init_weights(self, in_channels, out_channels, kernel_size):
        scale = 2/sqrt(in_channels*kernel_size*kernel_size)

        self.weights = np.random.normal(scale=scale, size=(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(shape=(out_channels, 1))

    def update_weights(self, Weights, Bias):  # bn callha f el optization 3la kol layer

        self.weights = Weights
        self.bias = Bias
        self()

    def output(self):

        self.forward_pass()
        out = self.cache['A'].Activation_Output
        return out

    def forward_pass(self):

        X = self.cache['X']

        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.out_channels

        n_H = int((n_H_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        n_W = int((n_W_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        X_col = im2col(X, self.kernel_size, self.kernel_size,
                       self.stride, self.padding)

        w_col = self.weights.reshape(self.out_channels, -1)

        b_col = self.bias.reshape(-1, 1)

        out = w_col @ X_col + b_col

        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))

        self.cache['X'] = X

        self.cache['other'] = X_col, w_col

        self.cache['Z'] = out

        A = self.activation_Fn(out)

        self.cache['A'] = A

        return A

    def backward_pass(self, dY):

        X = self.cache['X']
        X_col, w_col = self.cache['other']

        m, _, _, _ = X.shape

        #self.grad['db'] = np.sum(dY, axis=(0,2,3))
        db = np.sum(dY, axis=(0, 2, 3))

        db = db.reshape(self.bias.shape)

        self.bias_gradients['b'] = db

        dY = dY.reshape(dY.shape[0] * dY.shape[1], dY.shape[2] * dY.shape[3])

        dY = np.array(np.vsplit(dY, m))

        dY = np.concatenate(dY, axis=-1)

        dX_col = w_col.T @ dY

        dw_col = dY @ X_col.T

        dX = col2im(dX_col, X.shape, self.kernel_size,
                    self.kernel_size, self.stride, self.padding)

        #self.grad['dW'] = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))

        dw = dw_col.reshape(
            (dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))

        self.weights_gradients['W'] = dw

        #self.weights_gradients['W'] = self.grad['dW']

        #self.bias_gradients['b'] = self.grad['db']

        return dX


class MaxPool(Function):

    def __init__(self, X, kernel_size=3, stride=1, padding=0):

        super().__init__()

        self.cache['X'] = X
        self.inputs = X
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.last_layer = False
        self.weights = np.zeros((1, 1))
        self.bias = np.zeros((1, 1))
        self.weights_gradients = {'W': np.zeros((1, 1))}
        self.bias_gradients = {'b': np.zeros((1, 1))}

    def output(self):

        return self.forward_pass()

    def update_weights(self, Weights, Bias):  # bn callha f el optization 3la kol layer

        self.weights = Weights
        self.bias = Bias
        self()

    def forward_pass(self):

        X = self.cache['X']

        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev

        n_H = int((n_H_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        n_W = int((n_W_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        X_col = im2col(X, self.kernel_size, self.kernel_size,
                       self.stride, self.padding)

        X_col = X_col.reshape(n_C, X_col.shape[0]//n_C, -1)

        M_pool = np.max(X_col, axis=1)

        M_pool = np.array(np.hsplit(M_pool, m))

        M_pool = M_pool.reshape(m, n_C, n_H, n_W)

        self.cache['A'] = M_pool

        return M_pool

    def backward_pass(self, dY):

        X = self.cache['X']

        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev

        n_H = int((n_H_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        n_W = int((n_W_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        dY_flatten = dY.reshape(n_C, -1) / \
            (self.kernel_size * self.kernel_size)

        dX_col = np.repeat(dY_flatten, self.kernel_size *
                           self.kernel_size, axis=0)

        dX = col2im(dX_col, X.shape, self.kernel_size,
                    self.kernel_size, self.stride, self.padding)

        dX = dX.reshape(m, -1)

        dX = np.array(np.hsplit(dX, n_C_prev))

        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)

        return dX


class AvgPool(Function):

    def __init__(self, X, kernel_size=3, stride=1, padding=0):

        super().__init__()

        self.cache['X'] = X
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.has_weights = False

    def forward_pass(self):

        X = self.cache['X']

        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev

        n_H = int((n_H_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        n_W = int((n_W_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        X_col = im2col(X, self.kernel_size, self.kernel_size,
                       self.stride, self.padding)

        X_col = X_col.reshape(n_C, X_col.shape[0]//n_C, -1)

        A_pool = np.mean(X_col, axis=1)

        A_pool = np.array(np.hsplit(A_pool, m))

        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

    def backward_pass(self, dY):

        X = self.cache['X']

        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev

        n_H = int((n_H_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        n_W = int((n_W_prev + 2 * self.padding -
                   self.kernel_size) / self.stride) + 1

        dY_flatten = dY.reshape(n_C, -1) / \
            (self.kernel_size * self.kernel_size)

        dX_col = np.repeat(dY, self.kernel_size *
                           self.kernel_size, axis=0)

        dX = col2im(dX_col, X.shape, self.kernel_size,
                    self.kernel_size, self.stride, self.padding)

        dX = dX.reshape(m, -1)

        dX = np.array(np.hsplit(dX, n_C_prev))

        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)

        return dX


class Flatten(Function):

    def __init__(self, X):

        super().__init__()
        self.inputs = X
        self.last_layer = False
        self.cache['X'] = X
        self.weights = np.zeros((1, 1))
        self.bias = np.zeros((1, 1))
        self.weights_gradients = {'W': np.zeros((1, 1))}
        self.bias_gradients = {'b': np.zeros((1, 1))}

    def output(self):

        return self.forward_pass()

    def forward_pass(self):

        X = self.cache['X']
        self.cache['shape'] = X.shape

        n_batch = X.shape[0]
        return X.reshape(n_batch, -1)

    def backward_pass(self, dY):
        dX = dY.reshape(self.cache['shape'])
        return dX

    def update_weights(self, Weights, Bias):  # bn callha f el optization 3la kol layer

        self.weights = Weights
        self.bias = Bias
        self()


class Loss(Function):
    def __init__(self, Y, Y_hat, cache, weights, bias, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Y_hat = np.array(Y_hat)
        self.Y = np.array(Y)
        #self.Y = np.reshape(self.Y, np.shape(self.Y_hat))

        self.flag = 0

        self.layer_cache = cache
        self.layer_weights = weights
        self.layer_bias = bias

        self.eval_metrics={}

        self.Loss_Value = self.forward_pass()
        self.local_gradient()



    def F1_score(self, y, y_cap, binary_or_multi):  # binary = True ,  Multi = False

        if binary_or_multi:
           # y_cap = (np.round(y))
           # y_cap=y_cap.astype(int)
            y_index = np.greater_equal(y_cap,0.5)
            y_cap[y_index] = 1
            y_cap[np.invert(y_index)] = 0
            y_cap = y_cap.astype(int)

            index2 = np.equal(-1, y)
            y[index2] = 0

            y_cap_0= np.equal(y_cap,0)
            y_cap_1= np.equal(y_cap,1)
            y_0=np.equal(y,0)
            y_1=np.equal(y,1)

            TP= np.sum(np.equal(y_1,y_cap_1))
            TN= np.sum(np.equal(y_0,y_cap_0))
            FP=np.sum(np.equal(y_0,y_cap_1))
            FN=np.sum(np.equal(y_1,y_cap_0))

            acc = (TP+TN) / (TP+TN+FP+FN)
            recall = TP / (TP+FN)
            precision = TP / (TP+FP)
            specificity =TN / (TN+FP)

            F1= (2* precision * recall) / (precision + recall)
            self.eval_metrics={'accuracy':acc,'recall':recall,'precision':precision,'specificity':specificity,'F1 score':F1}
            return self.eval_metrics

        elif not binary_or_multi:  # T and F only

            y_cap_max = np.argmax(y_cap, axis=0)  # 1 x N
            y_cap_max =np.reshape(y_cap_max,((y.shape)))
            result = np.equal(y, y_cap_max)

            acc = (np.sum(result)) / (y_cap.shape[1])

            self.eval_metrics={'accuracy':acc}
            return self.eval_metrics


    def backward_pass(self):
            # Since it is the final layer the backward pass is the same as the local gradient
            # self.Y_hat = self.layer_cache['A'].forward()
            if self.flag == 2:
                self()
            return self.grad['dL']

    def MSE(self):
        Mse_Loss = MSE((self.Y), (self.Y_hat), self.layer_cache, self.layer_weights, self.layer_bias)
        self.Loss_type = Mse_Loss
        return Mse_Loss

    def SignLoss(self):
        Sign_Loss = SignLoss((self.Y), (self.Y_hat), self.layer_cache, self.layer_weights, self.layer_bias)
        self.Loss_type = Sign_Loss
        return Sign_Loss

    def SVMLoss(self):
        SVM_Loss = SVMLoss((self.Y), (self.Y_hat), self.layer_cache, self.layer_weights, self.layer_bias)
        self.Loss_type = SVM_Loss
        return SVM_Loss

    def LogisticRegressionSigmoid(self):
        LogisticRegressionSigmoid_loss = LogisticRegressionSigmoid((self.Y), (self.Y_hat), self.layer_cache,
                                                                   self.layer_weights, self.layer_bias)
        self.Loss_type = LogisticRegressionSigmoid_loss
        return LogisticRegressionSigmoid_loss

    def LogisticRegressionIdentity(self):
        LogisticRegressionIdentity_loss = LogisticRegressionIdentity((self.Y), (self.Y_hat), self.layer_cache,
                                                                   self.layer_weights, self.layer_bias)
        self.Loss_type = LogisticRegressionIdentity_loss
        return LogisticRegressionIdentity_loss

    def CrossEntropy(self):
        Cross_Loss = CrossEntropy((self.Y), (self.Y_hat), self.layer_cache, self.layer_weights, self.layer_bias)
        self.Loss_type = Cross_Loss
        return Cross_Loss


class MSE(Loss):
    def forward_pass(self):
        L = np.mean((self.Y - self.Y_hat) ** 2, axis=1, keepdims=True) / 2
        self.cache['Loss'] = L
        return L

    def local_gradient(self):
        self.grad = {'dL': ((self.Y_hat - self.Y) / self.Y_hat.shape[1])}
        self.flag = 1
        return self.grad


class SignLoss(Loss):
    def forward_pass(self):
        L = np.mean(np.max(0, -(np.dot(self.Y, self.Y_hat))), axis=1, keepdims=True)
        self.cache['Loss'] = L
        y = np.copy(self.Y)
        yhat = np.copy(self.Y_hat)
        eval_metric = self.F1_score(y, yhat, True)
        return L

    def local_gradient(self):
        #self.grad = {'dL': (- (np.dot(self.Y, self.layer_cache['X'])))}
        self.grad = {'dL': (- self.Y)/ self.Y_hat.shape[1]}
        return self.grad


class SVMLoss(Loss):
    def forward_pass(self):
        L = np.mean(np.max(0, -(np.dot(self.Y, self.Y_hat)) + 1), axis=1, keepdims=True)
        self.cache['Loss'] = L
        y = np.copy(self.Y)
        yhat = np.copy(self.Y_hat)
        eval_metric = self.F1_score(y, yhat, True)

        return L

    def local_gradient(self):
        #self.grad = {'dL': (- (np.dot(self.Y, self.cache['X'])))}  # X is not defined here yet
        self.grad = {'dL': (- self.Y)/ self.Y_hat.shape[1]}
        return self.grad


class LogisticRegressionSigmoid(Loss):
    def forward_pass(self):
        L = np.mean(-np.log(np.absolute((self.Y / 2) - (1 / 2) + self.Y_hat)+0.001), axis=1, keepdims=True)
        y=np.copy(self.Y)
        yhat=np.copy(self.Y_hat)
        eval_metric = self.F1_score(y, yhat, True)
        self.cache['Loss'] = L
        return L

    def local_gradient(self):
        # self.grad = {'dL': (- (np.multiply(self.Y, self.layer_cache['X'])) / (1 + np.exp(np.multiply(self.Y, np.dot(self.layer_weights,self.layer_cache['X'])))))}
        self.grad = {'dL': (-1 /( (self.Y / 2) - (1 / 2) + self.Y_hat +0.001) )/ self.Y_hat.shape[1]}

        return self.grad


class LogisticRegressionIdentity(Loss):
    def forward_pass(self):
        L = np.mean(np.log(np.add(1, np.exp(np.multiply(-self.Y, self.Y_hat)))), axis=1, keepdims=True)
        self.cache['Loss'] = L
        return L

    def local_gradient(self):
        # self.grad = {'dL': (- (np.multiply(self.Y, self.layer_cache['X'])) / (1 + np.exp(np.multiply(self.Y, np.dot(self.layer_weights,self.layer_cache['X'])))))}
        #self.grad = {'dL': (np.multiply(np.exp((np.dot(-self.Y, self.Y_hat), -self.Y))) / (
         #   np.add(1, np.exp(np.dot(-self.Y, self.Y_hat))))) / self.Y_hat.shape[1]}
        self.grad = {'dL': (-(np.divide((self.Y),np.multiply(np.add(1,(np.exp(np.multiply(self.Y,self.Y_hat)))),self.Y_hat.shape[1])))) }
        return self.grad


class CrossEntropy(Loss): ####### used for softmax only

    def forward_pass(self):
        #print("dy el y hat ablloss",self.Y_hat)
        abs_list = [self.Y_hat[(self.Y[i]), i] for i in range(self.Y_hat.shape[1])]

        log_Y_hat = -np.log(np.abs(np.array(abs_list)))

        log_Y_hat[np.isnan(log_Y_hat)] = 0
        crossentropy_loss = np.mean(log_Y_hat)

        y = np.copy(self.Y)
        yhat = np.copy(self.Y_hat)
        eval_metric = self.F1_score(y, yhat, False)


        self.cache['Loss'] = crossentropy_loss
        return crossentropy_loss


        '''
        log_Y_hat = -np.log(np.abs([self.Y_hat[(self.Y[i]),i] for i in range(np.shape(self.Y_hat)[1])]))

        log_Y_hat[np.isnan(log_Y_hat)] = 0
        crossentropy_loss = np.mean(log_Y_hat)
        print("da el log : ", log_Y_hat)
        self.cache['Loss'] = crossentropy_loss

        eval_metric=self.F1_score(self.Y,self.Y_hat,False)

        return crossentropy_loss
        '''


    def local_gradient(self):

        for i in range(self.Y_hat.shape[1]):
            self.Y_hat[self.Y[i],i] =self.Y_hat[self.Y[i],i] -1

        self.grad = {'dL': self.Y_hat/self.Y_hat.shape[1]}
        #self.grad = {'dL': (self.Y_hat - np.ones_like(self.Y_hat)) / float(len(self.Y))} ############           need to check for yi!=r
        return self.grad

        '''
        ones = np.zeros(self.Y_hat.shape, dtype='float16')
        for row_idx, col_idx in enumerate(self.Y):
            ones[row_idx, col_idx] = 1.0


def get_loss(loss):
    if (loss == 'mse'):
        return MSE()
    elif (loss == 'sign'):
        return SignLoss()
    elif (loss == 'svm'):
        return HingeLoss()
    elif (loss == 'crossentropy'):
        return CrossEntropy()
'''


class Optimization(Function):
    def __init__(self, my_Model ,layers, epochs, Loss, learning_rate=0.1, epsilon=0.01, batch_type='Batch', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = []
        if type(learning_rate) == float or type(learning_rate) == int:
            self.learning_rate.append(learning_rate)
            self.learning_rate.append(False)
        else:
            self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.batch_type = batch_type
        self.my_Model = my_Model
        self.loss = Loss
        self.loss_gradient = Loss.grad['dL']
        self.layers = layers
        self.weights = []
        self.bias = []
        self.weights_gradients = []
        self.bias_gradients = []
        self.N = self.layers[0].inputs.shape[1]
        self.epochs = epochs
        self.batch_size = self.N / (np.power(2, int(np.log2(self.N))))
        self.number_of_iterations = (self.N / self.batch_size)
        self.totalgradients = []

        for i in self.layers:
            self.weights.append(i.weights)
            self.bias.append(i.bias)
            self.weights_gradients.append(i.weights_gradients)
            self.bias_gradients.append(i.bias_gradients)
        '''
        if batch_type == 'Batch':
            self.batch_size = self.N
        elif batch_type == 'Online':
            self.batch_size = 1
        elif batch_type == 'minibatch':
            self.batch_size = self.N / (np.power(2, int(np.log2(self.N))))
        else:
            self.batch_size = batch_type
        
        self.number_of_iterations = self.N / self.batch_size
        '''
    def print_eval_metrics(self, eval_dict):
        if 'F1 score' in eval_dict.keys():
            for key, value in eval_dict.items():
                print(key, ' : ', value)
        elif 'accuracy' in eval_dict.keys():
            for key, value in eval_dict.items():
                print(key, ' : ', value)
        else:  # not a classification problem ( has no eval metrics )
            pass

    def calculate_backprop(self, layer):
        if self.flag0 == True:
            self.flag0 = False
            self.layers[-1].last_layer = True
            dL = self.loss.backward_pass()
            self.old_eval_metric = self.loss.eval_metrics
            self.loss()
            self.dD = dL
        # if layer.last_layer:
        # self.loss()
        self.totalgradients.append(self.dD)
        self.dD = layer.backward_pass(self.dD, )
        self.totalgradients.append(self.dD)

    def GradientDescent(self):
        # get weights , bias (done)
        # get weights gradients and bias gradients , delta is same shape as weights
        # this update is done for each layer
        visualizer = Visualizer()
        visualizer.add_plot(name='loss', colour='red')
        visualizer.add_plot(name='accuracy', colour='blue')
        epoch = 1
        while True:
            for g in range((int)(self.number_of_iterations)):
                self.flag0 = True
                for layer in reversed(self.layers):

                    self.calculate_backprop(layer)  # 3shan e7sb el grad bta3 el weights w el bias
                    #delta_weights = np.zeros_like(layer.weights_gradients['W'])
                    #delta_bias = np.zeros_like(layer.bias_gradients['b'])

                    delta_weights = layer.weights_gradients['W']
                    delta_bias = layer.bias_gradients['b']

                    if self.learning_rate[1] is True:
                        decay_rate = self.learning_rate[0] / self.N
                        self.learning_rate[0] = self.learning_rate[0] / (1 + decay_rate * epoch)
                    #print("alpha : ", self.learning_rate[0])
                    weights = layer.weights - self.learning_rate[0] * delta_weights
                    bias = layer.bias - self.learning_rate[0] * delta_bias
                    #print(f'iterations : {g} \n weights : \n{weights} \n bias : \n {bias}')
                    layer.update_weights(weights, bias)
                    if layer.last_layer:
                        self.loss.Y_hat = layer.out
                        print(f'\n Loss : {self.loss.out}')
                        self.print_eval_metrics(self.old_eval_metric )
                    if layer == self.layers[0] and self.batch_type == 'minibatch':
                        self.my_Model.split_Batches()
                visualizer.update([self.loss.out , self.old_eval_metric['accuracy']])
                # self.loss()
            #print(f'epoch : \n {epoch} \n weights : \n{weights} \n bias : \n {bias}')

            if ((epoch == self.epochs) ):#or ((np.linalg.norm(delta_weights) + np.linalg.norm(delta_bias)) < self.epsilon)):
                break
            epoch += 1

    def Momentum(self, beta=0.5):
        iteration = 1
        V_weights = []
        V_bias = []

        for I in range(len(self.layers)):
            V_weights.append(np.zeros_like(self.layers[I].weights))
            V_bias.append(np.zeros_like(self.layers[I].bias))

        while True:
            self.flag0 = True
            for index in reversed(range(len(self.layers))):

                self.calculate_backprop(self.layers[index])  # 3shan e7sb el grad bta3 el weights w el bias
                delta_weights = np.zeros_like(self.layers[index].weights_gradients['W'])

                delta_bias = np.zeros_like(self.layers[index].bias_gradients['b'])

                delta_weights = self.layers[index].weights_gradients['W']
                delta_bias = self.layers[index].bias_gradients['b']

                if self.learning_rate[1] is True:
                    decay_rate = self.learning_rate[0] / self.batch_size
                    self.learning_rate[0] = self.learning_rate[0] / (1 + decay_rate * iteration)

                V_weights[index] = np.multiply(beta, V_weights[index]) - self.learning_rate[0] * delta_weights
                V_bias[index] = np.multiply(beta, V_bias[index]) - self.learning_rate[0] * delta_bias

                weights = self.layers[index].weights + V_weights[index]
                bias = self.layers[index].bias + V_bias[index]

                print(f'iterations : \n {iteration} \n weights : \n{weights} \n bias : \n {bias}')
                self.layers[index].update_weights(weights, bias)

                if self.layers[index].last_layer:
                    self.loss.Y_hat = self.layers[index].out
                    print(f'\n Loss : {self.loss.out}')
                    self.print_eval_metrics(self.loss.eval_metrics)

            if ((iteration == self.epochs) or (
                    (np.linalg.norm(delta_weights) + np.linalg.norm(delta_bias)) < self.epsilon)): break
            iteration += 1

    def AdaGrad(self):
        iteration = 1
        A_weights = []
        A_bias = []

        for I in range(len(self.layers)):
            A_weights.append(np.zeros_like(self.layers[I].weights))
            A_bias.append(np.zeros_like(self.layers[I].bias))

        while True:
            self.flag0 = True
            for index in reversed(range(len(self.layers))):

                self.calculate_backprop(self.layers[index])  # 3shan e7sb el grad bta3 el weights w el bias
                delta_weights = np.zeros_like(self.layers[index].weights_gradients['W'])

                delta_bias = np.zeros_like(self.layers[index].bias_gradients['b'])

                delta_weights = self.layers[index].weights_gradients['W']
                delta_bias = self.layers[index].bias_gradients['b']

                if self.learning_rate[1] is True:
                    decay_rate = self.learning_rate[0] / self.batch_size
                    self.learning_rate[0] = self.learning_rate[0] / (1 + decay_rate * iteration)

                A_weights[index] = A_weights[index] + np.square(delta_weights)
                A_bias[index] = A_bias[index] + np.square(delta_bias)

                shape_w = self.layers[index].weights.shape
                shape_b = self.layers[index].bias.shape

                weights = self.layers[index].weights - np.multiply(
                    (np.divide(self.learning_rate[0], np.sqrt(A_weights))), delta_weights)
                weights = np.reshape(weights, shape_w)
                bias = self.layers[index].bias - np.multiply((np.divide(self.learning_rate[0], np.sqrt(A_bias))),
                                                             delta_bias)
                bias = np.reshape(bias, shape_b)
                print(f'iterations : \n {iteration} \n weights : \n{weights} \n bias : \n {bias}')

                self.layers[index].update_weights(weights, bias)

                if self.layers[index].last_layer:
                    self.loss.Y_hat = self.layers[index].out
                    print(f'\n Loss : {self.loss.out}')
                    self.print_eval_metrics(self.loss.eval_metrics)

            if ((iteration == self.epochs) or (
                    (np.linalg.norm(delta_weights) + np.linalg.norm(delta_bias)) < self.epsilon)): break
            iteration += 1

    def RMSprop(self, ro=0.5):
        iteration = 1
        A_weights = []
        A_bias = []

        for I in range(len(self.layers)):
            A_weights.append(np.zeros_like(self.layers[I].weights))
            A_bias.append(np.zeros_like(self.layers[I].bias))

        while True:
            self.flag0 = True
            for index in reversed(range(len(self.layers))):

                self.calculate_backprop(self.layers[index])  # 3shan e7sb el grad bta3 el weights w el bias
                delta_weights = np.zeros_like(self.layers[index].weights_gradients['W'])

                delta_bias = np.zeros_like(self.layers[index].bias_gradients['b'])

                delta_weights = self.layers[index].weights_gradients['W']
                delta_bias = self.layers[index].bias_gradients['b']

                if self.learning_rate[1] is True:
                    decay_rate = self.learning_rate[0] / self.batch_size
                    self.learning_rate[0] = self.learning_rate[0] / (1 + decay_rate * iteration)

                A_weights[index] = ro * A_weights[index] + (1 - ro) * np.square(delta_weights)
                A_bias[index] = A_bias[index] + np.square(delta_bias)

                shape_w = self.layers[index].weights.shape
                shape_b = self.layers[index].bias.shape

                weights = self.layers[index].weights - np.multiply(
                    (np.divide(self.learning_rate[0], np.sqrt(A_weights))), delta_weights)
                weights = np.reshape(weights, shape_w)
                bias = self.layers[index].bias - np.multiply((np.divide(self.learning_rate[0], np.sqrt(A_bias))),
                                                             delta_bias)
                bias = np.reshape(bias, shape_b)
                print(f'iterations : \n {iteration} \n weights : \n{weights} \n bias : \n {bias}')

                self.layers[index].update_weights(weights, bias)
                if self.layers[index].last_layer:
                    self.loss.Y_hat = self.layers[index].out
                    print(f'\n Loss : {self.loss.out}')
                    self.print_eval_metrics(self.loss.eval_metrics)

            if ((iteration == self.epochs) or (
                    (np.linalg.norm(delta_weights) + np.linalg.norm(delta_bias)) < self.epsilon)): break
            iteration += 1

    def AdaDelta(self, ro=0.5):

        iteration = 1
        A_weights = []
        A_bias = []

        d_weights = []
        d_bias = []

        for I in range(len(self.layers)):
            A_weights.append(np.zeros_like(self.layers[I].weights))
            A_bias.append(np.zeros_like(self.layers[I].bias))

            d_weights.append(np.random.random(self.layers[I].weights.shape))
            d_bias.append(np.random.random(self.layers[I].bias.shape))

        while True:
            self.flag0 = True
            for index in reversed(range(len(self.layers))):
                self.calculate_backprop(self.layers[index])  # 3shan e7sb el grad bta3 el weights w el bias
                delta_weights = np.zeros_like(self.layers[index].weights_gradients['W'])

                delta_bias = np.zeros_like(self.layers[index].bias_gradients['b'])

                delta_weights = self.layers[index].weights_gradients['W']
                delta_bias = self.layers[index].bias_gradients['b']

                A_weights[index] = A_weights[index] + np.square(delta_weights)
                A_bias[index] = A_bias[index] + np.square(delta_bias)

                triangle_w = np.multiply((np.sqrt(np.divide(d_weights, np.add(A_weights, self.epsilon)))),
                                         delta_weights)
                triangle_b = np.multiply((np.sqrt(np.divide(d_bias, np.add(A_bias, self.epsilon)))), delta_bias)

                d_weights = np.multiply(ro, d_weights) + np.multiply(1 - ro, np.square(triangle_w))
                d_bias = np.multiply(ro, d_bias) + np.multiply(1 - ro, np.square(triangle_b))

                shape_w = self.layers[index].weights.shape
                shape_b = self.layers[index].bias.shape

                weights = self.layers[index].weights - np.multiply(
                    np.sqrt(np.divide(d_weights, np.add(A_weights, self.epsilon))), delta_weights)
                weights = np.reshape(weights, shape_w)
                bias = self.layers[index].bias - np.multiply(np.sqrt(np.divide(d_bias, np.add(A_bias, self.epsilon))),
                                                             delta_bias)
                bias = np.reshape(bias, shape_b)
                print(f'iterations : \n {iteration} \n weights : \n{weights} \n bias : \n {bias}')

                self.layers[index].update_weights(weights, bias)
                if self.layers[index].last_layer:
                    self.loss.Y_hat = self.layers[index].out
                    print(f'\n Loss : {self.loss.out}')
                    self.print_eval_metrics(self.loss.eval_metrics)

            if ((iteration == self.epochs) or (
                    (np.linalg.norm(delta_weights) + np.linalg.norm(delta_bias)) < self.epsilon)): break
            iteration += 1

    def Adam(self, ro=0.5, ro_f=0.5):
        iteration = 1
        alpha = self.learning_rate
        A_weights = []
        A_bias = []

        F_weights = []
        F_bias = []

        for I in range(len(self.layers)):
            A_weights.append(np.zeros_like(self.layers[I].weights))
            A_bias.append(np.zeros_like(self.layers[I].bias))

            F_weights.append(np.zeros_like(self.layers[I].weights))
            F_bias.append(np.zeros_like(self.layers[I].bias))
        while True:
            self.flag0 = True
            for index in reversed(range(len(self.layers))):

                self.calculate_backprop(self.layers[index])  # 3shan e7sb el grad bta3 el weights w el bias

                delta_weights = self.layers[index].weights_gradients['W']
                delta_bias = self.layers[index].bias_gradients['b']

                A_weights[index] = np.multiply(ro, A_weights[index]) + np.multiply(1 - ro, np.square(delta_weights))
                A_bias[index] = np.multiply(ro, A_bias[index]) + np.multiply(1 - ro, np.square(delta_bias))

                F_weights[index] = np.multiply(ro, F_weights[index]) + np.multiply(1 - ro, np.square(delta_weights))
                F_bias[index] = np.multiply(ro_f, F_bias[index]) + np.multiply(1 - ro_f, np.square(delta_bias))

                alpha = np.multiply(alpha, np.sqrt((1 - ro) / (1 - ro_f)))

                shape_w = self.layers[index].weights.shape
                shape_b = self.layers[index].bias.shape

                weights = self.layers[index].weights - np.multiply(
                    np.sqrt(np.divide(alpha, np.add(A_weights, self.epsilon))), F_weights)
                weights = np.reshape(weights, shape_w)

                t1 = np.reshape(np.add(A_bias, self.epsilon), (A_bias[0].shape, A_bias.shape[1]))
                t2 = np.divide(alpha, t1)
                t3 = np.sqrt(t2)
                t4 = np.reshape(np.multiply(t3, F_bias), (t3.shape[0], t3.shape[1]))

                bias = self.layers[index].bias - t4
                bias = bias[0][0]
                bias = np.reshape(bias, (shape_b))
                print(f'iterations : \n {iteration} \n weights : \n{weights} \n bias : \n {bias}')

                self.layers[index].update_weights(weights, bias)
                if self.layers[index].last_layer:
                    self.loss.Y_hat = self.layers[index].out
                    print(f'\n Loss : {self.loss.out}')
                    self.print_eval_metrics(self.loss.eval_metrics)
            if ((iteration == self.epochs) or (
                    (np.linalg.norm(delta_weights) + np.linalg.norm(delta_bias)) < self.epsilon)): break
            iteration += 1

    def SteepestGradientDescent(self):
        delta = np.zeros((3, 2))

        while np.linalg.norm(delta) > self.epsilon:
            for j in range(0, self.N):
                delta = np.zeros((3, 2))
            for i in range(self.batch_size * j, self.batch_size):
                delta = delta + self.loss_gradient
            delta = delta / self.batch_size
            # self.learning_rate =0
            weights = weights - self.learning_rate * delta

    def NewtonRaphson(self):
        delta = np.zeros((3, 2))

        while np.linalg.norm(delta) > self.epsilon:
            for j in range(0, self.N):
                delta = np.zeros((3, 2))
            for i in range(self.batch_size * j, self.batch_size):
                delta = delta + self.loss_gradient
            delta = delta / self.batch_size
            # self.learning_rate =np.linalg.inv()
            weights = weights - self.learning_rate * delta

class Model:
    def __init__(self, layer_list=[], *args, **kwargs):
        self.layers_list = []
        self.layers_cache = []
        self.layers_weights = []
        self.layers_bias = []
        self.iterator = 0
        for l in layer_list:
            '''l.activation_fn'''
            l.activation_Fn = l()
            self.layers_list.append(l)
            self.layers_cache.append(l.cache)
            self.layers_weights.append(l.weights)
            self.layers_bias.append(l.bias)
        self.layers_no = len(self.layers_list)

    def split_Batches(self):

        if self.iterator == 0:
            self.total_Batch_Size = self.layers_list[0].inputs.shape[1]
            self.model_inputs = np.copy(self.layers_list[0].inputs)
            self.model_labels = np.copy(self.MyLoss.Loss_type.Y)
            self.mini_batch_size = self.total_Batch_Size / (np.power(2, int(np.log2(self.total_Batch_Size))))
            

        self.minibatch_input = np.copy(self.model_inputs[:,(int)(self.mini_batch_size * self.iterator) : int((self.mini_batch_size * self.iterator) + self.mini_batch_size )])
        self.minibatch_labels = np.copy(self.model_labels[:,(int)(self.mini_batch_size * self.iterator) : int((self.mini_batch_size * self.iterator) + self.mini_batch_size )])
        self.layers_list[0].inputs = np.copy(self.minibatch_input)
        self.MyLoss.Loss_type.Y = np.copy(self.minibatch_labels)
        self.iterator +=1
        self.iterator = self.iterator % (int)(self.total_Batch_Size / self.mini_batch_size)
        for l in self.layers_list:
            l()
        self.MyLoss.Loss_type.Y_hat = np.copy(self.layers_list[-1].out)
        self.MyLoss.Loss_type()

    def Evaluate(self, Test_Inputs, Test_Labels, print_value=False):
        Input = Test_Inputs
        last_output = 0
        for l in self.layers_list:
            old_inputs = l.inputs
            l.inputs = Input     # ba7sb el forward bta3 el test
            l()
            Input = l.out
            last_output = l.out
            l.inputs = old_inputs
            l()              # hena barag3 kol 7aga l sa7bha
        old_label = self.MyLoss.Y
        old_out = self.MyLoss.Y_hat
        self.MyLoss.Loss_type.Y = Test_Labels
        self.MyLoss.Loss_type.Y_hat = last_output
        self.MyLoss.Loss_type()                     # ba7sb el forward bta3 el test
        Loss = self.MyLoss.Loss_type.out
        self.MyLoss.Loss_type.Y_hat = old_out
        self.MyLoss.Loss_type.Y = old_label
        self.MyLoss.Loss_type()                   # hena barag3 kol 7aga l sa7bha
        if print_value:
            print(f'Y_Hat : {last_output}\n Loss : {Loss}')
        return last_output, Loss  # and return y_hat , loss


    def inference(self, Test_Inputs):

	    X = Test_Inputs
	    for layer in self.layers_list:
	        if (isinstance(layer, Linear)):
	            X = X.T
	        layer.cache['X'] = X
	        X = layer.forward_pass()

	    return X
    def AddLayer(self, L1):
        self.layers_list.append(L1)

    def Loss(self, Y, Y_hat):
        self.MyLoss = Loss(Y, Y_hat, self.layers_cache[-1], self.layers_weights[-1], self.layers_bias[-1])
        return self.MyLoss

    def Activation(self, Z):  # Z = W . X + b
        MyActivation = Activation(Z)
        return MyActivation

    def Optimization(self,epochs,Loss,learning_rate, epsilon,batch_type = 'Batch'):
        if batch_type == 'minibatch':
            self.split_Batches()
        MyOptimization = Optimization(self,self.layers_list, epochs,  Loss,learning_rate, epsilon,batch_type)
        return MyOptimization

class Plot():
    def __init__(self, name, colour):
        self.name = name
        self.colour = colour
        self.values = []

    def add_value(self, value):
        self.values.append(value)


class Visualizer():
    def __init__(self):
        self.iteration = 0
        self.fig = None
        self.ax = None
        self.plots = []

    def add_plot(self, name, colour):
        plot = Plot(name=name, colour=colour)
        self.plots.append(plot)

    # plot values are a list
    def update(self, plot_values):
        for i in range(len(self.plots)):
            self.plots[i].add_value(plot_values[i])

        # if first iteration
        if (self.iteration == 0):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
            plt.ion()
            self.fig.show()
            self.fig.canvas.draw()
            for plot in self.plots:
                self.ax.plot(plot.values, color=plot.colour, label=plot.name)
            legend = self.ax.legend()
            self.fig.canvas.draw()
        else:
            for plot in self.plots:
                self.ax.plot(plot.values, color=plot.colour, label=plot.name)
            self.fig.canvas.draw()

        self.iteration += 1


def train2img(image):
    c, h, w = image.shape
    image = image.reshape(h, w, c)
    return image


class Data:
    def __init__(self, path, test_frac, val_frac):
        self.path = path
        # load dataframe
        self.df = pd.read_csv(path)
        self.n_examples = self.df.shape[0]

        # shuffle it
        self.df = self.df.sample(frac=1)

        # split to train, test , validation
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.train_frac = 1 - self.test_frac + self.val_frac
        self.train_df = self.df

        self.test_df = self.train_df.sample(frac=self.test_frac)
        self.train_df = self.train_df.drop(self.test_df.index)

        val_number = self.n_examples * self.val_frac
        new_val_frac = val_number/self.train_df.shape[0]
        self.val_frac = new_val_frac
        self.val_df = self.train_df.sample(frac=self.val_frac)
        self.train_df = self.train_df.drop(self.val_df.index)

        # split labels and data
        self.train_labels = self.train_df.iloc[:, :1]
        self.train_data = self.train_df.iloc[:, 1:]

        self.test_labels = self.test_df.iloc[:, :1]
        self.test_data = self.test_df.iloc[:, 1:]

        self.val_labels = self.val_df.iloc[:, :1]
        self.val_data = self.val_df.iloc[:, 1:]

        # number of examples
        self.train_n_examples = self.train_df.shape[0]
        self.test_n_examples = self.test_df.shape[0]
        self.val_n_examples = self.val_df.shape[0]

        # delete unused df
        del(self.df)
        del(self.train_df)
        del(self.test_df)
        del(self.val_df)

        # convert to numpy
        self.train_data = self.train_data.to_numpy()
        self.train_labels = self.train_labels.to_numpy()

        self.test_data = self.test_data.to_numpy()
        self.test_labels = self.test_labels.to_numpy()

        self.val_data = self.val_data.to_numpy()
        self.val_labels = self.val_labels.to_numpy()


class Image(Data):
    def __init__(self, path, image_size, test_frac=0.2, val_frac=0.15, colour='rgb'):
        super().__init__(path, test_frac, val_frac)
        self.path = path
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.image_size = image_size
        self.colour = colour

        if (self.colour == 'gray'):
            self.cmap = 'gray'
            self.channels = 1
        elif (self.colour == 'rgb'):
            self.cmap = 'viridis'
            self.channels = 3

        # reshape to images
        self.train_data = self.train_data.reshape(
            (self.train_n_examples, self.channels, image_size[0], image_size[1]))/255.0
        self.train_labels = self.train_labels.reshape(
            (self.train_n_examples, 1))

        self.test_data = self.test_data.reshape(
            (self.test_n_examples, self.channels, image_size[0], image_size[1]))/255.0
        self.test_labels = self.test_labels.reshape((self.test_n_examples, 1))

        self.val_data = self.val_data.reshape(
            (self.val_n_examples, self.channels, image_size[0], image_size[1]))/255.0
        self.val_labels = self.val_labels.reshape((self.val_n_examples, 1))

    def get_img(self, image_number):
        img = self.train_data[image_number, :, :, :]
        c, h, w = img.shape
        img = img.reshape(h, w, c)
        return img

    def display_sample(self, samples=1):
        random_numbers = np.random.randint(
            low=0, high=self.train_n_examples - 1, size=samples)
        for number in random_numbers:
            img = self.get_img(number)
            plt.figure()
            plt.imshow(img, cmap=self.cmap)
