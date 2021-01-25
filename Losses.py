import numpy as np
from Function import Function
import math


class Loss(Function):
    def __init__(self, Y, Y_hat, cache, weights, bias, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Y_hat = np.array(Y_hat)
        self.Y = np.array(Y)
        # self.Y = np.reshape(self.Y, np.shape(self.Y_hat))

        self.flag = 0

        self.layer_cache = cache
        self.layer_weights = weights
        self.layer_bias = bias

        self.eval_metrics = {}

        self.Loss_Value = self.forward_pass()
        self.local_gradient()

    def F1_score(self, y, y_cap, binary_or_multi):  # binary = True ,  Multi = False

        if binary_or_multi:
            # y_cap = (np.round(y))
            # y_cap=y_cap.astype(int)
            y_index = np.greater_equal(y_cap, 0.5)
            y_cap[y_index] = 1
            y_cap[np.invert(y_index)] = 0
            y_cap = y_cap.astype(int)

            index2 = np.equal(-1, y)
            y[index2] = 0

            y_cap_0 = np.equal(y_cap, 0)
            y_cap_1 = np.equal(y_cap, 1)
            y_0 = np.equal(y, 0)
            y_1 = np.equal(y, 1)

            TP = np.sum(np.equal(y_1, y_cap_1))
            TN = np.sum(np.equal(y_0, y_cap_0))
            FP = np.sum(np.equal(y_0, y_cap_1))
            FN = np.sum(np.equal(y_1, y_cap_0))

            acc = (TP + TN) / (TP + TN + FP + FN)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            specificity = TN / (TN + FP)

            F1 = (2 * precision * recall) / (precision + recall)
            self.eval_metrics = {'accuracy': acc, 'recall': recall, 'precision': precision, 'specificity': specificity,
                                 'F1 score': F1}
            return self.eval_metrics

        elif not binary_or_multi:  # T and F only

            y_cap_max = np.argmax(y_cap, axis=0)  # 1 x N
            y_cap_max = np.reshape(y_cap_max, (np.shape(y)))
            result = np.equal(y, y_cap_max)

            acc = (np.sum(result)) / (np.shape(y_cap)[1])

            self.eval_metrics = {'accuracy': acc}
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
        # self.grad = {'dL': (- (np.dot(self.Y, self.layer_cache['X'])))}
        self.grad = {'dL': (- self.Y) / self.Y_hat.shape[1]}
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
        # self.grad = {'dL': (- (np.dot(self.Y, self.cache['X'])))}  # X is not defined here yet
        self.grad = {'dL': (- self.Y) / self.Y_hat.shape[1]}
        return self.grad


class LogisticRegressionSigmoid(Loss):
    def forward_pass(self):
        L = np.mean(-np.log(np.absolute((self.Y / 2) - (1 / 2) + self.Y_hat) + 0.001), axis=1, keepdims=True)
        y = np.copy(self.Y)
        yhat = np.copy(self.Y_hat)
        eval_metric = self.F1_score(y, yhat, True)
        self.cache['Loss'] = L
        return L

    def local_gradient(self):
        # self.grad = {'dL': (- (np.multiply(self.Y, self.layer_cache['X'])) / (1 + np.exp(np.multiply(self.Y, np.dot(self.layer_weights,self.layer_cache['X'])))))}
        self.grad = {'dL': (-1 / ((self.Y / 2) - (1 / 2) + self.Y_hat + 0.001)) / self.Y_hat.shape[1]}

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


class CrossEntropy(Loss):  #######33 used for softmax only

    def forward_pass(self):
        # print("dy el y hat ablloss",self.Y_hat)
        for i in range(np.shape(self.Y_hat)[1]):
            if self.Y_hat[(self.Y[:,i]), i] <= 0:
                log_Y_hat = -np.log(0.001)
            else:
                log_Y_hat = -np.log(self.Y_hat[(self.Y[:,i]), i])

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

        for i in range(np.shape(self.Y_hat)[1]):
            self.Y_hat[self.Y[:,i], i] = self.Y_hat[self.Y[:,i], i] - 1

        self.grad = {'dL': self.Y_hat / np.shape(self.Y_hat)[1]}
        # self.grad = {'dL': (self.Y_hat - np.ones_like(self.Y_hat)) / float(len(self.Y))} ############           need to check for yi!=r
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


