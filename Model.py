import numpy as np
from Losses import *
from Activations import Activation
from Optimizations import Optimization
from Layers import *


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
        self.model_inputs = np.copy(self.layers_list[0].inputs)
        self.split_flag=True
        #self.my_Optimizer


    def split_Batches(self):

        if self.split_flag:
            self.split_flag=False
            self.total_Batch_Size = np.shape(self.layers_list[0].inputs)[1]
            #self.my_Optimizer.N=self.total_Batch_Size
            self.model_inputs = np.copy(self.layers_list[0].inputs)
            self.model_labels = np.copy(self.MyLoss.Loss_type.Y)
            self.mini_batch_size =  int(np.log2(self.total_Batch_Size))
            if self.mini_batch_size == 0:
                raise Exception ("Cannot use mini batch on one training example")
            self.no_of_batches=((int)(self.total_Batch_Size / self.mini_batch_size)-1)

        self.minibatch_input = np.copy(self.model_inputs[:, (int)(self.mini_batch_size * self.iterator): int((self.mini_batch_size * self.iterator) + self.mini_batch_size - 1)])
        self.minibatch_labels = np.copy(self.model_labels[:,(int)(self.mini_batch_size * self.iterator): int((self.mini_batch_size * self.iterator) + self.mini_batch_size - 1)])
        self.layers_list[0].inputs = np.copy(self.minibatch_input)
        self.MyLoss.Loss_type.Y = np.copy(self.minibatch_labels)
        self.iterator +=1
        self.iterator = self.iterator % self.no_of_batches
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
        Eval_metrics = self.MyLoss.Loss_type.eval_metrics
        self.MyLoss.Loss_type.Y_hat = old_out
        self.MyLoss.Loss_type.Y = old_label
        self.MyLoss.Loss_type()                   # hena barag3 kol 7aga l sa7bha
        if print_value:
            print(f'Y_Hat : {last_output}\n Loss : {Loss}')
            self.my_Optimizer.print_eval_metrics(Eval_metrics,Loss)
        return last_output, Loss  # and return y_hat , loss

    def AddLayer(self, L1):
        self.layers_list.append(L1)

    def Loss(self, Y, Y_hat):
        self.MyLoss = Loss(Y, Y_hat, self.layers_cache[-1], self.layers_weights[-1], self.layers_bias[-1])

        return self.MyLoss

    def Activation(self, Z):  # Z = W . X + b
        MyActivation = Activation(Z)
        return MyActivation

    def Optimization(self,epochs,Loss,learning_rate, epsilon,batch_type = 'Batch'):
        self.model_labels = np.copy(self.MyLoss.Loss_type.Y)
        if batch_type == 'minibatch':
            self.split_Batches()
        MyOptimization = Optimization(self,self.layers_list, epochs,  Loss,learning_rate, epsilon,batch_type)
        self.my_Optimizer = MyOptimization
        return MyOptimization
