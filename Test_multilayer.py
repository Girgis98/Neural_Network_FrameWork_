from Model import *

X = [[1, 0.1]]
Y = [0.6,0.01]
X = np.array(X).reshape(len(X), -1)
X=X.T
print(np.shape(X))
Y = np.reshape(Y, (2, 1))

# Parameters
learning_rate = 1
training_epochs = 20
epsilon = 0.002

n_input = 1
n_classes = 2

#
# # Store layers weight & bias
# weights = {'h1': np.array([[1, 0], [0, 1]]),
#              'h2': np.array([[1, 0], [0, 1]])}
# biases = {'b1': np.array([1, 1], ndmin=2).T,
#            'b2': np.array([1, 1], ndmin=2).T}
# Store layers weight & bias
weights = {'h1': np.array([[1, 0], [0, 1]]),
             'h2': np.array([[1, 0], [0, 1]])}
biases = {'b1': np.array([1, 1], ndmin=2).T,
           'b2': np.array([1, 1], ndmin=2).T}



Layer1 = Linear(X, weights['h1'], biases['b1'], 'sigmoid')

Layer2 = Linear(Layer1.output(), weights['h2'], biases['b2'], 'sigmoid')

my_Model = Model([Layer1,Layer2])

my_Loss = my_Model.Loss(Y, Layer2.output()).MSE()
opt = my_Model.Optimization(3, my_Loss, learning_rate, epsilon,'Batch').GradientDescent()

print("opa")