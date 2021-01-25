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

# Store layers weight & bias
weights = {'h1': np.random.random((5,2)),
             'h2': np.random.random((2,5))}
biases = {'b1': np.random.random((5,1)),
           'b2': np.random.random((2,1))}

Layer1 = Linear(X, weights['h1'], biases['b1'], 'sigmoid')

Layer2 = Linear(Layer1.output(), weights['h2'], biases['b2'], 'sigmoid')

my_Model = Model([Layer1,Layer2])

my_Loss = my_Model.Loss(Y, Layer2.output()).MSE()
opt = my_Model.Optimization(10, my_Loss,learning_rate,epsilon).GradientDescent()

print("opa")