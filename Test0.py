from Model import *

X = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
Y = [1, 1, 1, -1]
X = np.array(X).reshape(len(X), -1)
X = X.T
Y = np.reshape(Y, (1, 4))

print(f'X : {X}')
print(f'Y : {Y}')

# Parameters
learning_rate = [1, False]
training_epochs = 20
epsilon = 0.00002

# Network Parameters
# n_hidden_1 = 1  # 1st layer number of neurons
# n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 4  # MNIST data input (img shape: 28*28)
n_classes = 1  # MNIST total classes (0-9 digits)

# Store layers weight & bias
weights = {
    'h1': np.zeros((n_classes, 2))
    # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'out': (np.random.rand(n_hidden_1, n_classes))
}

biases = {
    'b1': np.zeros((n_classes, 1))
    # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'out': (np.random.rand(n_classes))
}
# print(f'bias : {biases}')
# print(f'weights : {weights}')

Layer1 = Linear(X, weights['h1'], biases['b1'], 'sigmoid')
my_Model = Model([Layer1])

# Layer1()
my_Loss = my_Model.Loss(Y, Layer1.output()).LogisticRegressionSigmoid()
opt = my_Model.Optimization(10, my_Loss, learning_rate, epsilon,'minibatch').GradientDescent()
# my_Model.split_Batches()
my_Model.Evaluate(X, Y, True)

print("opa")
