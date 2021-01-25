from Model import *

X = [[-1, -1], [1, -1], [-1, 1], [1, 1],[1, 1],[1, 1]]
Y = [[1, 1, 1, -1 , 1 , -1]]
X = np.array(X).reshape(len(X), -1)
X = X.T
print(np.shape(X))
Y = np.reshape(Y, (1, 6))

print(f'X : {X}')
print(f'Y : {Y}')

# Parameters
learning_rate = [0.1, True]
training_epochs = 20
epsilon = 0.0000002

# Network Parameters
# n_hidden_1 = 1  # 1st layer number of neurons
# n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 4  # MNIST data input (img shape: 28*28)
n_classes = 1  # MNIST total classes (0-9 digits)

# Store layers weight & bias
# weights = {
#     'h1': np.zeros((n_classes,2))
#     # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     # 'out': (np.random.rand(n_hidden_1, n_classes))
#     }
#
# biases = {
#     'b1': np.zeros((n_classes,1))
#     # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     # 'out': (np.random.rand(n_classes))
# }
# print(f'bias : {biases}')
# print(f'weights : {weights}')

# w1 = np.random.random((3,4))
w2 = np.random.random((2, 2))
w3 = np.random.random((2, 2))
w30 = np.random.random((2, 2))
w31 = np.random.random((2, 2))
w32 = np.random.random((2, 2))
w4 = np.random.random((1, 2))

# b1 = np.random.random((4,1))
b2 = np.random.random((2, 1))
b3 = np.random.random((2, 1))
b30 = np.random.random((2, 1))
b31 = np.random.random((2, 1))
b32 = np.random.random((2, 1))
b4 = np.random.random((1, 1))

# Layer1 = Linear(X, w1, b1, 'sigmoid')
Layer2 = Linear(X, w2, b2, 'relu')
Layer3 = Linear(Layer2.output(), w3, b3, 'relu')
Layer30 = Linear(Layer3.output(), w30, b30, 'relu')
Layer31 = Linear(Layer30.output(), w31, b31, 'relu')
Layer32 = Linear(Layer31.output(), w32, b32, 'relu')
Layer4 = Linear(Layer32.output(), w4, b4, 'relu')
my_Model = Model([Layer2, Layer3, Layer4])

# Layer1()
my_Loss = my_Model.Loss(Y, Layer4.output()).LogisticRegressionIdentity()
opt = my_Model.Optimization(100, my_Loss, learning_rate, epsilon).GradientDescent()
my_Model.Evaluate(X, Y, True)

print("opa")
