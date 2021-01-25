from Model import *
X = [[0,0],[0,1],[1,0],[2,0],[2,1],[3,0],[0,3],[0,4],[1,3]]
Y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) - 1

X = np.array(X).reshape(len(X), -1)
X=X.T
#Y = np.reshape(Y, (9, 1))
print(np.shape(Y))
Y = np.reshape(Y,(1,9))

# Parameters
learning_rate = [1,True]
training_epochs = 20
epsilon = 0.01

n_input = 1
n_classes = 2

# Store layers weight & bias
weights = {'h1': np.array([[1, 0], [0, 1],[0,0]]),
             'h2': np.array([[1, 0], [0, 1]])}
biases = {'b1': np.array([0,0,1], ndmin=2).T,
           'b2': np.array([1, 1], ndmin=2).T}



Layer1 = Linear(X, weights['h1'], biases['b1'], 'softmax')

#Layer2 = Linear(Layer1.output(), weights['h2'], biases['b2'], 'softmax')

my_Model = Model([Layer1])

my_Loss = my_Model.Loss(Y, Layer1.output()).CrossEntropy()
opt = my_Model.Optimization(3, my_Loss, learning_rate, epsilon,'Batch').GradientDescent()






print("opa")


