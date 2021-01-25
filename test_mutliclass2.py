from Model import *

X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [2, 0],
        [2, 1],
        [3, 0],
        [0, 3],
        [0, 4],
        [1, 3],
    ]
)
X=X.T
print(np.shape(X))
y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) - 1
y=np.reshape(y,(np.shape(y)[0],-1))

n_classes = 3
n_features= 2
weights = {
    'h1': np.array([[1,0],[0,1],[0,0]])
    }

biases = {
    'b1': np.array([[0],[0],[1]])

}
learning_rate = [1,False]
training_epochs = 20
epsilon = 0.2

Layer1 = Linear(X, weights['h1'], biases['b1'], 'softmax')
my_Model = Model([Layer1])


my_Loss = my_Model.Loss(y, Layer1.output()).CrossEntropy()
opt = my_Model.Optimization(10, my_Loss,learning_rate,epsilon).GradientDescent()

