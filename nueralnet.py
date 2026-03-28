import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt







def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def Relu_activation(z):
    return np.maximum(0, z)

def Relu_deriv(z):
    return z > 0

def softmax_activation(z):
    return np.exp(z) / sum(np.exp(z))

def forward_pass(w1, b1, w2, b2, x):
    l1 = np.dot(w1, x) + b1
    a1 = Relu_activation(l1)
    l2 = np.dot(w2, a1) + b2
    a2 = softmax_activation(l2)

    return l1, a1, l2, a2

    
def one_hot(y):
    y = y.astype(int)
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T
    
def back_prop(l1, a1, l2, a2, w2, x, y): # Removed w1, added w2, l1
    one_hot_y = one_hot(y)
    m = y.size
    
    # Output Layer Gradients
    dZ2 = a2 - one_hot_y # Shape (10, m)
 # Shape (10, 1)

    dZ1 = np.dot(w2.T, dZ2) * Relu_deriv(l1) # Shape (10, m)
    dw1 = 1 / m * np.dot(dZ1, x.T) # Shape (10, 784)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True) # Shape (10, 1)

    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def gradient_descent(X, Y, x_test, y_test, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        l1, a1, l2, a2 = forward_pass(w1, b1, w2, b2, X)

        dw1, db1, dw2, db2 = back_prop(l1, a1, l2, a2, w2, X, Y)
        
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)


        if i % 10 == 0:
            predictions = get_predictions(a2)
            test_accuracy, y_hat = test_model(w1, b1, w2, b2, X_test, y_test)
            print(f"Iteration: {i} | Train Accuracy: {get_accuracy(predictions, Y):.2%} | Test Accuracy: {test_accuracy:.2%}")
            
    return w1, b1, w2, b2, y_hat

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def test_model(w1, b1, w2, b2, x, y):

    _, _, _, a2 = forward_pass(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    accuracy = get_accuracy(predictions, y)
    return accuracy, predictions


def visualize_network(k, x, y, y_hat):
    k = np.clip(k, 0, x.size)
    plt.imshow(x[k].reshape(28, 28) * 255.0, cmap='gray')
    print(f"Y: {y[k]} | Y_hat: {y_hat[k]}")
    plt.axis('off')
    plt.show()


def load_trained_params(file_path):
    data = np.load(file_path)
    return data['W1'], data['b1'], data['W2'], data['b2']


mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype('float32') / 255.0
y = mnist.target.to_numpy().astype('int')

del mnist


split = int(len(X)*0.8)
X_train = X[:split]
X_test = X[split:]
Y_train = y[:split]
Y_test = y[split:]


X_train = X_train.T


# W1, b1, W2, b2, y_hat = gradient_descent(X_train, Y_train, X_test.T, Y_test, 0.10, 500)
w1, b1, w2, b2 = load_trained_params('mnist_weights.npz')
test_accuracy, y_hat = test_model(w1, b1, w2, b2, X_test.T, Y_test)
print(f'Test Accuracy: {test_accuracy:.2%}')
# np.savez('mnist_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)
# print("Model weights saved to mnist_weights.npz")

for i in range(5):
    visualize_network(np.random.randint(0, len(X_test)), X_test, Y_test, y_hat)

