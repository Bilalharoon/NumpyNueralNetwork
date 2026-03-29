import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt







def init_params():
    # Layer 1: 784 inputs -> 10 neurons
    w1 = np.random.randn(10, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((10, 1))
    
    # Layer 2: 10 inputs -> 10 neurons
    w2 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b2 = np.zeros((10, 1))
    
    # Layer 3: 10 inputs -> 10 neurons
    w3 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b3 = np.zeros((10, 1))
    
    return w1, b1, w2, b2, w3, b3


def Relu_activation(z):
    return np.maximum(0, z)

def Relu_deriv(z):
    return z > 0

def softmax_activation(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    return np.exp(shift_z) / sum(np.exp(shift_z))

def forward_pass(w1, b1, w2, b2, w3, b3, x):
    l1 = np.dot(w1, x) + b1
    a1 = Relu_activation(l1)

    l2 = np.dot(w2, a1) + b2
    a2 = Relu_activation(l2)

    l3 = np.dot(w3, a2) + b3
    a3 = softmax_activation(l3)
    return l1, a1, l2, a2, l3, a3

    
def one_hot(y):
    y = y.astype(int)
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T
    
def back_prop(l1, a1, l2, a2, w2, l3, a3, w3, x, y):
    one_hot_y = one_hot(y)
    m = y.size
    
    # Output Layer Gradients
    dZ3 = a3 - one_hot_y # Shape (10, m)
    dw3 = 1 / m * np.dot(dZ3, a2.T) # Shape (10, 10)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True) # Shape (10, 1)

    dZ2 = np.dot(w3.T, dZ3) * Relu_deriv(l2)
    dw2 = 1/m * np.dot(dZ2, a1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(w2.T, dZ2) * Relu_deriv(l1) # Shape (10, m)
    dw1 = 1 / m * np.dot(dZ1, x.T) # Shape (10, 784)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True) # Shape (10, 1)

    return dw1, db1, dw2, db2, dw3, db3

def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3

    return w1, b1, w2, b2, w3, b3

def gradient_descent(X, Y, x_test, y_test, alpha, iterations):
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(iterations):
        l1, a1, l2, a2, l3, a3 = forward_pass(w1, b1, w2, b2, w3, b3, X)

        dw1, db1, dw2, db2, dw3, db3 = back_prop(l1, a1, l2, a2, w2, l3, a3, w3, X, Y)
        
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)


        if i % 10 == 0:
            predictions = get_predictions(a3)
            test_accuracy, y_hat = test_model(w1, b1, w2, b2, w3, b3, X_test.T, y_test)
            print(f"Iteration: {i} | Train Accuracy: {get_accuracy(predictions, Y):.2%} | Test Accuracy {test_accuracy:.2%}")
            
    return w1, b1, w2, b2, w3, b3, y_hat

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def test_model(w1, b1, w2, b2, w3, b3, x, y):

    _, _, _, _, _, a3 = forward_pass(w1, b1, w2, b2, w3, b3, x)
    predictions = get_predictions(a3)
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


W1, b1, W2, b2, W3, b3, y_hat = gradient_descent(X_train, Y_train, X_test, Y_test, 0.1, 500)
# w1, b1, w2, b2 = load_trained_params('mnist_weights.npz')

test_accuracy, y_hat = test_model(W1, b1, W2, b2, W3, b3, X_test.T, Y_test)
print(f'Test Accuracy: {test_accuracy:.2%}')

np.savez('mnist_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("Model weights saved to mnist_weights.npz")

for i in range(5):
    visualize_network(np.random.randint(0, len(X_test)), X_test, Y_test, y_hat)

