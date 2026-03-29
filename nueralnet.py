import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import sys







# He initialization we increase the variance because of Relu activation
def init_params():
    w1 = np.random.randn(10, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((10, 1))
    
    w2 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b2 = np.zeros((10, 1))
    
    w3 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b3 = np.zeros((10, 1))
    
    return w1, b1, w2, b2, w3, b3


# Relu activation because it's easy to differentiate and fast
# also the derivative is either 1 or zero 
def Relu_activation(z):
    return np.maximum(0, z)

# derivative is 1 if z > 0
# We use a boolean trick 
def Relu_deriv(z):
    return z > 0

# stable softmax
# subtract the largest number from z to prevent overflow then do the exponents so we don't get e^100000 and overflow into infinity
def softmax_activation(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    return np.exp(shift_z) / sum(np.exp(shift_z))

# The forward pass of the nueral network
# multiply the weights, add the biases and activate each layer
def forward_pass(w1, b1, w2, b2, w3, b3, x):
    l1 = np.dot(w1, x) + b1
    a1 = Relu_activation(l1)

    l2 = np.dot(w2, a1) + b2
    a2 = Relu_activation(l2)

    l3 = np.dot(w3, a2) + b3
    a3 = softmax_activation(l3)
    return l1, a1, l2, a2, l3, a3

    
# One hot encode y labels
def one_hot(y):
    y = y.astype(int)
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y] = 1 # arrange is basically range(), it creates a list from 0 to y.size 
    return one_hot_y.T
    

def back_prop(l1, a1, l2, a2, w2, l3, a3, w3, x, y):
    one_hot_y = one_hot(y)
    m = y.size
    
    # Output Layer Gradients
    dZ3 = a3 - one_hot_y # Shape (10, m) We subtract the y encoded from the output
    dw3 = 1 / m * np.dot(dZ3, a2.T) # Shape (10, 10)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True) # Shape (10, 1)

    dZ2 = np.dot(w3.T, dZ3) * Relu_deriv(l2)
    dw2 = 1/m * np.dot(dZ2, a1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(w2.T, dZ2) * Relu_deriv(l1) # Shape (10, m)
    dw1 = 1 / m * np.dot(dZ1, x.T) # Shape (10, 784)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True) # Shape (10, 1)

    return dw1, db1, dw2, db2, dw3, db3

# subtract because we want to go in the negative direction of the gradient to minimize loss so - - cancels out and becomes positive
def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3

    return w1, b1, w2, b2, w3, b3

def gradient_descent(X, Y, x_test, y_test, alpha, iterations, batch_size=64):
    w1, b1, w2, b2, w3, b3 = init_params()
    m = Y.size

    for i in range(iterations):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]
        for j in range(0, m, batch_size):
            x_batch = X_shuffled[:, j:j+batch_size]
            y_batch = Y_shuffled[j:j+batch_size]
            l1, a1, l2, a2, l3, a3 = forward_pass(w1, b1, w2, b2, w3, b3, x_batch)

            dw1, db1, dw2, db2, dw3, db3 = back_prop(l1, a1, l2, a2, w2, l3, a3, w3, x_batch, y_batch)
            
            w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)


        if i % 5 == 0:
            _, _, _, _, _, train_a3 = forward_pass(w1, b1, w2, b2, w3, b3, X)
            predictions = get_predictions(train_a3)
            test_accuracy, _ = test_model(w1, b1, w2, b2, w3, b3, x_test.T, y_test)
            print(f"Epoch: {i} | Train Accuracy: {get_accuracy(predictions, Y):.2%} | Test Accuracy {test_accuracy:.2%}")
            
    return w1, b1, w2, b2, w3, b3 

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
    return data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3']



def main(args):
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


    if args[1] == 'train':
        W1, b1, W2, b2, W3, b3, = gradient_descent(X_train, Y_train, X_test, Y_test, 0.1, 500)
        np.savez('mnist_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
        print("Model weights saved to mnist_weights.npz")
    elif args[1] == 'test':
        W1, b1, W2, b2, W3, b3 = load_trained_params('mnist_weights.npz')
        test_accuracy, y_hat = test_model(W1, b1, W2, b2, W3, b3, X_test.T, Y_test)
        print(f'Test Accuracy: {test_accuracy:.2%}')
        for i in range(5):
            visualize_network(np.random.randint(0, len(X_test)), X_test, Y_test, y_hat)
    else:
        print('''usage:
    python nueralnet.py 
options:
    test 
    train''')





if __name__ == '__main__':
    main(sys.argv)