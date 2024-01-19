import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Display one image for each digit
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(20, 4))
for digit in range(10):
    digit_indices = np.where(y_train == digit)[0]
    ax[digit].imshow(255 - X_train[digit_indices[0]], cmap='gray')
    ax[digit].set_title(f'Digit {digit}')
    ax[digit].axis('off')

plt.tight_layout()
plt.show()



# Reshape and normalize input data
X_train_reshape = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_reshape = X_test.reshape(X_test.shape[0], -1) / 255.0

# Transpose labels if needed
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False, categories='auto')
y_train_onehot = encoder.fit_transform(y_train)
y_test_onehot = encoder.transform(y_test)

# Select a subset of data
m_train = 5000
m_test = 1000

X_train_reshape = X_train_reshape[:m_train, :]
X_test_reshape = X_test_reshape[:m_test, :]
y_train_onehot = y_train_onehot[:m_train, :]
y_test_onehot = y_test_onehot[:m_test, :]

print(X_train_reshape.shape)
print(X_test_reshape.shape)
print(y_train_onehot.shape)
print(y_test_onehot.shape)




# Neural network initialization function
def initialization(dimensions):
    np.random.seed(seed=0)
    parameters = {}
    C = len(dimensions)
    for c in range(1, C):
        parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c-1])
        parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)
    return parameters

# Neural network forward propagation function
def forward_propagation(X, parameters):
    activations = {'A0': X}
    C = len(parameters) // 2
    for c in range(1, C + 1):
        Z = parameters['W' + str(c)].dot(activations['A' + str(c-1)]) + parameters['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
    return activations

# Neural network backpropagation function
def back_propagation(X, y, activations, parameters):
    m = y.shape[1]
    C = len(parameters) // 2
    dZ = activations['A' + str(C)] - y
    gradients = {}
    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = (1/m) * np.dot(dZ, activations['A' + str(c-1)].T)
        gradients['db' + str(c)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parameters['W' + str(c)].T, dZ) * activations['A' + str(c-1)] * (1 - activations['A' + str(c-1)])
    return gradients

# Neural network update function
def update(gradients, parameters, learning_rate):
    C = len(parameters) // 2
    for c in range(1, C + 1):
        parameters['W' + str(c)] = parameters['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parameters['b' + str(c)] = parameters['b' + str(c)] - learning_rate * gradients['db' + str(c)]
    return parameters

# Neural network prediction function
def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    C = len(parameters) // 2
    Af = activations['A' + str(C)]
    return (Af >= 0.5).astype(int)

# Neural network training function
def neural_network(X, y, hidden_layers=(100, 100), learning_rate=0.1, n_iter=1000):
    np.random.seed(0)
    dimensions = [X.shape[0]] + list(hidden_layers) + [y.shape[0]]
    parameters = initialization(dimensions)
    train_loss = []
    train_acc = []
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parameters)
        gradients = back_propagation(X, y, activations, parameters)
        parameters = update(gradients, parameters, learning_rate)
        if i % 10 == 0:
            C = len(parameters) // 2
            train_loss.append(log_loss(y, activations['A' + str(C)]))
            y_pred = predict(X, parameters)
            current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()
    return parameters

# Train the neural network
parameters = neural_network(X_train_reshape.T, y_train_onehot.T, hidden_layers=(100, 100), learning_rate=0.1, n_iter=5000)

# Predict on test data
y_pred = predict(X_test_reshape.T, parameters)

# Print accuracy on test data
test_accuracy = accuracy_score(y_test_onehot.flatten(), y_pred.flatten())
print(f"Test Accuracy: {test_accuracy}")
