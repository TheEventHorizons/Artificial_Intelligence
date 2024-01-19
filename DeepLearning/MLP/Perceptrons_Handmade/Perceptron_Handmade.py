import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate points for class 0
class0_points = np.random.normal(loc=[3, 6], scale=[1, 1], size=(200, 2))

# Generate points for class 1
class1_points = np.random.normal(loc=[7, 3], scale=[1, 1], size=(200, 2))

M0 = np.zeros((class0_points.shape[0], 1))
M1 = np.ones((class1_points.shape[0], 1))

X = np.concatenate([class0_points, class1_points], axis=0)
y = np.concatenate([M0, M1], axis=0)


# Create an initialization function to initialize parameters W and b of our model
def initialization(X):
    np.random.seed(seed=0)
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W, b

W, b = initialization(X)

print(W.shape)

# Next, create an iterative algorithm where we repeat the following functions in a loop:

# Start with the function that represents our artificial neuron model, where we find the function Z = X.W + b and the activation function A
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

A = model(X, W, b)

# Next, create an evaluation function, i.e., the cost function that evaluates the model's performance by comparing the output A to the reference data y
def log_loss(A, y):
    return (1 / len(y)) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

# In parallel, calculate the gradients of this cost function
def gradients(A, X, y):
    dW = (1 / len(y)) * np.dot(X.T, A - y)
    db = (1 / len(y)) * np.sum(A - y)
    return dW, db

dW, db = gradients(A, X, y)

print(dW.shape)
print(db.shape)

# Finally, use these gradients in an update function that updates the parameters W and b to reduce the model's errors
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

# Create a prediction function
def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

# Create an artificial neuron
def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    # Initialize W, b
    W, b = initialization(X)

    # Visualize the loss
    Loss = []

    # Create a training loop
    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    # Calculate predictions for all data
    y_pred = predict(X, W, b)

    # Display the performance of our model (e.g., accuracy), comparing the reference data y with our predictions
    # print(accuracy_score(y, y_pred))

    # Visualize the loss to see if our model has learned well
    plt.plot(Loss)
    plt.grid(ls='--')
    plt.show()
    return W, b

# Train the artificial neuron
W, b = artificial_neuron(X, y)

# Test on new data
new_data = np.array([6, 4])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', label='Class 0 (Blue) / Class 1 (Orange)')
plt.scatter(new_data[0], new_data[1], c='red', label='New Data (Red)')
plt.title('Classification with an Artificial Neuron')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(ls='--')
plt.legend()
plt.show()

prediction = predict(new_data, W, b)
print(f"Prediction for new data: {prediction}")

# Create the decision boundary
x0 = np.linspace(-1, 11, 100)
x1 = (-W[0] * x0 - b) / W[1]

plt.plot(x0, x1, c='red', lw=2, ls='--', label='Decision Boundary')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', label='Class 0 (Green) / Class 1 (Grey)')
plt.title('Classification with a Perceptron and Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(ls='--')
plt.legend()
plt.show()
