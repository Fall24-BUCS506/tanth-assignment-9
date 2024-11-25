import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.activations = None
        self.gradients = None

    def activate(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation function")

    def activate_derivative(self, Z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(Z) ** 2
        elif self.activation_fn == 'relu':
            return (Z > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
            return A * (1 - A)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activate(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid for output
        self.activations = {'A1': self.A1, 'A2': self.A2}
        return self.A2


    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = X.shape[0]
        dZ2 = self.activations['A2'] - y
        dW2 = np.dot(self.activations['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.activate_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        # TODO: update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        # TODO: store gradients for visualization
        self.gradients = {'dW1': dW1, 'dW2': dW2}
        
def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.activations['A1']
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    # TODO: Hyperplane visualization in the hidden space

    # TODO: Distorted input space transformed by the hidden layer

    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.3, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    grad_magnitude = np.linalg.norm(mlp.gradients['dW1'], axis=1)
    for i, mag in enumerate(grad_magnitude):
        ax_gradient.add_patch(Circle((mlp.W1[0, i], mlp.W1[1, i]), mag, color='r', alpha=0.5))

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)