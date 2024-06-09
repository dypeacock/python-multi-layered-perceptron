import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Constructor Method for Multi-Layer Perceptron (MLP) Neural Network.

        Parameters:
        -----------
        input_size : int
            Number of input features (predictors) in the dataset.

        hidden_size : int
            Number of nodes in the hidden layer of the neural network.
            This determines the complexity and capacity of the network to learn patterns in the data.

        output_size : int
            Number of output nodes in the neural network.
            This corresponds to the number of desired outputs (predictands) in the dataset.

        Attributes:
        -----------
        weights_input_hidden : numpy.ndarray
            Matrix of weights connecting the input layer to the hidden layer.
            Dimensions: (input_size, hidden_size)
            Initialised randomly to capture initial relationships between input features and hidden nodes.

        bias_hidden : numpy.ndarray
            Bias vector for the hidden layer.
            Dimensions: (1, hidden_size)
            Initialised randomly to introduce asymmetry and flexibility in the network.

        weights_hidden_output : numpy.ndarray
            Matrix of weights connecting the hidden layer to the output layer.
            Dimensions: (hidden_size, output_size)
            Initialised randomly to capture initial relationships between hidden nodes and output.

        bias_output : numpy.ndarray
            Bias vector for the output layer.
            Dimensions: (1, output_size)
            Initialised randomly to introduce asymmetry and flexibility in the network.

        Note:
        -----
        The constructor initialises the weights and biases of the neural network randomly.
        These parameters will be updated during the training process using backpropagation.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def forward_pass(self, X):
        """
        This method performs a forward pass through the neural network.
        It computes the output of the neural network given an input X.
        It calculates the output of the hidden layer using the sigmoid activation
        function (self.relu) and then computes the final output of the network using the same activation function.
        """
        # Forward pass
        self.hidden_output = self.relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.relu(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward_pass(self, X, y, learning_rate):
        """
        This method performs the backward pass or backpropagation to update the weights and biases of the network
        based on the computed error. It calculates the error between the predicted output and the actual output,
        computes the gradients of the error with respect to the weights and biases, and updates them using
        gradient descent.
        """
        # Backward pass
        error = y - self.output
        delta_output = error * self.relu_derivative(self.output)
        
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.relu_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, delta_hidden) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """
        This method trains the neural network using the provided input-output pairs X and y for a specified number of epochs.
        It iterates through the dataset, performs forward and backward passes for each input-output pair,
        and prints the loss at regular intervals.
        """
        losses = []  # To store loss values for each epoch
        for epoch in range(epochs):
            for i in range(len(X)):
                # Forward pass
                output = self.forward_pass(X[i])
                
                # Backward pass
                self.backward_pass(X[i].reshape(1, -1), y[i].reshape(1, -1), learning_rate)
                
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.forward_pass(X)))
                print(f'Epoch {epoch}, Loss: {loss}')
                losses.append(loss)

        # Plotting loss vs. epoch
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, epochs, 100), losses, color='blue')  # Plotting losses every 100 epochs
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evolution of Loss Rates')
        plt.grid(True)
        plt.show()

    def calculate_msre(self, y_true, y_pred):
        return np.mean(np.square((y_true - y_pred) / y_true))

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)


    def relu_derivative(self, x):
        # Derivative of ReLU activation function
        return np.where(x > 0, 1, 0)

# Example usage
if __name__ == "__main__":
    # Read data from Unstandardised Excel file
    file = pd.ExcelFile('FEHDataStudent.xlsx')
    with pd.ExcelFile('FEHDataStudent.xlsx') as xls:
        df1 = pd.read_excel(xls, "Sheet1", usecols=[0,1,2,3,4,5,6,7,8])
        #df1 = pd.read_excel(xls, "Sheet1", usecols=[0,1,5])

    # Prepare your dataset: 
    X = df1.iloc[:, :-1].values  # Features : This selects all rows (:) and all columns except the last one (:-1) from the DataFrame df1.
    #.values: This converts the selected DataFrame (df1.iloc[:, :-1]) into a NumPy array. 
    y = df1.iloc[:, -1].values.reshape(-1, 1)  # Target variable, reshape to ensure it's a column vector


    X_min = X.min()
    X_max = X.max()

    y_min = y.min()
    y_max = y.max()

    X_scaled = 0.8 * ((X - X_min) / (X_max - X_min)) + 0.1
    y_scaled = 0.8 * ((y - y_min) / (y_max - y_min)) + 0.1



    # Initialize MLP
    mlp = MLP(input_size=X.shape[1], hidden_size=1, output_size=1)  # Adjust input_size according to the number of features

    # Train MLP
    mlp.train(X_scaled, y_scaled, epochs=10000, learning_rate=0.1)

    # Test trained MLP
    print("Predictions after training:")
    predictions_scaled = mlp.forward_pass(X_scaled)
    
    # De-standardise predictions
    predictions = (predictions_scaled - 0.1) / 0.8 * (X_max - X_min) + X_min
    #print(predictions)

    # Calculate MSRE for final predictions
    msre = mlp.calculate_msre(y, predictions)
    print("Mean Squared Relative Error for Final Predictions:", msre)

    # Plot predicted values against actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, color='red')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='blue', linestyle='--')  # Diagonal line representing perfect prediction
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    plt.grid(True)
    plt.show()

