import numpy as np
import matplotlib.pyplot as plt

class HousePricePredictor:
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Store training history
        self.losses = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output, learning_rate=0.01):
        m = X.shape[0]
        
        # Calculate gradients
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((output - y) ** 2)
            self.losses.append(loss)
            
            # Backward propagation
            self.backward(X, y, output, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X)

# Generate sample data
np.random.seed(42)
n_samples = 100

# Features: house size (normalized) and location score (0-1)
house_sizes = np.random.uniform(500, 3000, n_samples)
location_scores = np.random.uniform(0.1, 1.0, n_samples)

# Normalize house sizes
house_sizes_norm = (house_sizes - house_sizes.min()) / (house_sizes.max() - house_sizes.min())

# Create features matrix
X = np.column_stack([house_sizes_norm, location_scores])

# Generate target: price depends on both size and location
# Price = base_price + size_factor * size + location_factor * location + noise
base_price = 200000
size_factor = 300000
location_factor = 200000
noise = np.random.normal(0, 25000, n_samples)

prices = base_price + size_factor * house_sizes_norm + location_factor * location_scores + noise

# Normalize prices for training
prices_norm = (prices - prices.min()) / (prices.max() - prices.min())
y = prices_norm.reshape(-1, 1)

# Create and train the model
model = HousePricePredictor()
print("Training neural network...")
model.train(X, y, epochs=2000, learning_rate=0.1)

# Make predictions
predictions = model.predict(X)

# Convert predictions back to actual prices
predictions_actual = predictions.flatten() * (prices.max() - prices.min()) + prices.min()

# Calculate accuracy
mse = np.mean((predictions_actual - prices) ** 2)
print(f"\nFinal Mean Squared Error: ${mse:.2f}")
print(f"Average prediction error: ${np.sqrt(mse):.2f}")

# Test with a new house
test_house = np.array([[0.7, 0.8]])  # 70% size, 80% location score
prediction = model.predict(test_house)
predicted_price = prediction[0][0] * (prices.max() - prices.min()) + prices.min()
print(f"\nPrediction for new house: ${predicted_price:.2f}")
