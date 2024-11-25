# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our custom deep learning library components
from YFlow import Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam, Model

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary classification: Setosa (1) vs Others (0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape y to have shape (n_samples, 1) as required by our library
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create the model
model = Model()

# Add layers to the model with regularization
# Input layer: 4 features (sepal length, sepal width, petal length, petal width)
# First hidden layer: 16 units with ReLU activation
model.add(Dense(4, 16, regularization='l2', reg_strength=0.01))
model.add(ReLU())

# Second hidden layer: 8 units with ReLU activation
model.add(Dense(16, 8, regularization='l2', reg_strength=0.01))
model.add(ReLU())

# Output layer: 1 unit with Sigmoid activation (for binary classification)
model.add(Dense(8, 1, regularization='l1', reg_strength=0.005))
model.add(Sigmoid())

# Compile the model
# Use Binary Cross-Entropy loss (suitable for binary classification)
# Use Adam optimizer with learning rate of 0.001
model.compile(loss=BinaryCrossEntropy(), optimizer=Adam(learning_rate=0.001))

# Train the model
# Use 2000 epochs and a batch size of 16
model.train(X_train, y_train, epochs=2000, batch_size=16)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions to binary (0 or 1) using a threshold of 0.5
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate and print the accuracy
accuracy = np.mean(y_pred_binary == y_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Optional: Print some example predictions
print("\nSample predictions:")
for i in range(5):
    print(f"True: {y_test[i][0]}, Predicted: {y_pred[i][0]:.4f}, Binary Prediction: {y_pred_binary[i][0]}")