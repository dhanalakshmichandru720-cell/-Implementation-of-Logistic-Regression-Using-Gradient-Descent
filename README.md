# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Generate synthetic dataset
# -----------------------------
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, flip_y=0.1, random_state=42
)
y = y.reshape(-1, 1)  # column vector

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 2. Sigmoid function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# 3. Logistic Regression Training
# -----------------------------
def logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    bias = 0
    losses = []

    for _ in range(epochs):
        # Linear model
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        # Loss (Binary Cross-Entropy)
        loss = - (1/m) * np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
        losses.append(loss)

        # Gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        # Update parameters
        weights -= lr * dw
        bias -= lr * db

    return weights, bias, losses

# Train model
weights, bias, losses = logistic_regression(X_train, y_train, lr=0.1, epochs=1000)

# -----------------------------
# 4. Prediction function
# -----------------------------
def predict(X, weights, bias):
    y_pred = sigmoid(np.dot(X, weights) + bias)
    return (y_pred >= 0.5).astype(int)

# Accuracy
y_pred_train = predict(X_train, weights, bias)
y_pred_test = predict(X_test, weights, bias)
train_acc = np.mean(y_pred_train == y_train) * 100
test_acc = np.mean(y_pred_test == y_test) * 100

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# -----------------------------
# 5. Visualization
# -----------------------------

# Loss curve
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses, color='blue')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# Decision boundary
plt.subplot(1,2,2)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train.ravel(), cmap='bwr', alpha=0.7)
x_values = [np.min(X_train[:, 0] - 1), np.max(X_train[:, 0] + 1)]
y_values = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, y_values, label='Decision Boundary', color='black')
plt.title("Decision Boundary (Train Data)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()

Developed by: dhanalakshmi.c
RegisterNumber:  25018616
*/
```

## Output:
<img width="1572" height="723" alt="Screenshot (150)" src="https://github.com/user-attachments/assets/7545d221-5c2d-40e0-b8db-ca281bf8d6d4" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

