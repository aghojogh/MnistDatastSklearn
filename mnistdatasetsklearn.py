# MNIST Classification Project with Scikit-Learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. Load MNIST Dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

# Convert to numpy arrays and normalize
X = np.array(X) / 255.0  # Normalize pixel values to [0, 1]
y = np.array(y).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 3. Visualize some samples
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# 4. Train Multiple Models

# Model 1: Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=100, solver='saga', random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Model 2: Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Model 3: Support Vector Machine (on subset for speed)
print("\nTraining SVM (on 10k samples)...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train[:10000], y_train[:10000])
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Model 4: Neural Network (MLP)
print("\nTraining Neural Network...")
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), 
                          max_iter=20, 
                          random_state=42,
                          verbose=True)
mlp_model.fit(X_train, y_train)
mlp_pred = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
print(f"Neural Network Accuracy: {mlp_accuracy:.4f}")

# 5. Detailed Evaluation (using Random Forest as best model)
print("\n" + "="*50)
print("DETAILED EVALUATION - Random Forest")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 6. Visualize Predictions
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    idx = i
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    pred_label = rf_pred[idx]
    true_label = y_test[idx]
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
    ax.axis('off')
plt.tight_layout()
plt.show()

# 7. Feature Importance (Random Forest)
feature_importance = rf_model.feature_importances_.reshape(28, 28)
plt.figure(figsize=(8, 6))
plt.imshow(feature_importance, cmap='hot', interpolation='nearest')
plt.colorbar(label='Importance')
plt.title('Feature Importance Heatmap - Random Forest')
plt.xlabel('Pixel Column')
plt.ylabel('Pixel Row')
plt.show()

# 8. Model Comparison
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Neural Network']
accuracies = [lr_accuracy, rf_accuracy, svm_accuracy, mlp_accuracy]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'])
plt.ylabel('Accuracy')
plt.title('Model Comparison on MNIST')
plt.ylim([0.85, 1.0])
for i, (model, acc) in enumerate(zip(models, accuracies)):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("Project Complete!")
print("="*50)
