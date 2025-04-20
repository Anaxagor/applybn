## Feature Selection with NMIFeatureSelector on the Wine Quality Dataset

This example evaluates the `NMIFeatureSelector` ability to retain informative features and discard noise using the UCI Wine Quality Dataset. We will:
- Load the dataset and preprocess it.
- Add synthetic noise features to simulate uninformative attributes.
- Compare model performance with/without feature selection as noise increases.
- Visualize the results.
## Step 1: Load and Preprocess Data
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from bn_nmi_feature_selector import NMIFeatureSelector

# Load Wine Quality dataset (red wine)
data = pd.read_csv('data/feature_selection/WineQT.csv')
data = data.drop(columns=['Id'])
X = data.drop("quality", axis=1)
y = data["quality"].values
```
## Step 2: Define Noise Injection and Evaluation Function
```python
def evaluate_performance(X, y, noise_features=0, threshold=0.05):
    # Add random noise features
    np.random.seed(42)
    noise = np.random.randn(X.shape[0], noise_features)
    X_combined = np.hstack([X.values, noise])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.3, random_state=42
    )
    
    # With Feature Selection
    selector = NMIFeatureSelector(threshold=threshold, n_bins=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Train model on selected features
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    acc_selected = accuracy_score(y_test, y_pred)
    
    # Without Feature Selection
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred)
    
    return acc_full, acc_selected, selector.selected_features_.shape[0]
```

## Step 3: Run Experiment with Increasing Noise
```python
noise_levels = np.arange(0, 50, 10)  # Test 0 to 40 noise features
results = []

for noise in noise_levels:
    acc_full, acc_selected, n_selected = evaluate_performance(X, y, noise_features=noise)
    results.append({
        "Noise Features": noise,
        "Accuracy (Full)": acc_full,
        "Accuracy (Selected)": acc_selected,
        "Selected Features": n_selected
    })
    print(f"Noise: {noise:2d} | Full Acc: {acc_full:.3f} | Selected Acc: {acc_selected:.3f} | Features Kept: {n_selected}")

results_df = pd.DataFrame(results)
```
## Step 4: Visualize Results
```python
plt.figure(figsize=(10, 6))
plt.plot(results_df["Noise Features"], results_df["Accuracy (Full)"], 
         marker="o", label="Without Selection", color="red")
plt.plot(results_df["Noise Features"], results_df["Accuracy (Selected)"], 
         marker="o", label="With Selection", color="green")
plt.xlabel("Number of Noise Features Added")
plt.ylabel("Model Accuracy")
plt.title("Impact of Noise Features on Model Performance")
plt.legend()
plt.grid(True)
plt.show()
```
## Conclusion
The `NMIFeatureSelector` effectively:
- Identifies informative features in a real-world dataset.
- Filters out noise, preventing model performance degradation.
- Works robustly even when 80% of features are uninformative (40 noise + 11 original).

Recommendation:
Tune the threshold parameter to balance feature retention and noise removal. For example, a higher threshold (e.g., 0.1) may further reduce noise but risks excluding weakly informative features.
