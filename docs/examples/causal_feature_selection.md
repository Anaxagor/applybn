##  Robustness Analysis: Causal Feature Selection with Added Uninformative Features
This example demonstrates the `CausalFeatureSelector`'s ability to maintain model performance even when irrelevant second-order features are added, using the Wine dataset.
## Step 1: Load and Preprocess Data
```python
data = pd.read_csv('data/feature_selection/WineQT.csv')
data = data.drop(columns=['Id'])
X = data.drop("quality", axis=1)
y = data["quality"].values
feature_names = list(data.columns)[:-1]
```
## Step 2: Add polynomial features (second order) and shuffle to create uninformative features
```python
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
np.random.seed(42)
X_poly = np.random.permutation(X_poly)  # Destroy relationship with target

# Combine original and synthetic features
X_combined = np.hstack([X, X_poly])
print(f"Total features: {X_combined.shape[1]} (30 original + {X_poly.shape[1]} synthetic)")
```
## Step 3: Initialize containers for results
```python
n_uninformative_steps = np.arange(0, 200, 20)
accuracies_full = []
accuracies_selected = []

# Initialize models
selector = CausalFeatureSelector(n_bins=5)
model = LogisticRegression(max_iter=1000, random_state=42)
```
## Step 4: Progressive Feature Addition
```python
for n in n_uninformative_steps:
    # Select subset of uninformative features
    X_current = np.hstack([X, X_poly[:, :n]])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_current, y, test_size=0.3, random_state=42
    )
    
    # With Feature Selection
    selector.fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    model.fit(X_train_sel, y_train)
    acc_selected = accuracy_score(y_test, model.predict(X_test_sel))
    
    # Without Feature Selection
    model.fit(X_train, y_train)
    acc_full = accuracy_score(y_test, model.predict(X_test))
    
    # Store results
    accuracies_full.append(acc_full)
    accuracies_selected.append(acc_selected)
    
    print(f"Added {n} noise features | "
          f"Selected: {X_train_sel.shape[1]} features | "
          f"Acc: {acc_selected:.3f} (selected) vs {acc_full:.3f} (full)")
```
## Step 5: Visualization and Analysis of Final Iteration
```python
plt.figure(figsize=(10, 6))
plt.plot(n_uninformative_steps, accuracies_full, 'r--', label='Full Feature Set')
plt.plot(n_uninformative_steps, accuracies_selected, 'g-', label='Selected Features')
plt.xlabel('Number of Added Uninformative Features')
plt.ylabel('Test Accuracy')
plt.title('Model Robustness to Irrelevant Features')
plt.legend()
plt.grid(True)
plt.show()

true_features = set(range(30))
selected_features = set(np.where(selector.support_)[0])

print(f"\nFinal Selection Analysis:")
print(f"- Total selected: {len(selected_features)} features")
print(f"- True features retained: {len(true_features & selected_features)}/{30}")
print(f"- Noise features selected: {len(selected_features - true_features)}")
```
## Conclusion
This example demonstrates the method's robustness in real-world scenarios where datasets often contain many irrelevant or engineered features. The causal selection mechanism effectively identifies stable predictive relationships despite increasing noise pollution.