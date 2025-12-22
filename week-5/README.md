Below is your 7‑day plan for **Week 5**, focused on:

- Deepening your understanding of **overfitting vs underfitting**
- Introducing **regularization** (Ridge/Lasso/Logistic with C)
- Practicing **cross‑validation** and **basic model comparison**

You’ll mostly reuse the **California Housing** (regression) and **Titanic** (classification) setups.

Assume ~1.5–2.5 hours/day. As always: try to reason first, then search.

---

## Overall Week 5 Goals

By the end of this week, you should be able to:

- Explain and recognize **overfitting** and **underfitting**.
- Use **L2 (Ridge)** and **L1 (Lasso)** regularization for regression.
- Control regularization strength in **logistic regression** (via `C`).
- Use **cross‑validation** (CV) for more reliable model evaluation.
- Compare models using CV and choose reasonable hyperparameters.

---

## Day 1 – Deep Dive: Bias–Variance, Underfitting & Overfitting (Conceptual + Simple Demo)

**Objectives:**
- Build a strong conceptual understanding of under/overfitting.
- See it visually with a simple synthetic example.

### 1. Notebook

Create: `week5_day1_bias_variance_overfitting.ipynb`.

### 2. Conceptual Notes (Markdown)

In your own words, write short sections for:

- **Bias**: how far the model’s average prediction is from the true function; relates to model *simplicity* / assumptions.
- **Variance**: how much predictions change with different training samples; relates to model *complexity* / sensitivity.
- **Underfitting**: high bias, low variance; model too simple.
- **Overfitting**: low bias on training data but high variance; memorizes noise.

Try to draw a very simple ASCII diagram (or just describe):

- Underfit: straight line missing curved pattern.
- Just right: line/curve close to true pattern.
- Overfit: crazy squiggle through all data points.

### 3. Simple Synthetic Example (1D)

You’ll create noisy data from a curve and fit polynomial models of increasing degree.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X).ravel()
noise = np.random.normal(scale=0.3, size=y_true.shape)
y = y_true + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

Fit polynomials of degrees 1, 3, 10:

```python
degrees = [1, 3, 10]

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"Degree {d}: Train MSE={train_mse:.3f}, Test MSE={test_mse:.3f}")
```

Plot for each degree:

```python
X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
X_plot_poly = poly.fit_transform(X_plot)  # use same 'poly' as for that degree
# (You'll need to do this inside loop for each degree)
```

Plot scatter of training data and the fitted curve.

### 4. Interpret

In Markdown:

- For each degree:
  - How do train and test errors compare?
  - Which degree is clearly underfitting?
  - Which appears to be overfitting?
- Relate this to bias vs variance.

### 5. Thinking Challenge

Imagine you’re given only the **training error** for several models, all very low.  

- Why is that not enough to judge which model is best?
- What additional information do you need (and why)?

Write 6–8 sentences.

---

## Day 2 – Regularization for Regression: Ridge (L2) & Lasso (L1)

**Objectives:**
- Understand how regularization combats overfitting.
- Use **Ridge** and **Lasso** regression on a real dataset.

### 1. Notebook

Create: `week5_day2_ridge_lasso_regression.ipynb`.

### 2. Conceptual Notes (Markdown)

Explain briefly:

- Regularization adds a **penalty** to large coefficients.
- **Ridge (L2)**: penalty on sum of squared coefficients; tends to shrink but rarely zero them.
- **Lasso (L1)**: penalty on sum of absolute coefficients; can drive some coefficients exactly to zero (feature selection effect).
- Trade‑off controlled by **alpha** (in scikit‑learn):
  - higher alpha → stronger regularization → more bias, less variance.

### 3. Use California Housing Dataset Again

Reuse Week 3 setup:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Scale features:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4. Compare Linear, Ridge, Lasso Across Alphas

Try a range of alphas:

```python
alphas = [0.01, 0.1, 1, 10, 100]
results = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_train_pred_r = ridge.predict(X_train_scaled)
    y_test_pred_r = ridge.predict(X_test_scaled)
    
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_train_pred_l = lasso.predict(X_train_scaled)
    y_test_pred_l = lasso.predict(X_test_scaled)
    
    results.append({
        "alpha": alpha,
        "model": "ridge",
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred_r)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred_r)),
    })
    
    results.append({
        "alpha": alpha,
        "model": "lasso",
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred_l)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred_l)),
    })

pd.DataFrame(results)
```

Also compute baseline unregularized linear regression for reference.

### 5. Inspect Coefficients (Especially for Lasso)

For one or two alphas, look at number of non‑zero coefficients:

```python
lasso = Lasso(alpha=1.0, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
coef_series = pd.Series(lasso.coef_, index=X.columns)
print("Non-zero coefficients:", (coef_series != 0).sum())
print(coef_series.sort_values())
```

### 6. Thinking Challenge

In Markdown:

- How does increasing alpha change train and test RMSE?
- Did you see any case where:
  - train error increased slightly,
  - but test error improved (less overfitting)?  
- Why might Lasso zero out some coefficients? When could that be useful in practice?

---

## Day 3 – Regularization in Logistic Regression (Classification)

**Objectives:**
- See how regularization also applies to logistic regression.
- Explore the effect of `C` (inverse regularization strength).

### 1. Notebook

Create: `week5_day3_logistic_regularization.ipynb`.

### 2. Conceptual Notes (Markdown)

- In scikit‑learn’s `LogisticRegression`:
  - `C` = inverse of regularization strength.
  - Small `C` → *strong* regularization.
  - Large `C` → *weak* regularization (approaches unregularized logistic).

### 3. Use Preprocessed Titanic Features

Reuse your cleaned, encoded Titanic dataset from Week 4 (or re‑implement quickly):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/train.csv")
# ... handle missing values as you did in Week 4 ...
# ... create features list ...
X = df[features].copy()
y = df["Survived"]
X_encoded = pd.get_dummies(X, columns=["Sex", "Embarked", "Pclass"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
```

### 4. Train Logistic Regression for Different C Values

```python
Cs = [0.01, 0.1, 1, 10, 100]
results = []

for C in Cs:
    log_reg = LogisticRegression(max_iter=1000, C=C)
    log_reg.fit(X_train, y_train)
    y_pred_train = log_reg.predict(X_train)
    y_pred_test = log_reg.predict(X_test)
    
    results.append({
        "C": C,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_f1": f1_score(y_train, y_pred_train),
        "test_f1": f1_score(y_test, y_pred_test),
    })

pd.DataFrame(results)
```

### 5. Plot or Inspect Results

- Identify:
  - Are there C values where training accuracy is very high but test accuracy doesn’t improve or gets worse?
  - Does too small C (very strong regularization) underfit (both train & test low)?

### 6. Thinking Challenge

In Markdown:

- Pick one or two `C` values and inspect coefficients:

  ```python
  log_reg = LogisticRegression(max_iter=1000, C=0.1)
  log_reg.fit(X_train, y_train)
  coeffs = pd.Series(log_reg.coef_[0], index=X_train.columns)
  print(coeffs.sort_values(key=abs, ascending=False).head(10))
  ```

- Compare for small vs large C:
  - Are coefficients generally smaller for smaller C?
  - How might this help control overfitting?

Explain in 6–8 sentences.

---

## Day 4 – Cross‑Validation Basics (K‑Fold, `cross_val_score`)

**Objectives:**
- Learn what **cross‑validation (CV)** is and why it’s used.
- Apply K‑fold CV to regression and classification tasks.

### 1. Notebook

Create: `week5_day4_cross_validation_basics.ipynb`.

### 2. Conceptual Notes (Markdown)

Explain:

- **K‑Fold CV**:
  - Split data into K folds.
  - Train on K-1 folds, validate on the remaining fold.
  - Repeat K times, average the scores.
- Why CV?
  - Reduces dependence on a single train/test split.
  - Gives a more stable estimate of performance.
- When helpful:
  - Model selection, hyperparameter tuning.

### 3. Cross‑Validation on California Housing (Regression)

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Create pipeline: scaling + Ridge
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

# Use negative MSE because scikit-learn's cross_val_score expects a score to maximize
neg_mse_scores = cross_val_score(
    ridge_pipeline, X, y,
    cv=5,
    scoring="neg_mean_squared_error"
)

rmse_scores = np.sqrt(-neg_mse_scores)
print("CV RMSE per fold:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean(), "Std:", rmse_scores.std())
```

### 4. Cross‑Validation on Titanic (Classification)

Use a pipeline for logistic regression:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Suppose you have:
# numeric_features = ["Age", "Fare", "SibSp", "Parch"]
# categorical_features = ["Sex", "Pclass", "Embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

clf = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000))
])

from sklearn.metrics import make_scorer, f1_score

f1_scorer = make_scorer(f1_score)

f1_scores = cross_val_score(clf, df[numeric_features + categorical_features], y,
                            cv=5, scoring=f1_scorer)
print("CV F1 per fold:", f1_scores)
print("Mean F1:", f1_scores.mean(), "Std:", f1_scores.std())
```

(If you haven’t used `ColumnTransformer` before, this is a good gentle introduction.)

### 5. Thinking Challenge

In Markdown:

- Compare a single train/test split result (from earlier weeks) with CV mean and std.
- Why might a model that looks slightly better on a single split not be **truly** better when you consider CV?
- When is it especially important to use CV (e.g., small datasets)?

---

## Day 5 – Simple Hyperparameter Tuning with Cross‑Validation

**Objectives:**
- Use cross‑validation to choose hyperparameters (`alpha`, `C`).
- Practice systematic comparison instead of ad‑hoc guessing.

### 1. Notebook

Create: `week5_day5_hyperparameter_tuning.ipynb`.

### 2. Ridge Alpha Tuning (Regression)

Using California Housing again:

```python
from sklearn.model_selection import GridSearchCV

alphas = [0.01, 0.1, 1, 10, 100]

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

param_grid = {
    "ridge__alpha": alphas
}

grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X, y)

print("Best alpha:", grid_search.best_params_)
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))
```

Inspect:

```python
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[["param_ridge__alpha", "mean_test_score", "std_test_score"]]
```

### 3. Logistic Regression C Tuning (Classification)

Using Titanic:

```python
Cs = [0.01, 0.1, 1, 10, 100]

clf = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000))
])

param_grid = {
    "logreg__C": Cs
}

grid_search_class = GridSearchCV(
    clf,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search_class.fit(df[numeric_features + categorical_features], y)

print("Best C:", grid_search_class.best_params_)
print("Best CV F1:", grid_search_class.best_score_)
```

Inspect `grid_search_class.cv_results_`.

### 4. Thinking Challenge

In Markdown:

- Why do we tune hyperparameters on CV **within** the training data (or full dataset with nested CV) rather than on the **test set**?
- What would go wrong if we kept tweaking hyperparameters until test performance looked best?

---

## Day 6 – Applying Regularization & CV to Your Previous Mini‑Projects

**Objectives:**
- Revisit Week 3 regression mini‑project and Week 4 classification mini‑project.
- Improve them using regularization and CV.
- Practice iteration and comparison.

### 1. Regression Mini‑Project Upgrade

Open `week3_regression_mini_project.ipynb` or create a new “v2” copy.

Add:

- A **Ridge regression** model with alpha tuned via CV (like Day 5).
- Optionally, a **Lasso** model for comparison.
- Evaluate best model on your held‑out test set (kept from Week 3).

Document:

- Baseline vs Linear vs Ridge/Lasso metrics.
- CV results used to select alpha.

### 2. Classification Mini‑Project Upgrade

Open `week4_titanic_classification_project.ipynb` or a copy.

Add:

- A **LogisticRegression + pipeline** with `C` tuned via CV (as in Day 5).
- Optionally, try:
  - Different subsets of features (include/exclude engineered ones like `FamilySize`).
- Evaluate best model on test set:
  - Accuracy, precision, recall, F1, ROC AUC.

### 3. Thinking Challenge

In Markdown for each project:

- Did regularization + CV actually **improve** test performance?
- Even if performance didn’t increase much:
  - Did CV make you more confident in your model’s reliability?
  - Did regularization simplify coefficients or reduce overfitting indications?

Write 8–12 sentences reflecting on this.

---

## Day 7 – Week 5 Mini‑Synthesis: Overfitting, Regularization & CV Report

**Objectives:**
- Consolidate the key ideas from this week.
- Write a concise “theory + practice” summary you can return to later.

### 1. Notebook or Markdown Doc

Create: `week5_summary_overfitting_regularization_cv.ipynb`  
(or a `.md` file if you prefer plain text).

### 2. Structured Written Summary

Create sections in Markdown:

1. **Underfitting & Overfitting**
   - Definitions.
   - How you can detect each (train vs test error).
   - Example from your Week 1 synthetic polynomial experiment.

2. **Regularization (Ridge & Lasso & Logistic)**
   - What problem regularization solves.
   - Brief explanation of L1 vs L2.
   - How `alpha` and `C` affect model behavior.
   - One or two real‑world scenarios where regularization is essential.

3. **Cross‑Validation**
   - What K‑fold CV is.
   - Why it’s more reliable than a single split.
   - When you would definitely want to use it.

4. **Hyperparameter Tuning**
   - Role of GridSearchCV (and mention RandomizedSearchCV as an alternative).
   - Why you must not tune on test set.
   - Short example: tuning Ridge alpha and Logistic C.

5. **Lessons from Your Projects**
   - For your regression project:
     - How did Ridge/Lasso compare to plain Linear?
   - For Titanic:
     - How did tuned logistic regression compare to default?
   - 3–5 bullet points on what *you* found most surprising or insightful.

### 3. Short “Explain It to a Friend” Paragraph

As a final thinking exercise, write a 10–12 sentence explanation for a friend who knows some programming but no ML:

- What is overfitting?
- How does regularization help?
- What is cross‑validation, and why is it important?
- How did you use these ideas in your house price and Titanic projects?

Try to avoid jargon where possible.

---

## After Week 5: What You Should Be Able to Do

You should now:

- Recognize underfitting vs overfitting from train/test (or CV) metrics.
- Use **Ridge/Lasso** and **regularized logistic regression** with different regularization strengths.
- Run **cross‑validation** and basic hyperparameter tuning with `GridSearchCV`.
- Confidently iterate on models in your tabular projects, choosing more robust configurations instead of guessing.

If you’d like, the next logical step is Week 6: **decision trees and random forests**, plus more systematic **feature engineering**—a big practical leap in performance on many real‑world tabular problems.
