Here’s your 7‑day plan for **Week 7**, focused on:

- Learning **gradient boosting** (GradientBoosting, XGBoost/LightGBM if possible)
- Applying it to **regression and classification**
- Dealing with **messier, more realistic data**
- Comparing boosted trees with random forests and linear/logistic models

Assume ~1.5–2.5 hours/day. Continue to reason first, then search.

---

## Overall Week 7 Goals

By the end of this week, you should be able to:

- Explain what **gradient boosting** is in simple terms.
- Train and evaluate **GradientBoostingRegressor/Classifier** (sklearn).
- Optionally try **XGBoost or LightGBM** on tabular data.
- Handle somewhat messy data: more missing values, more features, less “curated” than Titanic/California Housing.
- Compare gradient boosting vs random forest vs linear/logistic models on a real dataset.

A good idea is to pick a **new Kaggle tabular dataset** this week (classification or regression) to stretch yourself beyond Titanic/California Housing. I’ll assume you do that in a couple of the days.

---

## Day 1 – Gradient Boosting Intuition & Simple Demo (Synthetic or Small Dataset)

**Objectives:**
- Build a mental model of boosting.
- Run a first gradient boosting model on a simple dataset.

### 1. Notebook

Create: `week7_day1_gradient_boosting_intro.ipynb`.

### 2. Conceptual Notes (Markdown)

In your own words, write:

- **Bagging vs Boosting**
  - Bagging (e.g., random forests):
    - Trains many trees **in parallel** on different bootstrapped samples.
    - Averages predictions to reduce variance.
  - Boosting:
    - Trains trees **sequentially**, each new tree focuses on correcting errors of the previous ones.
    - Combines many weak learners into a strong one.

- **Gradient Boosting (high-level)**
  - Starts with a simple prediction (e.g., mean of targets).
  - Computes residuals (errors).
  - Fits a small tree to predict residuals.
  - Adds that tree to the model and repeats, each time trying to reduce remaining error.
  - “Gradient” refers to using gradient descent on the loss function (you don’t need full math now).

Keep it intuitive, not formal.

### 3. Try GradientBoostingRegressor on California Housing

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
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

gbr = GradientBoostingRegressor(
    random_state=42
)
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse
```

Compare with random forest RMSE from Week 6 (rerun if necessary).

### 4. Inspect Training vs Test Performance

```python
y_train_pred = gbr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = rmse
print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)
```

Is there a big gap (overfitting) or are they close?

### 5. Thinking Challenge

In Markdown:

1. Compare gradient boosting’s performance to:
   - Linear regression or Ridge/Lasso.
   - Random forest (if you have previous numbers).
2. Why might gradient boosting match or outperform random forests in many tabular problems?
3. What trade‑offs might there be (e.g., training time, sensitivity to hyperparameters)?

Write 8–12 sentences.

---

## Day 2 – Hyperparameters of Gradient Boosting (Learning Rate, n_estimators, max_depth)

**Objectives:**
- Understand main knobs of gradient boosting.
- See how they affect over/underfitting.

### 1. Notebook

Create: `week7_day2_gb_hyperparameters.ipynb`.

Use California Housing again for clear numerical metrics.

### 2. Key Hyperparameters (Markdown)

In your own words:

- `n_estimators`: number of boosting stages (trees).
  - More trees → more capacity → risk of overfitting if learning rate is too high.
- `learning_rate`: how much each tree contributes.
  - Lower = slower learning, often requires more trees, but can generalize better.
- `max_depth` (or `max_leaf_nodes`): size/complexity of each individual tree.

Explain that there’s a **trade‑off**:
- Many shallow trees with small learning rate vs fewer deeper trees with larger learning rate.

### 3. Simple Grid Exploration

Set up a small experiment:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

configs = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3},
]

results = []

for cfg in configs:
    gbr = GradientBoostingRegressor(
        random_state=42,
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"]
    )
    gbr.fit(X_train, y_train)
    y_train_pred = gbr.predict(X_train)
    y_test_pred = gbr.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    result = cfg.copy()
    result["train_rmse"] = train_rmse
    result["test_rmse"] = test_rmse
    results.append(result)

pd.DataFrame(results)
```

### 4. Interpret

- Which config overfits? (Very low train error, worse test error.)
- Which underfits? (Both train and test errors high.)
- Which is a good balance?

### 5. Thinking Challenge

Imagine you have **very limited data** (small training set). In Markdown:

- How might you choose `n_estimators` and `learning_rate` to reduce overfitting?
- Which would you prioritize: a simpler tree (`max_depth` small) or fewer trees?

Write 6–8 sentences.

---

## Day 3 – GradientBoostingClassifier on Titanic (or Similar Classification Task)

**Objectives:**
- Apply gradient boosting to a classification problem.
- Compare with logistic regression and random forest.

### 1. Notebook

Create: `week7_day3_gb_classifier_titanic.ipynb`.

Reuse your cleaned Titanic dataset with:
- `X` (numeric + encoded categorical OR use a pipeline).
- `y = df["Survived"]`.

Train/test split as before:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 2. GradientBoostingClassifier

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

gb_clf = GradientBoostingClassifier(
    random_state=42
)
gb_clf.fit(X_train, y_train)

y_pred = gb_clf.predict(X_test)
y_proba = gb_clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
```

Compare with random forest and logistic regression metrics from earlier weeks.

### 3. Train vs Test Check

```python
y_train_pred = gb_clf.predict(X_train)
y_train_proba = gb_clf.predict_proba(X_train)[:, 1]

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Train F1:", f1_score(y_train, y_train_pred))
print("Train ROC AUC:", roc_auc_score(y_train, y_train_proba))
```

Is it overfitting significantly?

### 4. Feature Importances

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = gb_clf.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
feat_imp.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Feature Importances (GradientBoostingClassifier)")
plt.show()
```

Which features are top? Do they align with your expectations?

### 5. Thinking Challenge

In Markdown:

- Compare GB classifier vs:
  - Logistic regression.
  - Random forest.
- Is there a clear winner on:
  - F1.
  - ROC AUC.
- In what kind of classification problems would you be especially excited to try gradient boosting?

Write 8–12 sentences.

---

## Day 4 – New, Messier Dataset: Selection & Initial EDA

**Objectives:**
- Pick a new real‑world tabular dataset (preferably Kaggle).
- Do minimal but focused EDA to understand its structure.

### 1. Choose a Dataset

Go to Kaggle and pick a tabular competition or dataset that is:

- Not Titanic.
- Not image/NLP yet.
- With a **clear supervised learning target** (classification or regression).

Examples (if available when you look):
- Classification:
  - “Spaceship Titanic”
  - “Home Credit Default Risk” (more advanced, but interesting)
  - “Bank Marketing” datasets.
- Regression:
  - Another house price dataset.
  - Any numeric target competition.

Download the data, e.g., `train.csv`, into `ml-learning/week7/data/`.

### 2. Notebook

Create: `week7_day4_new_dataset_eda.ipynb`.

### 3. Load and Inspect

```python
import pandas as pd

df = pd.read_csv("data/train.csv")  # adjust name/path
df.head()
df.info()
df.describe()
df.isna().sum()
```

In Markdown:

- Identify the **target column** (name and type).
- Identify feature types:
  - Numerical columns.
  - Categorical/boolean columns.
- Note any obvious issues:
  - Lots of missing values in certain columns?
  - Weird data types (e.g., numeric IDs you may want to drop)?

### 4. Minimal EDA

Do a **lightweight but targeted EDA**:

- For numeric columns:
  - `df[numeric_cols].describe()`
  - 1–2 histograms for important continuous features.
- For categorical:
  - `value_counts()` for key categorical features.
- For target:
  - If classification: class balance (value_counts with normalize=True).
  - If regression: distribution (histogram, range, mean, std).

### 5. Thinking Challenge

In Markdown:

- Based on this first look:
  - What do you think could be your **top 3 most useful features**?
  - What do you think could be **potential problems** (e.g., high cardinality categoricals, strong skew, heavy missingness)?
- If you had to start with **only 5 features**, which would you pick and why?

Write 8–12 sentences.

---

## Day 5 – Preprocessing & Baseline Models on the New Dataset

**Objectives:**
- Build a baseline pipeline with simple preprocessing.
- Train at least:
  - Linear/logistic baseline.
  - Random forest baseline.
- Prepare to compare with gradient boosting later.

### 1. Notebook

Create: `week7_day5_new_dataset_baselines.ipynb`.

Use the same dataset as Day 4.

### 2. Define Features and Target

In code:

- Choose:
  - `target_col = "..."`  
  - `feature_cols = [...]` (exclude clear IDs; keep manageable subset).
- Split into `X = df[feature_cols]`, `y = df[target_col]`.

### 3. Preprocessing Strategy

Using `ColumnTransformer` and `Pipeline` (like you started Week 5/6):

- Identify:
  - `numeric_features`
  - `categorical_features`

Example:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
```

### 4. Choose Model Type (Classification vs Regression)

If **classification**:

- Start with **LogisticRegression** and **RandomForestClassifier**.

If **regression**:

- Start with **Ridge** (or LinearRegression) and **RandomForestRegressor**.

Example (classification):

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if classification else None
)

logit_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
```

Train & evaluate both, storing metrics (accuracy, F1 for classification; RMSE/R² for regression).

### 5. Thinking Challenge

In Markdown:

- Between your two baselines:
  - Which performs better on your chosen metric?
  - Does either obviously overfit (huge train vs test gap)?
- What is a **reasonable performance target** you’d like to hit with boosting? (e.g., +2–5% improvement over baseline.)

Write 6–10 sentences.

---

## Day 6 – Gradient Boosting on the New Dataset + Basic Tuning

**Objectives:**
- Add gradient boosting to your new dataset.
- Do quick hyperparameter exploration (not full-blown tuning yet).
- Compare to baselines.

### 1. Notebook

Create: `week7_day6_new_dataset_gradient_boosting.ipynb`.

Reuse:

- `X_train`, `X_test`, `y_train`, `y_test`.
- `preprocessor`.

### 2. Gradient Boosting Pipeline

For **classification**:

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", GradientBoostingClassifier(random_state=42))
])
```

For **regression**:

```python
from sklearn.ensemble import GradientBoostingRegressor

gb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("reg", GradientBoostingRegressor(random_state=42))
])
```

Fit and evaluate (accuracy/F1 or RMSE/R²) as before.

### 3. Quick Hyperparameter Tweaks

Try 3–5 configurations, varying:

- `n_estimators` (e.g., 100, 200).
- `learning_rate` (e.g., 0.1, 0.05).
- `max_depth` (e.g., 2, 3).

Example (classification):

```python
configs = [
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
]

results = []

for cfg in configs:
    gb = GradientBoostingClassifier(
        random_state=42,
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"]
    )
    gb_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", gb)
    ])
    gb_pipeline.fit(X_train, y_train)
    
    y_pred = gb_pipeline.predict(X_test)
    # Replace with appropriate metric(s)
    acc = accuracy_score(y_test, y_pred)
    
    row = cfg.copy()
    row["test_accuracy"] = acc
    results.append(row)

pd.DataFrame(results)
```

Pick the config with best test metric.

### 4. Thinking Challenge

In Markdown:

- How does your best gradient boosting configuration compare to:
  - Logistic/linear baseline?
  - Random forest baseline?
- Did tuning `n_estimators`, `learning_rate`, `max_depth` give a **meaningful improvement** or just a small tweak?
- If you had more time, how would you tune these hyperparameters more systematically (e.g., GridSearchCV or RandomizedSearchCV)?

Write 8–12 sentences.

---

## Day 7 – Week 7 Mini‑Project: Boosting vs Forests vs Linear/Logistic (New Dataset)

**Objectives:**
- Consolidate Week 7 skills in a focused project.
- Produce a small comparative study on your new dataset.

### 1. Notebook

Create: `week7_boosting_vs_forest_project.ipynb`.

### 2. Project Structure (Markdown + Code)

Sections:

1. **Introduction**
   - Brief description of the dataset and task (classification or regression).
   - Goal: compare **three model families**:
     - Linear/logistic.
     - Random forest.
     - Gradient boosting.

2. **Data & Preprocessing**
   - Load data.
   - Short EDA recap (just the essentials).
   - Define `X`, `y`.
   - Define `numeric_features`, `categorical_features`.
   - Create `preprocessor`.

3. **Models**
   - Baseline: LinearRegression/Ridge or LogisticRegression pipeline.
   - RandomForest pipeline.
   - GradientBoosting pipeline (with your best set of hyperparameters from Day 6).

4. **Evaluation**
   - Use **the same train/test split** for all models.
   - Compute performance metrics:
     - Classification: Accuracy, Precision, Recall, F1, ROC AUC.
     - Regression: RMSE, MAE, R².
   - Present results in a small table.
   - Optionally:
     - For classification, plot ROC curves for Random Forest vs Gradient Boosting.
     - For regression, compare predicted vs true scatter plots.

5. **Interpretation & Feature Importances**
   - For Random Forest and Gradient Boosting, extract and plot top 10–15 feature importances.
   - Compare:
     - Which features are important across both models?
     - Any features important for one but not the other?
   - Briefly connect feature importance to your data understanding.

6. **Conclusions**
   - Which model performed best and by how much?
   - If performance differences are small, discuss:
     - Stability.
     - Interpretability.
     - Computational cost.
   - Which model you would deploy in a practical setting and why.

### 3. Thinking / Stretch Tasks

Choose at least one:

- **Error Analysis**:  
  For classification: inspect a subset of misclassified examples from the best model.  
  For regression: analyze where errors are large (e.g., high target values).
- **Threshold Tuning (Classification)**:  
  For best GB model, adjust decision threshold to trade off precision vs recall, as you did in Week 5.
- **Compare with Simple CV**:  
  Do a quick 3–5 fold CV for RF and GB, to see if one is more stable (lower std of scores).

### 4. Non‑Technical Summary

Write a 8–12 sentence **non‑technical summary** for a hypothetical stakeholder:

- What dataset and problem you worked on.
- What models you tried.
- How well they performed in intuitive terms.
- Which you would choose and why.
- Key limitations and how you might improve further.

---

## After Week 7: What You Should Be Able to Do

You should now:

- Understand **bagging vs boosting** and why gradient boosting is powerful for tabular data.
- Train **GradientBoostingRegressor/Classifier** and interpret basic hyperparameters.
- Work more confidently with **messy, real‑world datasets**, including:
  - Selecting features.
  - Building preprocessing+model pipelines.
- Compare and choose between:
  - Linear/logistic models.
  - Random forests.
  - Gradient boosting.

Next logical steps:

- Week 8: **Unsupervised learning** (PCA, clustering) and/or
- Deepening your experience in Kaggle‑style competitions using these models, and later:
- Intro to **XGBoost/LightGBM/CatBoost** and more advanced feature engineering.
