Below is your 7‑day plan for **Week 6**, focused on:

- Understanding and using **decision trees** (classification & regression)
- Using **random forests** (powerful, practical default for tabular data)
- Practicing **basic feature engineering** with these models
- Comparing tree‑based models to linear/logistic regression on your previous projects

Assume ~1.5–2.5 hours/day. Continue the habit: think first, then search.

---

## Overall Week 6 Goals

By the end of this week, you should be able to:

- Explain how **decision trees** split data and why they can overfit.
- Train and evaluate **DecisionTreeClassifier/Regressor**.
- Train and evaluate **RandomForestClassifier/Regressor**.
- Read **feature importances** and use them to guide feature engineering.
- Compare tree‑based models vs linear/logistic models on your regression & classification projects.

---

## Day 1 – Intuition & First Decision Tree Classifier (Titanic)

**Objectives:**
- Build intuitive understanding of decision trees.
- Train a first decision tree classifier on Titanic.

### 1. Notebook

Create: `week6_day1_decision_tree_intro.ipynb`.

Load your preprocessed Titanic dataset (from Week 4/5) or quickly re‑create it:
- Cleaned `df`.
- `y = df["Survived"]`.
- `X_encoded` (numeric + one‑hot encoded categorical features).

### 2. Conceptual Notes (Markdown)

In your own words, write short sections:

- What a **decision tree** is:
  - A sequence of “if/else” split rules on features, forming a tree.
- How it works at a high level:
  - At each node, it chooses a feature and a threshold/categorical split that best separates the classes.
- Why trees can **overfit**:
  - They can keep splitting until each leaf has very few samples (even memorize training set).

Also mention:

- For classification trees, impurity measures like:
  - Gini impurity.
  - Entropy.
  - You don’t need formulas yet—just know it’s a measure of “mixedness” of classes.

### 3. Train a Basic Decision Tree on Titanic

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
```

### 4. Inspect Overfitting Quickly

Check performance on training set:

```python
y_train_pred = tree_clf.predict(X_train)
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Train F1:", f1_score(y_train, y_train_pred))
```

If train accuracy is 1.0 and test much lower, that’s a sign of overfitting.

### 5. Thinking Challenge

In Markdown:

1. Compare this tree’s performance with your **logistic regression** from Week 4:
   - Which has higher accuracy? Higher F1?
2. Is the tree overfitting? What evidence do you see?
3. Why might a tree overfit more easily than logistic regression on Titanic?

Write 6–10 sentences.

---

## Day 2 – Controlling Tree Complexity (max_depth, min_samples_leaf)

**Objectives:**
- Learn to control tree size to reduce overfitting.
- See how hyperparameters like `max_depth` and `min_samples_leaf` affect performance.

### 1. Notebook

Create: `week6_day2_tree_hyperparameters.ipynb`.

Reuse Titanic `X_train`, `X_test`, `y_train`, `y_test`.

### 2. Experiment with max_depth

Try a range of depths:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

depths = [1, 2, 3, 4, 5, 8, 12, None]  # None = full depth
results = []

for d in depths:
    tree_clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree_clf.fit(X_train, y_train)
    
    y_train_pred = tree_clf.predict(X_train)
    y_test_pred = tree_clf.predict(X_test)
    
    results.append({
        "max_depth": d,
        "train_acc": accuracy_score(y_train, y_train_pred),
        "test_acc": accuracy_score(y_test, y_test_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "test_f1": f1_score(y_test, y_test_pred),
    })

import pandas as pd
pd.DataFrame(results)
```

Interpret:
- Which depth seems to give best **test** performance?
- How do train metrics vs depth behave?

### 3. Experiment with min_samples_leaf

Keep a moderately deep tree (e.g., `max_depth=5` or `None`) and vary `min_samples_leaf`:

```python
leaf_sizes = [1, 2, 5, 10, 20]
results_leaf = []

for leaf in leaf_sizes:
    tree_clf = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=leaf,
        random_state=42
    )
    tree_clf.fit(X_train, y_train)
    y_train_pred = tree_clf.predict(X_train)
    y_test_pred = tree_clf.predict(X_test)
    
    results_leaf.append({
        "min_samples_leaf": leaf,
        "train_acc": accuracy_score(y_train, y_train_pred),
        "test_acc": accuracy_score(y_test, y_test_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "test_f1": f1_score(y_test, y_test_pred),
    })

pd.DataFrame(results_leaf)
```

### 4. Thinking Challenge

In Markdown:

- For both `max_depth` and `min_samples_leaf` experiments:
  - Identify one configuration that is clearly **underfitting**.
  - Identify one that is clearly **overfitting**.
  - Identify one that is a reasonable compromise.

Explain how you know, using the train vs test metrics.

---

## Day 3 – Visualizing and Interpreting a Small Tree

**Objectives:**
- Visualize a shallow decision tree.
- Learn to read decision rules and interpret splits.

### 1. Notebook

Create: `week6_day3_visualize_tree.ipynb`.

### 2. Train a Very Small Tree

To make plotting manageable, restrict depth:

```python
from sklearn.tree import DecisionTreeClassifier

small_tree = DecisionTreeClassifier(
    max_depth=3,  # small
    random_state=42
)
small_tree.fit(X_train, y_train)
```

### 3. Visualize the Tree

Option A (textual rules):

```python
from sklearn.tree import export_text

tree_rules = export_text(small_tree, feature_names=list(X_train.columns))
print(tree_rules)
```

Option B (graphical, if you install graphviz; optional):

Search: `sklearn plot_tree`:

```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 10))
tree.plot_tree(
    small_tree,
    feature_names=X_train.columns,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    proportion=True
)
plt.show()
```

### 4. Interpret Splits

Pick a path from root to a leaf and write in **plain language** what it means:

Example:

- Root: `Sex_male <= 0.5`  
  → “Is passenger female?”
- Next node: maybe `Pclass_3 <= 0.5`  
  → “Is passenger not in 3rd class?” etc.

For at least 2–3 leaves:

- Describe:
  - Conditions (feature thresholds).
  - Class prediction at leaf.
  - Fraction of survivors vs non‑survivors.

### 5. Thinking Challenge

In Markdown:

1. Does the tree’s top split match your intuition about what matters most for survival?
2. Do you see any rules that look **overly specific** or maybe spurious?
3. Compare the interpretability of:
   - This small tree.
   - Your logistic regression coefficients from Week 4/5.  
   Which is easier to explain to a non‑technical person?

---

## Day 4 – Random Forests: Intuition & First Models

**Objectives:**
- Understand why **random forests** improve on single trees.
- Train random forest models for both regression and classification.

### 1. Notebook

Create: `week6_day4_random_forest_intro.ipynb`.

### 2. Conceptual Notes (Markdown)

Explain in your own words:

- Random forest = **ensemble** of many decision trees.
- Key ideas:
  - Each tree sees a **bootstrap sample** (random subset with replacement) of the training data.
  - At each split, a random subset of features is considered.
- Benefits:
  - Reduces variance / overfitting of a single deep tree.
  - Often strong performance “out of the box.”

### 3. Random Forest Classifier (Titanic)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

rf_clf = RandomForestClassifier(
    n_estimators=100,  # number of trees
    max_depth=None,    # or limit if you like
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
print("RF F1:", f1_score(y_test, y_pred_rf))
```

Also check training performance:

```python
y_train_pred_rf = rf_clf.predict(X_train)
print("RF Train Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("RF Train F1:", f1_score(y_train, y_train_pred_rf))
```

### 4. Random Forest Regressor (California Housing)

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame
X_reg = df.drop(columns=["MedHouseVal"])
y_reg = df["MedHouseVal"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(Xr_train, yr_train)
yr_pred_rf = rf_reg.predict(Xr_test)

rmse_rf = np.sqrt(mean_squared_error(yr_test, yr_pred_rf))
print("Random Forest RMSE:", rmse_rf)
```

Compare to your linear/Ridge models from Week 3/5.

### 5. Thinking Challenge

In Markdown:

- How do random forest metrics compare to:
  - Decision tree (same data)?
  - Logistic/linear models?
- Why might random forests often outperform single trees and linear models on tabular data?
- What are potential downsides of random forests (e.g., interpretability, speed)?

Write 8–12 sentences.

---

## Day 5 – Feature Importance & Basic Feature Engineering with Trees

**Objectives:**
- Use tree‑based **feature importances** to understand what matters.
- Try simple feature engineering and see its effect.

### 1. Notebook

Create: `week6_day5_feature_importance_engineering.ipynb`.

### 2. Feature Importance (Titanic)

Assume you have `rf_clf` already trained.

```python
import pandas as pd
import numpy as np

importances = rf_clf.feature_importances_
feature_names = X_train.columns

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp
```

Plot:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
feat_imp.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top Feature Importances (Random Forest - Titanic)")
plt.show()
```

Interpret:

- Which features are most important?
- Do they match your prior intuition (e.g., Sex, Pclass, Age)?

### 3. Basic Feature Engineering

Try at least one of:

- **FamilySize**:
  ```python
  df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
  ```
- **IsAlone**:
  ```python
  df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
  ```
- **Age bins** (optional, trees often handle continuous features fine):
  ```python
  df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 80], labels=False)
  ```

Recreate `X_new` including these features, re‑encode (using `get_dummies` where needed), and retrain `rf_clf`. Compare:

- Accuracy/F1 before and after.
- Feature importances now—does `FamilySize` or `IsAlone` appear near the top?

### 4. Feature Importance (California Housing)

Do similarly with `rf_reg`:

```python
imp_reg = pd.Series(rf_reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
imp_reg
```

Interpret:

- Which features most impact predicted house value (e.g., median income, latitude/longitude)?

### 5. Thinking Challenge

In Markdown:

1. Based on feature importance, suggest **at least 3 potential new features** that could be useful (even if you don’t implement all now). For example, interactions or ratios.
2. For each suggested feature, explain:
   - How you would compute it.
   - Why it might help the model.

---

## Day 6 – Comparing Models Systematically (Trees vs Linear/Logistic vs RF)

**Objectives:**
- Systematically compare models on the same data.
- Practice model selection based on metrics and practical considerations.

### 1. Notebook

Create: `week6_day6_model_comparison.ipynb`.

### 2. Classification Comparison (Titanic)

For Titanic, compare:

- Logistic Regression (best version from Week 5, maybe tuned with CV).
- Decision Tree (with a reasonable max_depth).
- Random Forest (default or lightly tuned).

Standardize your evaluation:

- Use the same train/test split.
- Compute:
  - Accuracy
  - Precision, recall, F1
  - ROC AUC

Organize results in a DataFrame:

```python
results = []

def evaluate_classifier(name, model, X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    })

# Call evaluate_classifier for each model, then:
pd.DataFrame(results)
```

### 3. Regression Comparison (California Housing)

Similarly, compare:

- Linear Regression (or Ridge/Lasso best from Week 5).
- DecisionTreeRegressor.
- RandomForestRegressor.

Metrics:

- RMSE on test.
- Maybe R².

Organize results in a DataFrame for easy comparison.

### 4. Thinking Challenge

In Markdown:

- For each dataset (Titanic, Housing):
  - Which model performs best on your chosen metric(s)?
  - Are there big gaps between train and test metrics for any model?
  - How do you weigh:
    - Performance
    - Interpretability
    - Training/inference time (roughly)
- If you had to choose a **single model type** for a real tabular business problem tomorrow, which would you pick and why?

Write ~10–15 sentences reflecting on this.

---

## Day 7 – Week 6 Mini‑Project: Tree & Forest Exploration

**Objectives:**
- Cement your understanding by doing a focused mini‑project.
- Combine decision trees, random forests, feature engineering, and evaluation.

### 1. Notebook

Create: `week6_tree_forest_mini_project.ipynb`.

You can choose:

- **Option A (Recommended):** Titanic classification, but now focused on trees/forests.
- **Option B:** California Housing regression, focused on trees/forests.

Below assumes Titanic; adapt similarly for regression if you prefer.

### 2. Project Structure (Markdown + Code)

Use headings:

1. **Introduction**
   - Problem: predict survival using tree‑based models.
   - Goal: compare decision tree vs random forest and understand key features.

2. **Data & Preprocessing**
   - Load dataset.
   - Handle missing values.
   - Create engineered features (at least FamilySize or IsAlone).
   - Encode categoricals.
   - Train/test split.

3. **Models**
   - Baseline:
     - Simple decision tree (with sensible max_depth).
   - Improved:
     - Decision tree with tuned `max_depth` and `min_samples_leaf` (using quick grid or manual search).
     - Random forest (tune at least `n_estimators` and `max_depth` a little).
   - For at least one model, optionally do simple CV to estimate performance robustness.

4. **Evaluation**
   - For each model, compute:
     - Accuracy, precision, recall, F1, ROC AUC.
   - Compare in a table.
   - Plot ROC curves for 2–3 models on same plot.

5. **Interpretation**
   - Use feature importance from random forest:
     - List top 10 features.
     - Plot them.
   - Interpret a small tree (max_depth=3) as in Day 3.

6. **Conclusions & Next Steps**
   - Summarize which model you’d choose and why.
   - Note any limitations or weird behaviors you saw.
   - Suggest 3–5 future improvements (e.g., more feature engineering, gradient boosting, deeper CV).

### 3. Thinking / Stretch Tasks

Pick at least one:

- **Ablation Study:**  
  Train a random forest with and without your engineered feature(s) (e.g., FamilySize); quantify how much performance changes.
- **Hyperparameter Sensitivity:**  
  Vary one RF parameter (e.g., `max_depth` or `n_estimators`) and see how performance changes.
- **Different Dataset:**  
  If you have time, quickly repeat a lighter version of this mini‑project on a different small tabular dataset (e.g., another Kaggle dataset). This reinforces transfer of skills, not just memorization.

---

## After Week 6: What You Should Be Able to Do

You should now:

- Comfortably use **DecisionTreeClassifier/Regressor** with sensible hyperparameters.
- Build and evaluate **RandomForestClassifier/Regressor** and understand why they work well.
- Use **feature importances** to guide your intuition and feature engineering.
- Compare linear/logistic models vs tree‑based models and make an informed choice for a tabular ML problem.

Next natural step (Weeks 7–8) is to move into **gradient boosting** (XGBoost/LightGBM or sklearn’s GradientBoosting) and **more realistic messy data**, building on the tree‑based foundation you now have.
