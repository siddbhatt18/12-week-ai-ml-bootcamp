Here is your 7‑day plan for **Week 10**, focused on:

- Building **clean, reusable ML pipelines** (especially with scikit‑learn).
- Doing **systematic hyperparameter tuning**.
- Comparing different model families in a structured way.
- Making your work more **reproducible** and **project‑ready**.

Assume ~1.5–2.5 hours/day.

You’ll mainly reuse:
- A **tabular classification dataset** you know well (Titanic or your Week 7 dataset).
- A **tabular regression dataset** (California Housing or your Week 3 project).

---

## Overall Week 10 Goals

By the end of this week, you should be able to:

- Use **Pipeline** and **ColumnTransformer** to combine preprocessing + model cleanly.
- Avoid **data leakage** by doing all transformations inside the pipeline.
- Use **GridSearchCV** / **RandomizedSearchCV** for hyperparameter tuning.
- Compare several models on the same dataset in a reusable way.
- Structure an end‑to‑end notebook that looks close to a “real” ML project.

---

## Day 1 – Pipelines & ColumnTransformer (Clean Preprocessing)

**Objectives:**
- Turn your messy preprocessing+model code into a neat, single pipeline.
- Practice on one familiar classification dataset (e.g., Titanic or Week 7 dataset).

### 1. Notebook

Create: `week10_day1_pipelines_intro.ipynb`.

### 2. Choose Dataset

Pick your main **tabular classification** dataset:
- Titanic, or
- Your Week 7 Kaggle dataset.

You need:
- Raw `df` (no pre‑encoding).
- Clear `target_col` (e.g., "Survived" or similar).

### 3. Identify Feature Types

In code and Markdown:

- `numeric_features`: list numeric columns.
- `categorical_features`: list categorical columns (object/string, or known categories).

Example:

```python
numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
categorical_features = ["Sex", "Pclass", "Embarked"]
```

### 4. Build Preprocessor with ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
```

In Markdown:
- Explain what each transformer does.
- Why `handle_unknown="ignore"` matters (in case test has unseen categories).

### 5. Build a Simple Pipeline with LogisticRegression

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df[numeric_features + categorical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000)

clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", log_reg)
])

clf.fit(X_train, y_train)
```

Evaluate:

```python
from sklearn.metrics import accuracy_score, f1_score

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
```

### 6. Thinking Challenge

In Markdown:

- Why is doing preprocessing (scaling + encoding) **inside** a pipeline safer than doing it manually beforehand?
- How does this help avoid **data leakage**?
- If you deploy this model, what advantage does the pipeline give you?

Write 8–12 sentences.

---

## Day 2 – Pipelines for Regression & Reusing Code Patterns

**Objectives:**
- Build a similar pipeline for a regression dataset (California Housing or your Week 3 project).
- Learn to reuse patterns instead of rewriting everything.

### 1. Notebook

Create: `week10_day2_regression_pipelines.ipynb`.

### 2. Choose Regression Dataset

Use:
- California Housing (from sklearn), or
- Your house price regression dataset from Week 3.

Load into `df_reg`.

Identify:

- `target_reg` (e.g., "MedHouseVal" or "SalePrice").
- `numeric_features_reg` (most/all numeric columns).
- `categorical_features_reg` (if any; otherwise, they can be empty).

### 3. Build Preprocessor & Pipeline

If mostly numeric:

```python
numeric_features_reg = [...]  # list numeric columns
categorical_features_reg = [...]  # if any, else []

numeric_transformer_reg = StandardScaler()
categorical_transformer_reg = OneHotEncoder(handle_unknown="ignore")

preprocessor_reg = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_reg, numeric_features_reg),
        ("cat", categorical_transformer_reg, categorical_features_reg)
    ],
    remainder="drop"
)
```

Create pipeline with Ridge (for example):

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_reg = df_reg[numeric_features_reg + categorical_features_reg]
y_reg = df_reg[target_reg]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor_reg),
    ("model", Ridge(alpha=1.0))
])

reg_pipeline.fit(Xr_train, yr_train)
```

Evaluate:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

yr_pred = reg_pipeline.predict(Xr_test)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
rmse
```

### 4. Thinking Challenge

In Markdown:

- Compare this pipeline‑based solution to your Week 3/5 non‑pipeline code:
  - Which is easier to read and reuse?
  - Which is less error‑prone if the dataset changes slightly (e.g., new columns)?
- If you had to hand this off to another engineer, would they prefer pipeline code or scattered preprocessing/modeling code? Why?

Write 8–12 sentences.

---

## Day 3 – Systematic Hyperparameter Tuning with GridSearchCV

**Objectives:**
- Use **GridSearchCV** to tune hyperparameters within a pipeline.
- Practice on your classification pipeline (e.g., Titanic).

### 1. Notebook

Create: `week10_day3_gridsearch_classification.ipynb`.

Rebuild (or import) your Titanic pipeline `clf` from Day 1.

### 2. Define Parameter Grid (with Pipeline Names)

Recall: inside the pipeline, your model step is called `"model"`.

Example for LogisticRegression:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.01, 0.1, 1, 10],  # inverse regularization strength
    "model__penalty": ["l2"],        # stick to l2 for now
    # you might also experiment with class_weight
}
```

### 3. Set Up GridSearchCV

```python
grid_search = GridSearchCV(
    clf,                # pipeline
    param_grid,
    cv=5,
    scoring="f1",       # or "roc_auc" or "accuracy" depending on your goal
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

Check results:

```python
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
```

Evaluate best model on test:

```python
best_clf = grid_search.best_estimator_
y_test_pred = best_clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_test_pred))
```

### 4. Inspect Full CV Results

```python
import pandas as pd

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[["params", "mean_test_score", "std_test_score"]]
```

### 5. Thinking Challenge

In Markdown:

- Did the tuned hyperparameters significantly improve performance over default?
- Why might **cross‑validated performance** possibly differ from test performance?
- If you had more hyperparameters (like switching model types), how would you extend the grid (and what are the risks of making it too big)?

Write 8–12 sentences.

---

## Day 4 – RandomizedSearchCV & Comparing Multiple Model Types

**Objectives:**
- Use **RandomizedSearchCV** when the search space is larger.
- Compare different model families (e.g., LogisticRegression vs RandomForest vs GradientBoosting) within the same pipeline structure.

### 1. Notebook

Create: `week10_day4_randomsearch_model_comparison.ipynb`.

Reuse your classification dataset and `preprocessor`.

### 2. Define Multiple Pipelines

Example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

log_reg_clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

rf_clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

gb_clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])
```

### 3. RandomizedSearch for One Model (e.g., RandomForest)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    "model__n_estimators": randint(50, 300),
    "model__max_depth": [None, 3, 5, 10],
    "model__min_samples_split": randint(2, 10),
}

rf_search = RandomizedSearchCV(
    rf_clf,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)
print("Best RF CV F1:", rf_search.best_score_)
```

Repeat a smaller search for GradientBoosting if time allows.

### 4. Evaluate and Compare on Test Set

Evaluate:

- Tuned Logistic (from Day 3).
- Tuned RandomForest.
- Maybe tuned GradientBoosting.

Create a results table:

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    if y_proba is not None:
        row["roc_auc"] = roc_auc_score(y_test, y_proba)
    return row

results = []
results.append(evaluate_model("LogReg", grid_search.best_estimator_, X_test, y_test))
results.append(evaluate_model("RandomForest", rf_search.best_estimator_, X_test, y_test))
# add GB if you tuned it

import pandas as pd
pd.DataFrame(results)
```

### 5. Thinking Challenge

In Markdown:

- When would you prefer **RandomizedSearchCV** over GridSearchCV?
- How do you balance:
  - Number of parameter combinations.
  - CV folds.
  - Training time?
- If two models have very similar test F1 but one is much simpler/faster, which should you choose and why?

Write 10–15 sentences.

---

## Day 5 – Robust Evaluation: Cross‑Validation on the Entire Pipeline

**Objectives:**
- Evaluate the full pipeline using cross‑validation for a more stable estimate.
- Understand variance in performance.

### 1. Notebook

Create: `week10_day5_cv_on_pipelines.ipynb`.

Choose **one** tuned model (e.g., best RandomForest pipeline).

### 2. Cross‑Validation

Use cross‑validation directly on your **full pipeline** (not splitting train/test manually this time, or use only training part):

```python
from sklearn.model_selection import cross_val_score

best_rf_pipeline = rf_search.best_estimator_

cv_scores = cross_val_score(
    best_rf_pipeline,
    X, y,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("CV F1 scores:", cv_scores)
print("Mean F1:", cv_scores.mean(), "Std:", cv_scores.std())
```

Discuss:

- Mean performance.
- Variation across folds (std).

### 3. Train Final Model on All Data

Once you’re happy:

```python
final_model = best_rf_pipeline
final_model.fit(X, y)
```

Note: In a real project, you’d probably retain a hidden test set, but this is fine for now as a learning exercise.

### 4. Thinking Challenge

In Markdown:

- Why might CV mean F1 be a better indicator of performance than a **single** train/test split?
- What does a high standard deviation across folds suggest?
- If you saw that your model performs great on some folds but poorly on others, what would you investigate?

Write 8–12 sentences.

---

## Day 6 – Refactoring a Previous Project with Pipelines & Tuning

**Objectives:**
- Take one of your older projects (regression or classification) and **refactor** it to use:
  - Pipelines
  - ColumnTransformer
  - Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)

### 1. Choose Project

Pick one:

- House price regression (Week 3).
- Titanic classification (Week 4).
- Your Week 7 Kaggle‑style dataset.

### 2. Notebook

Create: `week10_day6_project_refactor.ipynb`.

### 3. Refactor Steps

Follow this sequence:

1. Load data.
2. Define target + feature sets.
3. Define `numeric_features` and `categorical_features`.
4. Create `preprocessor` with `ColumnTransformer`.
5. Create at least **two** pipelines for two different models:
   - For regression: Ridge + RandomForestRegressor.
   - For classification: LogisticRegression + GradientBoostingClassifier (or RandomForest).
6. Use **GridSearchCV or RandomizedSearchCV** to tune at least one hyperparameter for each pipeline.
7. Evaluate best model(s) on held‑out test set.
8. Summarize results in a small table.

### 4. Thinking Challenge

In Markdown:

- How much did refactoring clarify and simplify your previous notebook?
- Did you discover any **bugs or leakage** in your old workflow when you tried to pipeline it?
- If you had to put this into a production script or API, how would the pipeline help?

Write 10–15 sentences.

---

## Day 7 – Week 10 Mini‑Project: End‑to‑End, Pipeline‑Based ML System

**Objectives:**
- Create a polished, end‑to‑end notebook that reads like a real ML project.
- Combine:
  - Clean pipeline.
  - Hyperparameter tuning.
  - Solid evaluation and comparison.
  - Clear documentation.

Use your **favorite dataset so far** (classification or regression). I’ll assume classification (e.g., Titanic or your Week 7 dataset).

### 1. Notebook

Create: `week10_end_to_end_pipeline_project.ipynb`.

### 2. Project Structure

Use clear headings:

1. **Problem Definition**
   - What are you predicting (target)?
   - Why would someone care (business/user value)?
   - What kind of model task is this (classification/regression)?

2. **Data Loading & Initial Inspection**
   - Load data.
   - `head()`, `info()`, `describe()`.
   - Basic notes about missing values and feature types.

3. **Feature Engineering Plan**
   - Decide on:
     - Which columns to keep.
     - Engineered features (e.g., FamilySize, IsAlone, ratios).
   - Implement these features in code.

4. **Preprocessing & Pipelines**
   - Define `numeric_features`, `categorical_features`.
   - Build `preprocessor` with ColumnTransformer.
   - Create at least two pipelines:
     - Baseline linear/logistic model.
     - One tree‑based/ensemble model (RF or GB).

5. **Train/Validation/Test Split & Hyperparameter Tuning**
   - Create train/test split.
   - On training data (possibly with CV inside):
     - Use GridSearchCV/RandomizedSearchCV to tune hyperparameters for each pipeline.
   - Report:
     - Best params.
     - Best CV score.

6. **Final Evaluation on Test Set**
   - Evaluate best pipeline(s) on test set.
   - Use appropriate metrics:
     - Classification: Accuracy, F1, ROC AUC, confusion matrix.
     - Regression: RMSE, MAE, R².
   - Compare models in a table.

7. **Model Interpretation**
   - For tree/ensemble models:
     - Show feature importances.
   - Briefly interpret top features.

8. **Conclusions & Next Steps**
   - Summarize:
     - Which model you’d choose and why.
     - Strengths and weaknesses of the final system.
   - Suggest 3–5 next steps:
     - More features.
     - Different models (e.g., XGBoost).
     - Better cross‑validation scheme.
     - Deployment considerations.

### 3. Thinking / Stretch Tasks

Optional but recommended:

- **Reproducibility**:
  - Set random seeds consistently.
  - Export best model with `joblib.dump`.
- **Simple Config Refactor**:
  - Put key parameters (like target column name, list of features, chosen model class) in a small config dictionary or at the top of the notebook.

### 4. Non‑Technical Summary

Write a 8–12 sentence executive summary:

- What problem you solved.
- What data you used.
- What models you tried, and which performed best.
- How you made the system robust (pipelines, CV, tuning).
- What the model can and **cannot** reliably do.

---

## After Week 10: What You Should Be Able to Do

You should now:

- Build **clean, leak‑free pipelines** for both regression and classification.
- Apply **ColumnTransformer** to handle mixed numeric/categorical features.
- Use **GridSearchCV** and **RandomizedSearchCV** for systematic hyperparameter tuning.
- Compare multiple model families on the same dataset using consistent evaluation.
- Produce notebooks that resemble real, professional ML workflows.

From here, natural next steps:

- Week 11: focus on **error analysis, model comparison & communication** (turn your projects into portfolio‑ready case studies).
- Or dive further into:
  - **Advanced tree ensembles** (XGBoost/LightGBM/CatBoost).
  - **MLOps / deployment** (APIs, model monitoring).
  - Or deeper **deep learning** (CNNs for images, NLP models) depending on your interests.
