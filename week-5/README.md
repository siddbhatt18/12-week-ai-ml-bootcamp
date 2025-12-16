**Week 5 Goal:**  
Master practical **feature preprocessing and pipelines** using scikit‑learn. By the end of the week, you should be able to take messy tabular data with mixed numeric and categorical features, build a clean preprocessing + model pipeline, and evaluate it robustly.

Assumption: ~1.5–2 hours/day. Use **one main dataset with both numeric and categorical features** (Titanic or Ames Housing are ideal).

---

## Day 1 – Review Your Dataset & Identify Feature Types

**Objectives**
- Choose a main Week‑5 dataset.
- Clearly identify which columns are numeric, categorical, text‑like, or IDs.
- Decide on a target variable and basic problem type (regression or classification).

**Tasks**

1. **Choose dataset & open notebook**
   - New notebook: `week5_day1_data_review.ipynb`.
   - Load your dataset (example: Titanic from Kaggle):
     ```python
     import pandas as pd

     df = pd.read_csv("data/titanic.csv")  # adjust path/name
     df.head()
     df.info()
     ```

2. **Basic inspection**
   - Show:
     ```python
     df.head()
     df.sample(5, random_state=42)
     df.describe(include="all")
     df.isna().sum()
     ```

3. **Identify target and problem type**
   - For Titanic example:
     - Target = `"Survived"` (0/1) → classification.
   - For Ames/House Prices:
     - Target = `"SalePrice"` → regression.
   - In markdown, state:
     - What is the target?
     - Regression or classification?

4. **Categorize columns**
   - Programmatically:
     ```python
     num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
     cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

     num_cols, cat_cols
     ```
   - Manually adjust lists:
     - Move ID‑like columns (e.g., `PassengerId`, `Name`) to `id_cols`.
     - Some integers might actually be categories (e.g., “Pclass” on Titanic).

   ```python
   id_cols = ["PassengerId"]  # adjust
   # Example: Pclass is categorical though it's int
   num_cols = [c for c in num_cols if c not in id_cols + ["Pclass"]]
   cat_cols = cat_cols + ["Pclass"]
   ```

5. **Mini exercise**
   - In markdown, list:
     - `num_cols`
     - `cat_cols`
     - `id_cols` (to be dropped from features).
   - Brief justification for 2–3 columns you classified as categorical vs numeric.

**Outcome Day 1**
- Clear decision: which columns are numeric/categorical/IDs.
- Target variable and task type defined.

---

## Day 2 – Manual Preprocessing (Without Pipelines) to Understand Steps

**Objectives**
- Manually perform:
  - Dropping ID columns.
  - Handling missing values (simple strategies).
  - Encoding categoricals (one‑hot encoding).
- Get intuition before automating via pipelines.

**Tasks**

1. **New notebook**
   - `week5_day2_manual_preprocessing.ipynb`.
   - Load your dataset and reuse `num_cols`, `cat_cols`, `id_cols` from Day 1.

2. **Drop ID columns & separate target**
   ```python
   target_col = "Survived"  # or your target
   X = df.drop(columns=[target_col] + id_cols)
   y = df[target_col]
   ```

3. **Handle missing values (simple approach)**
   - Numeric: fill with median.
   - Categorical: fill with “Unknown”.
   ```python
   import numpy as np

   X_num = X[num_cols].copy()
   X_cat = X[cat_cols].copy()

   for col in num_cols:
       X_num[col] = X_num[col].fillna(X_num[col].median())

   for col in cat_cols:
       X_cat[col] = X_cat[col].fillna("Unknown")
   ```

4. **One‑hot encode categorical columns (pandas)**
   ```python
   X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)  # to avoid full multicollinearity
   ```

5. **Combine numeric + encoded categorical**
   ```python
   X_processed = pd.concat([X_num, X_cat_encoded], axis=1)
   X_processed.head()
   X_processed.shape
   ```

6. **Train/test split + simple model**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score

   X_train, X_test, y_train, y_test = train_test_split(
       X_processed, y, test_size=0.2, random_state=42, stratify=y
   )

   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```

7. **Mini reflection**
   - How many features after one‑hot encoding?
   - What are the main tedious parts of doing this manually?

**Outcome Day 2**
- You can preprocess a mixed‑type dataset “by hand” and fit a basic model.
- You see why automating with pipelines is useful.

---

## Day 3 – ColumnTransformer & Pipelines (Core Skill)

**Objectives**
- Use **ColumnTransformer** to apply different preprocessing to numeric vs categorical columns.
- Wrap preprocessing + model into a **single Pipeline**.

**Tasks**

1. **New notebook**
   - `week5_day3_columntransformer_pipeline.ipynb`.

2. **Prepare data**
   - Load original raw dataset again (not preprocessed).
   - Define `X` and `y` as on Day 2 (dropping ID columns, not manually handling missing now).

3. **Define transformers**
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.impute import SimpleImputer

   numeric_transformer = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="median")),
       # optionally add scaler later: ("scaler", StandardScaler())
   ])

   categorical_transformer = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="most_frequent")),
       ("onehot", OneHotEncoder(handle_unknown="ignore"))
   ])

   preprocessor = ColumnTransformer(
       transformers=[
           ("num", numeric_transformer, num_cols),
           ("cat", categorical_transformer, cat_cols)
       ]
   )
   ```

4. **Create full pipeline with a model**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression

   clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", LogisticRegression(max_iter=1000))
   ])
   ```

5. **Train/test split and fit**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```

6. **Mini exercises**
   - Add a **StandardScaler** into the numeric pipeline:
     ```python
     from sklearn.preprocessing import StandardScaler

     numeric_transformer = Pipeline(steps=[
         ("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler())
     ])
     ```
   - Refit and compare accuracy.
   - Print the number of features **after** preprocessing:
     ```python
     X_train_trans = clf.named_steps["preprocessor"].fit_transform(X_train)
     X_train_trans.shape
     ```

7. **Reflection**
   - How is this approach better than manual preprocessing?
   - Why is it important that preprocessing is inside the pipeline (not done before split)?

**Outcome Day 3**
- You can build a robust `ColumnTransformer` + `Pipeline` that handles both numeric and categorical data cleanly.

---

## Day 4 – Using Pipelines with Tree‑Based Models

**Objectives**
- Swap the linear model with tree‑based models (Decision Tree, Random Forest) inside the same preprocessing pipeline.
- Compare performance and behavior.

**Tasks**

1. **New notebook**
   - `week5_day4_trees_with_pipelines.ipynb`.
   - Reuse `X`, `y`, `num_cols`, `cat_cols`, `preprocessor` from Day 3.

2. **Decision Tree pipeline**
   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.pipeline import Pipeline
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   tree_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", DecisionTreeClassifier(random_state=42))
   ])

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   tree_clf.fit(X_train, y_train)
   y_pred_tree = tree_clf.predict(X_test)
   print("Decision Tree accuracy:", accuracy_score(y_test, y_pred_tree))
   ```

3. **Random Forest pipeline**
   ```python
   from sklearn.ensemble import RandomForestClassifier

   rf_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", RandomForestClassifier(
           n_estimators=100,
           random_state=42,
           n_jobs=-1
       ))
   ])

   rf_clf.fit(X_train, y_train)
   y_pred_rf = rf_clf.predict(X_test)
   print("Random Forest accuracy:", accuracy_score(y_test, y_pred_rf))
   ```

4. **Compare to logistic regression**
   - Quickly refit logistic regression pipeline from Day 3 and record its accuracy.
   - Create a small comparison table:
     ```python
     import pandas as pd

     results = [
         ("LogisticRegression", logreg_acc),
         ("DecisionTree", accuracy_score(y_test, y_pred_tree)),
         ("RandomForest", accuracy_score(y_test, y_pred_rf)),
     ]

     pd.DataFrame(results, columns=["Model", "Test Accuracy"])
     ```

5. **Feature importance (from Random Forest)**
   - Extract after fitting:
     ```python
     # Get feature names after preprocessing
     ohe = rf_clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
     cat_feature_names = ohe.get_feature_names_out(cat_cols)
     all_feature_names = np.concatenate([num_cols, cat_feature_names])

     import numpy as np

     rf_model = rf_clf.named_steps["model"]
     importances = rf_model.feature_importances_
     fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
     fi.head(10)
     ```

6. **Mini reflection**
   - Which model performs best?
   - Do tree‑based models seem to capture nonlinear patterns better than logistic regression?
   - Which features does the random forest consider most important?

**Outcome Day 4**
- You can plug different models into your preprocessing pipeline and compare them.
- You’ve seen feature importance from a tree‑based model.

---

## Day 5 – Hyperparameter Tuning with Pipelines (GridSearchCV)

**Objectives**
- Use `GridSearchCV` to tune hyperparameters of a model **inside** a pipeline.
- Understand how to search a small, meaningful hyperparameter grid.

**Tasks**

1. **New notebook**
   - `week5_day5_gridsearch_pipelines.ipynb`.

2. **Base pipeline (Random Forest example)**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.pipeline import Pipeline

   base_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
   ])
   ```

3. **Define parameter grid**
   - Use `model__` prefix to refer to parameters of the model in the pipeline:
   ```python
   param_grid = {
       "model__n_estimators": [100, 200],
       "model__max_depth": [None, 5, 10],
       "model__min_samples_split": [2, 10],
   }
   ```

4. **Run GridSearchCV**
   ```python
   from sklearn.model_selection import GridSearchCV

   grid_search = GridSearchCV(
       estimator=base_clf,
       param_grid=param_grid,
       cv=3,
       scoring="accuracy",
       n_jobs=-1,
       verbose=1
   )

   grid_search.fit(X, y)
   ```

5. **Inspect results**
   ```python
   print("Best params:", grid_search.best_params_)
   print("Best CV accuracy:", grid_search.best_score_)

   cv_results = pd.DataFrame(grid_search.cv_results_)
   cv_results[["params", "mean_test_score", "std_test_score"]].sort_values(
       by="mean_test_score", ascending=False
   ).head()
   ```

6. **Evaluate best model on a hold‑out test set**
   - Better: re‑do your own train/test split, run GridSearchCV on the **train only**, and test on `X_test, y_test`:
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   grid_search = GridSearchCV(
       estimator=base_clf,
       param_grid=param_grid,
       cv=3,
       scoring="accuracy",
       n_jobs=-1,
       verbose=1
   )
   grid_search.fit(X_train, y_train)

   best_model = grid_search.best_estimator_
   from sklearn.metrics import accuracy_score
   y_pred_test = best_model.predict(X_test)
   print("Test accuracy (best model):", accuracy_score(y_test, y_pred_test))
   ```

7. **Mini reflection**
   - How much did tuning improve your performance vs default?
   - Which hyperparameters made the biggest difference?

**Outcome Day 5**
- You can run a grid search over a pipeline, understand the syntax, and interpret the best parameters and scores.

---

## Day 6 – Handling Real‑World Messiness & Robustness

**Objectives**
- Stress‑test your pipeline with:
  - Unexpected categories.
  - Dropped/extra columns.
- Make sure preprocessing is robust and doesn’t break easily.

**Tasks**

1. **New notebook**
   - `week5_day6_robustness.ipynb`.

2. **Refit your best pipeline**
   - Use the tuned or best‑performing pipeline from Day 5 (or Day 4) and fit on full training data (`X_train`, `y_train`).

3. **Simulate new data with unseen categories**
   - Take a few rows from `X_test` and modify categorical values:
     ```python
     X_new = X_test.copy().iloc[:5].copy()

     # Example: if you have a column "Embarked":
     if "Embarked" in X_new.columns:
         X_new["Embarked"] = "Z"  # a category not seen in training
     ```
   - Call:
     ```python
     y_new_pred = best_model.predict(X_new)
     y_new_pred
     ```
   - Because `OneHotEncoder(handle_unknown="ignore")` is set, this should work without crashing.

4. **What if a column is missing?**
   - Drop a column in X_new:
     ```python
     X_missing = X_test.copy().iloc[:5].copy()
     X_missing = X_missing.drop(columns=[some_col_name])
     ```
   - Try:
     ```python
     best_model.predict(X_missing)
     ```
   - It should raise an error: the pipeline expects the same columns.  
     Write down why this is actually helpful (catches issues early).

5. **Saving and loading your pipeline**
   ```python
   import joblib

   joblib.dump(best_model, "best_pipeline.joblib")

   loaded_model = joblib.load("best_pipeline.joblib")
   y_pred_loaded = loaded_model.predict(X_test)
   ```
   - Confirm predictions from `loaded_model` match original `best_model` (or are extremely close).

6. **Mini reflection**
   - Why is it useful that the pipeline includes all preprocessing, not just the model?
   - In a production scenario, what would break if you tried to do manual preprocessing separately and got it slightly wrong?

**Outcome Day 6**
- Your pipeline is robust to unseen categories and can be saved/loaded safely.
- You understand the importance of consistent features and automated preprocessing.

---

## Day 7 – Week 5 Mini Project: Full Preprocessing + Model Pipeline

**Objectives**
- Build a complete, well‑documented ML pipeline on your dataset:
  - Clean definition of feature types.
  - ColumnTransformer + Pipeline.
  - Model comparison + simple tuning.
  - Clear write‑up of choices and results.

**Tasks**

1. **New notebook**
   - `week5_day7_mini_project_pipeline.ipynb`.

2. **Notebook structure**

   ### 1. Problem & Data
   - 3–5 sentences:
     - What dataset is this?
     - What is the target?
     - Why is it an interesting problem?

   ### 2. Data Loading & Feature Typing
   - Load raw dataset.
   - Show `.head()`, `.info()`, and `value_counts()` for target.
   - Explicitly define:
     ```python
     target_col = "..."
     id_cols = [...]
     num_cols = [...]
     cat_cols = [...]
     ```

   ### 3. Preprocessing Strategy
   - Explain (markdown):
     - How you will handle missing values for numeric vs categorical.
     - How you will encode categoricals (OneHot).
   - Implement `ColumnTransformer` and numeric/categorical pipelines.

   ### 4. Models & Comparison
   - Define 2–3 models inside pipelines using the same `preprocessor`:
     - LogisticRegression (or LinearRegression for regression task).
     - RandomForest (and optionally DecisionTree or k‑NN).
   - Use train/test split, then:
     - Fit each model.
     - Evaluate with relevant metrics (accuracy, F1, ROC‑AUC for classification; RMSE/R² for regression).
   - Summarize in a small DataFrame.

   ### 5. Hyperparameter Tuning (Small Grid)
   - Pick the best baseline model and run a small `GridSearchCV`:
     - 1–3 hyperparameters.
     - Reasonable small search space.
   - Report:
     - Best params.
     - Best CV score.
     - Test performance of best model.

   ### 6. Interpretation & Conclusions
   - If using random forest:
     - Show top 10 feature importances.
   - In markdown, write 8–10 bullet points:
     - What preprocessing steps were most important?
     - Which model worked best and why do you think so?
     - Any evidence of overfitting?
     - Which features appear most important for predictions?
     - What would you do next if this were a real project?

3. **Polish**
   - Ensure cells execute from top to bottom with no errors.
   - Add comments and markdown explanations for major steps.
   - Rename notebook title to something descriptive (“Titanic – Preprocessing & Modeling with Pipelines”).

**Outcome Day 7**
- A complete, production‑style ML pipeline notebook handling real‑world tabular data.
- Solid grasp of ColumnTransformer, Pipelines, and how to plug in and tune different models—perfect foundation for Week 6 (decision trees, random forests, and overfitting in more depth).
