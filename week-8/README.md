**Week 8 Goal:**  
Deepen your understanding of **model evaluation**, **cross‑validation**, and **model comparison**. By the end of this week you should be able to:  
- Design proper evaluation schemes (train/val/test).  
- Use cross‑validation well (including stratified CV).  
- Compare models fairly with multiple metrics.  
- Communicate results clearly.

Use the same main dataset(s) from Weeks 6–7 (one classification dataset; optionally also 1 regression dataset). Aim for ~1.5–2 hours/day.

---

## Day 1 – Evaluation Strategy: Train/Validation/Test & Data Leakage

**Objectives**
- Clarify the roles of train, validation, and test sets.
- Understand data leakage and how to avoid it.

**Tasks**

1. **New notebook**
   - `week8_day1_eval_strategy.ipynb`.

2. **Concepts in markdown (short, but precise)**
   In your own words, write:

   - What is the purpose of:
     - **Training set**  
     - **Validation set**  
     - **Test set**
   - Why you typically:
     - Tune hyperparameters on **validation** (or via cross‑validation on training).
     - Only evaluate once on **test** at the end.
   - What is **data leakage**?  
     Give 2 examples (e.g., using test data in preprocessing, using features generated from the target).

3. **Implement explicit train/val/test split**

   Use your main dataset:

   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split

   df = pd.read_csv("data/your_dataset.csv")
   target_col = "Survived"  # or your target
   id_cols = ["PassengerId"]  # adjust for your dataset

   X = df.drop(columns=[target_col] + id_cols)
   y = df[target_col]

   # First: split off test set
   X_temp, X_test, y_temp, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42,
       stratify=y if y.nunique() < 20 else None
   )

   # Second: split temp into train/val
   X_train, X_val, y_train, y_val = train_test_split(
       X_temp, y_temp, test_size=0.25, random_state=42,
       stratify=y_temp if y.nunique() < 20 else None
   )

   X_train.shape, X_val.shape, X_test.shape
   ```

   - Result: 60% train, 20% val, 20% test (approx).

4. **Connect this to pipelines / no leakage**

   - Re‑define your `preprocessor` (ColumnTransformer) and pick one simple model (e.g., logistic regression or random forest).

   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.impute import SimpleImputer
   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression
   import numpy as np

   num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
   cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

   numeric_transformer = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="median")),
   ])

   categorical_transformer = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="most_frequent")),
       ("onehot", OneHotEncoder(handle_unknown="ignore"))
   ])

   preprocessor = ColumnTransformer(
       transformers=[
           ("num", numeric_transformer, num_cols),
           ("cat", categorical_transformer, cat_cols),
       ]
   )

   model = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("clf", LogisticRegression(max_iter=1000))
   ])

   model.fit(X_train, y_train)
   ```

   - Note: **fit only on train**.  
     Then evaluate on **val**, and finally on **test** once you’re done tuning.

5. **Evaluate on val set (not test yet)**
   ```python
   from sklearn.metrics import accuracy_score, classification_report

   y_val_pred = model.predict(X_val)
   print("Val accuracy:", accuracy_score(y_val, y_val_pred))
   print("Val classification report:\n", classification_report(y_val, y_val_pred))
   ```

6. **Mini reflection (markdown)**
   - Why is it important you didn’t touch `X_test` yet?
   - Give 2 things that would cause data leakage if done incorrectly.

**Outcome Day 1**
- You can implement a clean train/val/test split and understand where leakage can occur.

---

## Day 2 – Cross‑Validation Basics & Stratified k‑Fold

**Objectives**
- Learn how k‑fold and stratified k‑fold cross‑validation work.
- Use `cross_val_score` with pipelines.

**Tasks**

1. **New notebook**
   - `week8_day2_crossval_basics.ipynb`.

2. **Concepts in markdown**
   Answer:

   - What is **k‑fold cross‑validation**?
   - Why is **stratified** k‑fold important for classification?
   - When might you prefer CV over a single train/validation split?

3. **StratifiedKFold demo (classification)**

   ```python
   from sklearn.model_selection import StratifiedKFold, cross_val_score
   from sklearn.linear_model import LogisticRegression
   from sklearn.pipeline import Pipeline

   base_model = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("clf", LogisticRegression(max_iter=1000))
   ])

   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   scores = cross_val_score(
       base_model,
       X, y,
       cv=skf,
       scoring="accuracy",
       n_jobs=-1
   )

   import numpy as np
   print("CV scores:", scores)
   print("Mean:", scores.mean(), "Std:", scores.std())
   ```

4. **Compare multiple scoring metrics**
   ```python
   from sklearn.model_selection import cross_validate

   scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

   cv_results = cross_validate(
       base_model,
       X, y,
       cv=skf,
       scoring=scoring,
       return_train_score=False,
       n_jobs=-1
   )

   import pandas as pd
   pd.DataFrame(cv_results)
   ```

   - Compute column means:
     ```python
     pd.DataFrame(cv_results).mean()
     ```

5. **Mini exercise**
   - Try a different model (e.g., RandomForest or GradientBoosting) in the same CV setup.
   - Compare mean `accuracy` and `f1` for the two models.

6. **Mini reflection**
   - Are CV scores similar across folds (low std) or highly variable?
   - When might a high variance across folds be a warning sign?

**Outcome Day 2**
- You can perform stratified cross‑validation with multiple metrics and interpret mean ± std.

---

## Day 3 – Robust Model Comparison with Cross‑Validation

**Objectives**
- Compare several models in a consistent CV framework.
- Evaluate models beyond a single metric.

**Tasks**

1. **New notebook**
   - `week8_day3_robust_model_comparison.ipynb`.

2. **Define candidate models**

   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from xgboost import XGBClassifier  # if installed, or skip if not

   models = {
       "LogReg": Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", LogisticRegression(max_iter=1000))
       ]),
       "RandomForest": Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", RandomForestClassifier(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ]),
       "GradBoost": Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", GradientBoostingClassifier(
               random_state=42,
               n_estimators=200,
               learning_rate=0.1,
               max_depth=3
           ))
       ]),
   }

   # Optional XGBoost model
   models["XGBoost"] = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", XGBClassifier(
           n_estimators=200,
           learning_rate=0.1,
           max_depth=3,
           subsample=0.8,
           colsample_bytree=0.8,
           eval_metric="logloss",
           tree_method="hist",
           random_state=42,
           n_jobs=-1
       ))
   ])
   ```

3. **Cross‑validate all models**

   ```python
   from sklearn.model_selection import StratifiedKFold, cross_validate

   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   scoring = ["accuracy", "f1", "roc_auc"]

   import pandas as pd
   rows = []

   for name, model in models.items():
       cv_results = cross_validate(
           model,
           X, y,
           cv=skf,
           scoring=scoring,
           return_train_score=False,
           n_jobs=-1
       )
       row = {"model": name}
       for metric in scoring:
           row[f"{metric}_mean"] = cv_results[f"test_{metric}"].mean()
           row[f"{metric}_std"] = cv_results[f"test_{metric}"].std()
       rows.append(row)

   results_df = pd.DataFrame(rows)
   results_df.sort_values(by="roc_auc_mean", ascending=False)
   ```

4. **Interpret the results**
   - Which model has the best **AUC**?
   - Is the ranking the same for `accuracy_mean` and `f1_mean`?
   - Which model seems most stable across folds (`*_std`)?

5. **Mini reflection**
   - If two models are very close in mean metrics but one is simpler (logistic regression), how might you decide?
   - In a real case, what other factors matter (training time, interpretability, deployment)?

**Outcome Day 3**
- You can run fair, metric‑rich model comparisons using cross‑validation.

---

## Day 4 – Choosing and Calibrating a Final Model (Validation → Test)

**Objectives**
- Use cross‑validation (or val set) to select a final model and hyperparameters.
- Evaluate once on the untouched test set.

**Tasks**

1. **New notebook**
   - `week8_day4_model_selection_to_test.ipynb`.

2. **Use previous results to pick 1–2 contenders**
   - From Day 3 results, choose:
     - One simple model (e.g., LogisticRegression).
     - One strong model (e.g., GradientBoosting or XGBoost).

3. **Hyperparameter tuning for top model using train+val (no test)**
   - Combine `X_train` and `X_val` into `X_train_full`, `y_train_full`:
     ```python
     import pandas as pd

     X_train_full = pd.concat([X_train, X_val], axis=0)
     y_train_full = pd.concat([y_train, y_val], axis=0)
     ```

   - Set up a tuned model, e.g., GradientBoosting:

     ```python
     from sklearn.ensemble import GradientBoostingClassifier
     from sklearn.model_selection import GridSearchCV

     gb_base = Pipeline(steps=[
         ("preprocessor", preprocessor),
         ("model", GradientBoostingClassifier(random_state=42))
     ])

     param_grid = {
         "model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3, 4],
     }

     grid_search = GridSearchCV(
         estimator=gb_base,
         param_grid=param_grid,
         cv=5,
         scoring="roc_auc",
         n_jobs=-1,
         verbose=1
     )

     grid_search.fit(X_train_full, y_train_full)

     print("Best params:", grid_search.best_params_)
     print("Best CV AUC:", grid_search.best_score_)
     best_model = grid_search.best_estimator_
     ```

4. **Final evaluation on test set**
   ```python
   from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

   y_test_pred = best_model.predict(X_test)
   y_test_proba = best_model.predict_proba(X_test)[:, 1]

   print("Test accuracy:", accuracy_score(y_test, y_test_pred))
   print("Test AUC:", roc_auc_score(y_test, y_test_proba))
   print("Test classification report:\n", classification_report(y_test, y_test_pred))
   ```

5. **Compare to a simple baseline model**

   ```python
   simple_model = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", LogisticRegression(max_iter=1000))
   ])

   simple_model.fit(X_train_full, y_train_full)
   y_test_pred_simple = simple_model.predict(X_test)
   y_test_proba_simple = simple_model.predict_proba(X_test)[:, 1]

   print("Simple Test accuracy:", accuracy_score(y_test, y_test_pred_simple))
   print("Simple Test AUC:", roc_auc_score(y_test, y_test_proba_simple))
   ```

6. **Mini reflection**
   - Does the tuned model clearly outperform the simple baseline on the test set?
   - How does the test AUC compare to cross‑validation AUC (over/under‑estimation)?

**Outcome Day 4**
- You can go from CV‑based model selection to a final test evaluation in a disciplined way.

---

## Day 5 – Learning Curves and When More Data Helps

**Objectives**
- Use **learning curves** to see how performance scales with training set size.
- Decide whether gathering more data would likely help.

**Tasks**

1. **New notebook**
   - `week8_day5_learning_curves.ipynb`.

2. **Pick one model (e.g., your best GradientBoosting or RandomForest)**

   ```python
   from sklearn.ensemble import GradientBoostingClassifier
   from sklearn.pipeline import Pipeline

   best_pipe = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", GradientBoostingClassifier(
           random_state=42,
           n_estimators=200,
           learning_rate=0.1,
           max_depth=3
       ))
   ])
   ```

3. **Compute learning curve**

   ```python
   from sklearn.model_selection import learning_curve
   import numpy as np
   import matplotlib.pyplot as plt

   train_sizes, train_scores, val_scores = learning_curve(
       estimator=best_pipe,
       X=X,
       y=y,
       cv=5,
       scoring="roc_auc",  # or "accuracy" for simplicity
       train_sizes=np.linspace(0.1, 1.0, 5),
       n_jobs=-1
   )

   train_mean = train_scores.mean(axis=1)
   train_std = train_scores.std(axis=1)
   val_mean = val_scores.mean(axis=1)
   val_std = val_scores.std(axis=1)

   plt.figure(figsize=(7, 5))
   plt.plot(train_sizes, train_mean, "o-", label="Train score")
   plt.plot(train_sizes, val_mean, "o-", label="Validation score")
   plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
   plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
   plt.xlabel("Training set size")
   plt.ylabel("Score (AUC)")
   plt.title("Learning Curve")
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

4. **Interpret the curve**
   - If both train and validation scores are low and close together → **high bias**: model may be too simple.
   - If train is high and validation is much lower → **high variance**: more data or stronger regularization may help.

5. **Mini reflection**
   - Based on the curve, would you:
     - Focus on getting more data?
     - Try a more/less complex model?
     - Add regularization or do hyperparameter tuning?
   - 5–6 sentences summarizing your observations.

**Outcome Day 5**
- You can generate and interpret learning curves to inform next steps in model development.

---

## Day 6 – Evaluation for Regression (Optional if You Only Have Classification)

**Objectives**
- Practice similar evaluation ideas for a regression task.
- Compare metrics: RMSE, MAE, R².

**Tasks** (skip or shorten if you’re only doing classification)

1. **New notebook**
   - `week8_day6_regression_eval.ipynb`.

2. **Pick a regression dataset**
   - e.g., California Housing (`fetch_california_housing`) or House Prices.

   ```python
   from sklearn.datasets import fetch_california_housing
   import pandas as pd

   data = fetch_california_housing(as_frame=True)
   df_reg = data.frame
   target_col_reg = "MedHouseVal"

   X_reg = df_reg.drop(columns=[target_col_reg])
   y_reg = df_reg[target_col_reg]
   ```

3. **Train/test split + preprocessing**
   - Define numeric/categorical (likely mostly numeric).
   - Build `preprocessor_reg` (imputation + optional scaling).
   - Create 2 models:
     - LinearRegression
     - RandomForestRegressor (or GradientBoostingRegressor).

4. **Cross‑validation with regression metrics**
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import Pipeline
   from sklearn.impute import SimpleImputer
   from sklearn.linear_model import LinearRegression
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import cross_validate, KFold
   import numpy as np

   num_cols_reg = X_reg.select_dtypes(include=["int64", "float64"]).columns.tolist()

   preprocessor_reg = ColumnTransformer(
       transformers=[
           ("num", SimpleImputer(strategy="median"), num_cols_reg)
       ]
   )

   models_reg = {
       "LinearReg": Pipeline(steps=[
           ("preprocessor", preprocessor_reg),
           ("model", LinearRegression())
       ]),
       "RandomForestReg": Pipeline(steps=[
           ("preprocessor", preprocessor_reg),
           ("model", RandomForestRegressor(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ])
   }

   scoring_reg = {
       "rmse": "neg_root_mean_squared_error",
       "mae": "neg_mean_absolute_error",
       "r2": "r2"
   }

   kf = KFold(n_splits=5, shuffle=True, random_state=42)

   import pandas as pd
   rows = []
   for name, model in models_reg.items():
       cv_results = cross_validate(
           model, X_reg, y_reg,
           cv=kf,
           scoring=scoring_reg,
           return_train_score=False,
           n_jobs=-1
       )
       row = {"model": name}
       for metric_name, metric_key in scoring_reg.items():
           scores = cv_results[f"test_{metric_name}"]
           row[f"{metric_name}_mean"] = -scores.mean() if metric_name in ["rmse", "mae"] else scores.mean()
           row[f"{metric_name}_std"] = scores.std()
       rows.append(row)

   pd.DataFrame(rows)
   ```

5. **Mini reflection**
   - Which regression metric do you find most intuitive (RMSE, MAE, R²)?
   - Does the tree‑based model significantly beat linear regression?

**Outcome Day 6**
- You can extend your evaluation skills to regression, using appropriate metrics and CV.

---

## Day 7 – Week 8 Mini Project: Evaluation & Comparison Report

**Objectives**
- Create a polished “evaluation‑centric” notebook:
  - Clear problem framing.
  - Careful evaluation strategy.
  - Cross‑validated model comparison.
  - Final model choice with test evaluation and commentary.

**Tasks**

1. **New notebook**
   - `week8_day7_mini_project_evaluation.ipynb`.

2. **Structure**

   ### 1. Problem & Data
   - 4–6 sentences:
     - Dataset, target, task type.
     - Why model evaluation matters here (e.g., false negatives vs false positives).

   ### 2. Evaluation Strategy
   - Markdown:
     - How you split data: train/val/test.
     - If/when you use cross‑validation vs a simple validation set.
     - How you avoid data leakage (preprocessing inside pipelines).

   ### 3. Baseline Models
   - Define:
     - Simple baseline: majority class baseline (classification) or mean baseline (regression).
     - Simple model: LogisticRegression / LinearRegression.
   - Report:
     - Baseline metric(s) (accuracy, RMSE, etc.).
     - Simple model CV metrics (mean ± std).

   ### 4. Strong Models & CV Comparison
   - Fit and cross‑validate:
     - RandomForest.
     - GradientBoosting (and optionally XGBoost).
   - Use 2–3 metrics:
     - Classification: accuracy, F1, AUC.
     - Regression: RMSE, MAE, R².
   - Summarize results in a comparison table:
     - Model name, mean metrics, std metrics.

   ### 5. Model Selection & Final Test Evaluation
   - Choose the final model based on CV.
   - Tune a small hyperparameter grid on train+val if needed.
   - Evaluate final model on **test** once:
     - Key metrics.
     - Confusion matrix (for classification).
     - Short discussion: how good is this in practice?

   ### 6. Learning Curve and Conclusions
   - Include one learning curve for the final model (even if already computed in Day 5).
   - Markdown bullets (8–10):
     - What did CV reveal about model stability?
     - Did the final test performance match CV expectations?
     - Do you see high bias or high variance?
     - Is there evidence that more data would help?
     - Main strengths/weaknesses of the final model.

3. **Polish**
   - Ensure notebook runs top‑to‑bottom without errors.
   - Add concise explanations before/after major code sections.
   - Clear plots (ROC, learning curve) and labeled axes.

**Outcome Day 7**
- A well‑structured evaluation and model comparison notebook that demonstrates you understand how to properly judge and select models—excellent preparation for Week 9, where you’ll start working more with neural networks.
