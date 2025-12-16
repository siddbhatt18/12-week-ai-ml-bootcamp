**Week 7 Goal:**  
Learn and apply **gradient boosting** (GradientBoosting, XGBoost/LightGBM if you want), and get more systematic with **hyperparameter tuning** and **model comparison**. You’ll use your existing pipelines and dataset from Weeks 5–6.

Assumption: ~1.5–2 hours/day. Use one main tabular dataset (classification or regression)—preferably the same as Week 6.

---

## Day 1 – Conceptual Foundations: Gradient Boosting

**Objectives**
- Understand boosting vs bagging.
- Get high‑level intuition for gradient boosting.

**Tasks**

1. **New notebook**
   - `week7_day1_boosting_concepts.ipynb`.

2. **Concepts in markdown (10–15 minutes)**
   In your own words, answer:

   - What is **bagging** (used by random forests)?
     - Many trees trained in parallel on bootstrapped samples; average their predictions → reduce variance.
   - What is **boosting**?
     - Build trees **sequentially**.
     - Each new tree focuses on correcting the mistakes (residuals) of the previous ones.
   - Why is gradient boosting powerful?
     - Can model complex relationships.
     - Often best on tabular data with good hyperparameter tuning.

3. **Visual intuition (text, no math needed)**
   - Write a short 5–7 line story:
     - Start with a simple prediction (e.g., mean of y).
     - Compute residuals (errors).
     - Fit a small tree to those residuals.
     - Add this tree’s predictions to the previous predictions.
     - Repeat many times with a small **learning rate**.

4. **Inspect scikit‑learn’s GradientBoosting* docs (5–10 minutes)**
   - Open the docs for `GradientBoostingClassifier` or `GradientBoostingRegressor`.
   - Note down (in markdown):
     - Key parameters: `n_estimators`, `learning_rate`, `max_depth` (or `max_leaf_nodes`), `subsample`.

5. **Mini reflection**
   - Compare random forest vs gradient boosting conceptually:
     - How they build trees (parallel vs sequential).
     - What they try to reduce (variance vs bias).

**Outcome Day 1**
- Solid conceptual grasp of boosting and how gradient boosting differs from random forests.

---

## Day 2 – First Gradient Boosting Model (scikit‑learn)

**Objectives**
- Train a basic gradient boosting model in a pipeline.
- Compare its performance to your tuned/proper random forest from Week 6.

**Tasks**

1. **New notebook**
   - `week7_day2_gb_basic_model.ipynb`.

2. **Set up data & preprocessor**
   - Load your dataset:
     ```python
     import pandas as pd

     df = pd.read_csv("data/your_dataset.csv")
     target_col = "Survived"  # or your target
     id_cols = ["PassengerId"]  # adjust

     X = df.drop(columns=[target_col] + id_cols)
     y = df[target_col]
     ```
   - Reuse or redefine `num_cols`, `cat_cols`, `preprocessor` as in Week 5/6.

3. **Train/test split**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) < 20 else None
   )
   ```

4. **GradientBoostingClassifier / Regressor pipeline**
   - For classification:
     ```python
     from sklearn.ensemble import GradientBoostingClassifier
     from sklearn.pipeline import Pipeline

     gb_clf = Pipeline(steps=[
         ("preprocessor", preprocessor),
         ("model", GradientBoostingClassifier(
             random_state=42
         ))
     ])

     gb_clf.fit(X_train, y_train)
     ```
   - For regression, use `GradientBoostingRegressor` analogously.

5. **Evaluate performance**
   - Classification example:
     ```python
     from sklearn.metrics import accuracy_score, classification_report

     y_pred = gb_clf.predict(X_test)
     print("GB Test accuracy:", accuracy_score(y_test, y_pred))
     print("Classification report:\n", classification_report(y_test, y_pred))
     ```
   - Regression example:
     ```python
     from sklearn.metrics import mean_squared_error, r2_score
     import numpy as np

     y_pred = gb_reg.predict(X_test)
     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     r2 = r2_score(y_test, y_pred)
     print("GB Test RMSE:", rmse, "R²:", r2)
     ```

6. **Compare to random forest**
   - Fit or reuse your best random forest pipeline:
     ```python
     from sklearn.ensemble import RandomForestClassifier

     rf_clf = Pipeline(steps=[
         ("preprocessor", preprocessor),
         ("model", RandomForestClassifier(
             n_estimators=200,
             random_state=42,
             n_jobs=-1
         ))
     ])
     rf_clf.fit(X_train, y_train)
     y_pred_rf = rf_clf.predict(X_test)
     from sklearn.metrics import accuracy_score
     print("Random Forest Test accuracy:", accuracy_score(y_test, y_pred_rf))
     ```

7. **Mini reflection**
   - Does gradient boosting outperform or underperform your random forest?
   - Quick guess: why might that be (hyperparameters, data size, etc.)?

**Outcome Day 2**
- Working gradient boosting model as a pipeline and an initial comparison against a random forest baseline.

---

## Day 3 – Tuning Gradient Boosting Hyperparameters (scikit‑learn)

**Objectives**
- Get a feel for how `n_estimators`, `learning_rate`, and `max_depth` affect performance.
- Do a small grid search to find better settings.

**Tasks**

1. **New notebook**
   - `week7_day3_gb_hyperparameters.ipynb`.

2. **Base pipeline**
   ```python
   from sklearn.ensemble import GradientBoostingClassifier
   from sklearn.pipeline import Pipeline

   gb_base = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", GradientBoostingClassifier(random_state=42))
   ])
   ```

3. **Small manual grid (fast, intuitive)**
   ```python
   from sklearn.metrics import accuracy_score  # or regression metrics
   import pandas as pd

   results = []
   for n_estimators in [50, 100, 200]:
       for learning_rate in [0.05, 0.1, 0.2]:
           for max_depth in [2, 3, 4]:
               gb_clf = Pipeline(steps=[
                   ("preprocessor", preprocessor),
                   ("model", GradientBoostingClassifier(
                       random_state=42,
                       n_estimators=n_estimators,
                       learning_rate=learning_rate,
                       max_depth=max_depth
                   ))
               ])
               gb_clf.fit(X_train, y_train)
               y_pred_train = gb_clf.predict(X_train)
               y_pred_test = gb_clf.predict(X_test)
               train_acc = accuracy_score(y_train, y_pred_train)
               test_acc = accuracy_score(y_test, y_pred_test)
               results.append((n_estimators, learning_rate, max_depth, train_acc, test_acc))

   res_df = pd.DataFrame(
       results,
       columns=["n_estimators", "learning_rate", "max_depth", "train_acc", "test_acc"]
   )
   res_df.sort_values(by="test_acc", ascending=False).head(10)
   ```

4. **Inspect patterns**
   - Does decreasing `learning_rate` with larger `n_estimators` help?
   - Do deeper trees (`max_depth`) overfit?

5. **(Optional) GridSearchCV or RandomizedSearchCV**
   ```python
   from sklearn.model_selection import GridSearchCV

   param_grid = {
       "model__n_estimators": [100, 200],
       "model__learning_rate": [0.05, 0.1],
       "model__max_depth": [2, 3, 4],
   }

   grid_search = GridSearchCV(
       estimator=gb_base,
       param_grid=param_grid,
       cv=3,
       scoring="accuracy",
       n_jobs=-1,
       verbose=1
   )
   grid_search.fit(X_train, y_train)

   print("Best params:", grid_search.best_params_)
   print("Best CV score:", grid_search.best_score_)
   ```

6. **Evaluate best model on test set**
   ```python
   best_gb = grid_search.best_estimator_
   y_pred_best = best_gb.predict(X_test)
   print("Best GB Test accuracy:", accuracy_score(y_test, y_pred_best))
   ```

7. **Mini reflection**
   - How much did tuning improve test performance vs the default GB model?
   - Which hyperparameters seemed most sensitive?

**Outcome Day 3**
- You can tune gradient boosting more systematically and understand key hyperparameters.

---

## Day 4 – Gradient Boosting vs Random Forest vs Linear/Logistic (Model Comparison)

**Objectives**
- Run a clean model comparison:
  - Baseline (linear/logistic)
  - Random forest
  - Gradient boosting (tuned)
- Use cross‑validation to compare fairly.

**Tasks**

1. **New notebook**
   - `week7_day4_model_comparison.ipynb`.

2. **Set up candidate models**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
       "GradBoost_default": Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", GradientBoostingClassifier(random_state=42))
       ]),
       # If you have a tuned GB model, substitute its params:
       "GradBoost_tuned": Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", GradientBoostingClassifier(
               random_state=42,
               n_estimators=200,
               learning_rate=0.05,
               max_depth=3
           ))
       ])
   }
   ```

3. **Cross‑validation comparison**
   ```python
   from sklearn.model_selection import cross_val_score
   import pandas as pd

   scoring_metrics = ["accuracy", "f1", "roc_auc"]  # for regression: ["neg_root_mean_squared_error", "r2"]

   rows = []
   for name, pipe in models.items():
       row = {"model": name}
       for metric in scoring_metrics:
           scores = cross_val_score(pipe, X, y, cv=5, scoring=metric)
           row[f"{metric}_mean"] = scores.mean()
           row[f"{metric}_std"] = scores.std()
       rows.append(row)

   results_df = pd.DataFrame(rows)
   results_df.sort_values(by="roc_auc_mean" if "roc_auc_mean" in results_df else "accuracy_mean", ascending=False)
   ```

4. **Interpretation**
   - Which model:
     - Has highest mean accuracy/AUC?
     - Has lowest std (most stable)?
   - How much better is the best tree‑based model compared to linear/logistic?

5. **(Optional) Refit best model and get detailed test metrics**
   ```python
   best_name = results_df.sort_values(by="roc_auc_mean", ascending=False)["model"].iloc[0]
   best_model = models[best_name]
   best_model.fit(X_train, y_train)
   y_pred_best = best_model.predict(X_test)
   ```

6. **Mini reflection**
   - In your problem, do tree‑based or linear models win?
   - Would you always choose gradient boosting over random forests? Why/why not (think about training time, tuning complexity, interpretability)?

**Outcome Day 4**
- A clear, metric‑based comparison of several model families, using cross‑validation.

---

## Day 5 – Introduce XGBoost or LightGBM (Optional but Valuable)

**Objectives**
- Install and run a basic **XGBoost** (or **LightGBM**) model.
- Compare its performance with scikit‑learn’s GradientBoosting.

**Tasks**

1. **New notebook**
   - `week7_day5_xgboost_intro.ipynb`.

2. **Install XGBoost (if not already)**
   - In terminal (with your env active):
     ```bash
     pip install xgboost
     ```
   - Or for LightGBM:
     ```bash
     pip install lightgbm
     ```

3. **Set up an XGBoost pipeline**
   ```python
   from xgboost import XGBClassifier  # or XGBRegressor
   from sklearn.pipeline import Pipeline

   xgb_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", XGBClassifier(
           n_estimators=200,
           learning_rate=0.1,
           max_depth=3,
           subsample=0.8,
           colsample_bytree=0.8,
           objective="binary:logistic",  # for binary classification
           eval_metric="logloss",
           tree_method="hist",  # faster
           random_state=42,
           n_jobs=-1
       ))
   ])
   ```

4. **Train/test evaluation**
   ```python
   xgb_clf.fit(X_train, y_train)

   from sklearn.metrics import accuracy_score, classification_report

   y_pred_xgb = xgb_clf.predict(X_test)
   print("XGBoost Test accuracy:", accuracy_score(y_test, y_pred_xgb))
   print("Classification report:\n", classification_report(y_test, y_pred_xgb))
   ```

5. **Quick CV comparison with GradientBoosting**
   ```python
   from sklearn.model_selection import cross_val_score

   import numpy as np

   scores_xgb = cross_val_score(xgb_clf, X, y, cv=5, scoring="accuracy")
   print("XGBoost CV mean acc:", scores_xgb.mean(), "std:", scores_xgb.std())

   gb_pipe = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", GradientBoostingClassifier(
           random_state=42,
           n_estimators=200,
           learning_rate=0.1,
           max_depth=3
       ))
   ])

   scores_gb = cross_val_score(gb_pipe, X, y, cv=5, scoring="accuracy")
   print("GradBoost CV mean acc:", scores_gb.mean(), "std:", scores_gb.std())
   ```

6. **Mini reflection**
   - Does XGBoost significantly outperform GradientBoosting on your dataset?
   - Is training time noticeably different?

**Outcome Day 5**
- You have at least one modern gradient boosting library (XGBoost/LightGBM) running and compared to sklearn GB.

---

## Day 6 – Feature Importance & Partial Dependence (Model Interpretation)

**Objectives**
- Inspect feature importances from a boosting model.
- (If possible) visualize simple **Partial Dependence Plots (PDPs)** to see how features affect predictions.

**Tasks**

1. **New notebook**
   - `week7_day6_gb_interpretation.ipynb`.

2. **Use your best gradient boosting model (sklearn GB or XGBoost)**
   - Fit it on `X_train`, `y_train` if not already fitted.

3. **Feature importances (sklearn GradientBoosting)**
   ```python
   import numpy as np
   import pandas as pd

   gb_model = best_gb.named_steps["model"]
   ohe = best_gb.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
   cat_feature_names = ohe.get_feature_names_out(cat_cols)
   feature_names = np.concatenate([num_cols, cat_feature_names])

   importances = gb_model.feature_importances_
   fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
   fi.head(10)
   ```

4. **Partial Dependence Plots (sklearn)**
   ```python
   from sklearn.inspection import PartialDependenceDisplay
   import matplotlib.pyplot as plt

   # choose 1–2 important numeric features
   top_features = fi.head(2).index.tolist()

   for feat in top_features:
       fig, ax = plt.subplots(figsize=(6, 4))
       PartialDependenceDisplay.from_estimator(
           best_gb, X_train, [feat], ax=ax
       )
       plt.title(f"Partial Dependence of {feat}")
       plt.show()
   ```

   - For classification, this shows how predicted probability of the positive class changes with the feature.

5. **(If using XGBoost) Feature importance**
   ```python
   xgb_model = xgb_clf.named_steps["model"]
   xgb_importance = xgb_model.feature_importances_
   fi_xgb = pd.Series(xgb_importance, index=feature_names).sort_values(ascending=False)
   fi_xgb.head(10)
   ```

6. **Mini reflection**
   - Which features does the boosting model rely on most?
   - From PDPs, how does increasing a particular feature affect predictions?
   - Do these patterns make sense with domain knowledge?

**Outcome Day 6**
- You can interpret gradient boosting models via feature importances and basic partial dependence.

---

## Day 7 – Week 7 Mini Project: Boosting & Tuning End‑to‑End

**Objectives**
- Build a polished notebook centered around gradient boosting:
  - Compare random forest vs gradient boosting (and optionally XGBoost).
  - Do limited but thoughtful tuning.
  - Interpret the final model.

**Tasks**

1. **New notebook**
   - `week7_day7_mini_project_boosting.ipynb`.

2. **Structure**

   ### 1. Problem Overview
   - 3–5 sentences:
     - Dataset, target, task.
     - Why tree‑based / boosting models are suitable.

   ### 2. Data & Preprocessing
   - Load dataset, define `X`, `y`, `id_cols`, `num_cols`, `cat_cols`.
   - Build `preprocessor` (ColumnTransformer with imputation + OneHot, maybe scaling numeric).

   ### 3. Baseline & Random Forest
   - Fit baseline model (logistic or linear).
   - Fit reasonable random forest.
   - Report:
     - CV metrics (accuracy/F1/AUC or RMSE/R²).
     - Test set metrics.

   ### 4. Gradient Boosting (sklearn)
   - Train a default GradientBoosting* model via pipeline.
   - Tune small hyperparameter grid (e.g., `n_estimators`, `learning_rate`, `max_depth`).
   - Report:
     - Best params.
     - CV performance.
     - Test performance.

   ### 5. (Optional) XGBoost / LightGBM
   - Fit a basic XGBoost/LightGBM model.
   - Compare its test performance to tuned sklearn GB and random forest.

   ### 6. Interpretation
   - Show top 10 feature importances for best boosting model.
   - Produce 1–2 Partial Dependence Plots for key features.
   - 8–10 bullet points in markdown:
     - Which model achieved best performance?
     - How sensitive was GB to hyperparameters?
     - What features drive predictions most?
     - Any overfitting signs (train vs test, CV mean vs test)?

   ### 7. Conclusions & Next Steps
   - 4–6 bullets:
     - When would you pick boosting vs random forests?
     - Potential risks (overfitting, tuning cost).
     - Ideas for further improvement: more features, better handling of imbalance, larger hyperparameter search, etc.

3. **Polish**
   - Clean, commented code.
   - Clear section headings and short explanations before major blocks.
   - Notebook runs from top to bottom without errors.

**Outcome Day 7**
- A strong, focused project notebook demonstrating gradient boosting, hyperparameter tuning, model comparison, and basic interpretation—excellent preparation for later weeks (cross‑validation strategy, model evaluation in depth, and eventually neural networks).
