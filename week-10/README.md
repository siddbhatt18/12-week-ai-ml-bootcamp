**Week 10 Goal:**  
Practice an end‑to‑end ML workflow: from problem framing and data to model, evaluation, basic interpretability, and saving/using your model. This week is about **gluing everything together** into something closer to a real project.

Assume ~1.5–2 hours/day. Use one main tabular dataset (ideally the one you care most about for your portfolio—e.g., Titanic, House Prices, Churn, or a Kaggle dataset you like).

---

## Day 1 – Project Framing & Data Understanding

**Objectives**
- Choose a concrete problem and dataset for an end‑to‑end project.
- Clearly define target, features, and success metrics.

**Tasks**

1. **Pick your Week 10 project dataset**
   - Options:
     - A Kaggle competition dataset (even if the comp is over).
     - Titanic / Telco Churn / House Prices / Ames Housing.
   - Choose **one** and stick with it all week.

2. **New notebook**
   - `week10_day1_project_setup.ipynb`.

3. **Problem framing (markdown)**
   - Write 1–2 short paragraphs:
     - What is the **business or practical question**?
       - Examples:
         - “Can we predict customer churn to prioritize retention?”
         - “Can we estimate house sale prices to guide listing strategies?”
     - What is the **target variable**?
     - Is this a **classification** or **regression** problem?
     - What are the **constraints** or stakes?
       - E.g., false negatives are worse than false positives, or vice versa.

4. **Load & inspect data**
   ```python
   import pandas as pd

   df = pd.read_csv("data/your_dataset.csv")  # adjust path
   df.head()
   df.info()
   df.describe(include="all")
   df.isna().sum()
   ```

5. **Initial target/feature selection**
   ```python
   target_col = "..."  # set this
   id_cols = ["..."]   # any ID columns, e.g., "PassengerId", "CustomerID"
   X = df.drop(columns=[target_col] + id_cols)
   y = df[target_col]
   ```

6. **Basic checks**
   - Value counts for target (classification) or summary stats (regression):
     ```python
     y.value_counts(normalize=True)  # classification
     # or
     y.describe()                    # regression
     ```
   - Quick list of numeric vs categorical cols:
     ```python
     num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
     cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
     num_cols, cat_cols
     ```

7. **Mini reflection (markdown)**
   - Is your target balanced or imbalanced?
   - Any obvious data issues (lots of missing values, weird types)?

**Outcome Day 1**
- A clearly defined project (problem, target, task type) and a first look at the data.

---

## Day 2 – Focused EDA & Simple Baseline

**Objectives**
- Perform focused EDA relevant to the target.
- Build a very simple baseline model for comparison.

**Tasks**

1. **New notebook**
   - `week10_day2_eda_and_baseline.ipynb` (or continue from Day 1, with clear headings).

2. **Focused EDA**
   - Examine target vs key features:
     - For classification:
       ```python
       import seaborn as sns
       import matplotlib.pyplot as plt

       # Example for one numeric feature
       for col in num_cols[:5]:
           plt.figure(figsize=(4,3))
           sns.boxplot(x=y, y=df[col])
           plt.title(f"{col} vs target")
           plt.show()
       ```
     - For regression:
       ```python
       for col in num_cols[:5]:
           plt.figure(figsize=(4,3))
           sns.scatterplot(x=df[col], y=y)
           plt.title(f"{col} vs target")
           plt.show()
       ```
   - Look at:
     - Histograms for a few numeric features.
     - Value counts for a few important categoricals.

3. **Write 5–8 EDA observations in markdown**
   - Examples:
     - “Higher tenure seems associated with lower churn.”
     - “Expensive houses are mostly in neighborhood X.”
     - “Class 3 passengers have lower survival rates.”

4. **Simple baseline model**
   - Split train/test **once** (you’ll later do CV if needed):
     ```python
     from sklearn.model_selection import train_test_split

     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42,
         stratify=y if y.nunique() < 20 else None
     )
     ```
   - Basic preprocessing:
     ```python
     from sklearn.compose import ColumnTransformer
     from sklearn.pipeline import Pipeline
     from sklearn.impute import SimpleImputer
     from sklearn.preprocessing import OneHotEncoder
     import numpy as np

     numeric_transformer = Pipeline(steps=[
         ("imputer", SimpleImputer(strategy="median"))
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

   - Simple model:
     - Classification: `LogisticRegression`.
     - Regression: `LinearRegression`.

     ```python
     from sklearn.linear_model import LogisticRegression, LinearRegression

     if y.nunique() < 20:
         model = Pipeline(steps=[
             ("preprocessor", preprocessor),
             ("model", LogisticRegression(max_iter=1000))
         ])
     else:
         model = Pipeline(steps=[
             ("preprocessor", preprocessor),
             ("model", LinearRegression())
         ])

     model.fit(X_train, y_train)
     ```

5. **Evaluate baseline**
   - Classification:
     ```python
     from sklearn.metrics import accuracy_score, classification_report

     y_pred = model.predict(X_test)
     print("Baseline Test accuracy:", accuracy_score(y_test, y_pred))
     print("Classification report:\n", classification_report(y_test, y_pred))
     ```
   - Regression:
     ```python
     from sklearn.metrics import mean_squared_error, r2_score
     import numpy as np

     y_pred = model.predict(X_test)
     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     r2 = r2_score(y_test, y_pred)
     print("Baseline Test RMSE:", rmse, "R²:", r2)
     ```

6. **Mini reflection**
   - What is your baseline performance?
   - What metric will you focus on improving (accuracy, F1, AUC, RMSE, etc.)?

**Outcome Day 2**
- Insightful EDA grounded in the target and a simple yet working baseline model.

---

## Day 3 – Stronger Models & Cross‑Validated Comparison

**Objectives**
- Plug in stronger models (Random Forest, Gradient Boosting, maybe XGBoost).
- Compare them with cross‑validation.

**Tasks**

1. **New notebook**
   - `week10_day3_model_comparison.ipynb`.

2. **Reuse `preprocessor` and `X`, `y`** (no train/test split yet for CV).

3. **Define candidate models**
   ```python
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
   from sklearn.linear_model import LogisticRegression, LinearRegression
   from sklearn.pipeline import Pipeline

   classification = (y.nunique() < 20)

   models = {}

   if classification:
       models["LogReg"] = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", LogisticRegression(max_iter=1000))
       ])
       models["RandomForest"] = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", RandomForestClassifier(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ])
       models["GradBoost"] = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", GradientBoostingClassifier(
               random_state=42,
               n_estimators=200,
               learning_rate=0.1,
               max_depth=3
           ))
       ])
   else:
       models["LinearReg"] = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", LinearRegression())
       ])
       models["RandomForestReg"] = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", RandomForestRegressor(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ])
       models["GradBoostReg"] = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", GradientBoostingRegressor(
               random_state=42,
               n_estimators=200,
               learning_rate=0.1,
               max_depth=3
           ))
       ])
   ```

4. **Cross‑validation comparison**
   ```python
   from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
   import pandas as pd

   if classification:
       cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
       scoring = ["accuracy", "f1", "roc_auc"]
   else:
       cv = KFold(n_splits=5, shuffle=True, random_state=42)
       scoring = ["neg_root_mean_squared_error", "r2"]

   rows = []
   for name, pipe in models.items():
       cv_results = cross_validate(
           pipe, X, y,
           cv=cv,
           scoring=scoring,
           n_jobs=-1,
           return_train_score=False
       )
       row = {"model": name}
       for metric in scoring:
           scores = cv_results[f"test_{metric}"]
           mean_score = scores.mean()
           std_score = scores.std()
           if metric.startswith("neg_"):
               mean_score = -mean_score
           row[f"{metric}_mean"] = mean_score
           row[f"{metric}_std"] = std_score
       rows.append(row)

   results_df = pd.DataFrame(rows)
   results_df
   ```

5. **Sort & inspect**
   - For classification:
     ```python
     results_df.sort_values(by="roc_auc_mean", ascending=False)
     ```
   - For regression:
     ```python
     results_df.sort_values(by="neg_root_mean_squared_error_mean", ascending=True)  # smaller RMSE is better
     ```

6. **Mini reflection**
   - Which model is best according to your main metric?
   - Is the improvement over baseline (linear/logistic) large enough to justify extra complexity?

**Outcome Day 3**
- A clear model comparison table using cross‑validation, guiding which model to focus on.

---

## Day 4 – Hyperparameter Tuning of the Best Model

**Objectives**
- Perform a focused hyperparameter search on your best model.
- Use CV on the training set only and keep the test set untouched.

**Tasks**

1. **New notebook**
   - `week10_day4_hyperparameter_tuning.ipynb`.

2. **Create train/test split for final evaluation later**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42,
       stratify=y if classification else None
   )
   ```

3. **Choose best model type from Day 3**
   - Suppose it’s `RandomForest` (similar for GradientBoosting, etc.).
   ```python
   from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
   from sklearn.model_selection import GridSearchCV

   if classification:
       base_model = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
       ])
       param_grid = {
           "model__n_estimators": [100, 200],
           "model__max_depth": [None, 5, 10],
           "model__min_samples_leaf": [1, 5, 10]
       }
       scoring_metric = "roc_auc"
   else:
       base_model = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
       ])
       param_grid = {
           "model__n_estimators": [100, 200],
           "model__max_depth": [None, 5, 10],
           "model__min_samples_leaf": [1, 5, 10]
       }
       scoring_metric = "neg_root_mean_squared_error"
   ```

4. **GridSearchCV on training data only**
   ```python
   grid_search = GridSearchCV(
       estimator=base_model,
       param_grid=param_grid,
       cv=5,
       scoring=scoring_metric,
       n_jobs=-1,
       verbose=1
   )

   grid_search.fit(X_train, y_train)

   print("Best params:", grid_search.best_params_)
   print("Best CV score:", grid_search.best_score_)
   best_model = grid_search.best_estimator_
   ```

5. **Evaluate tuned model on test set**
   - Classification:
     ```python
     from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

     y_test_pred = best_model.predict(X_test)
     y_test_proba = best_model.predict_proba(X_test)[:, 1]

     print("Test accuracy:", accuracy_score(y_test, y_test_pred))
     print("Test AUC:", roc_auc_score(y_test, y_test_proba))
     print("Test classification report:\n", classification_report(y_test, y_test_pred))
     ```
   - Regression:
     ```python
     from sklearn.metrics import mean_squared_error, r2_score
     import numpy as np

     y_test_pred = best_model.predict(X_test)
     rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
     r2_test = r2_score(y_test, y_test_pred)
     print("Test RMSE:", rmse_test, "R²:", r2_test)
     ```

6. **Mini reflection**
   - How much did tuning improve performance over default?
   - Does test performance match CV expectations, or is there a gap?

**Outcome Day 4**
- A tuned model with solid test performance and understanding of how hyperparameters influenced it.

---

## Day 5 – Model Interpretability: Feature Importance & Simple Explanations

**Objectives**
- Interpret your final model via feature importances.
- Optionally use permutation importance or partial dependence plots.

**Tasks**

1. **New notebook**
   - `week10_day5_interpretability.ipynb`.

2. **Refit your best model if needed**
   - Ensure `best_model` is fitted on `X_train`, `y_train`.

3. **Feature importances (tree‑based model)**
   ```python
   import numpy as np
   import pandas as pd

   # Extract underlying model and feature names from pipeline
   model_step = [step for name, step in best_model.named_steps.items() if name != "preprocessor"][0]
   raw_model = model_step  # e.g., RandomForestClassifier

   # Get transformed feature names:
   ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
   cat_feature_names = ohe.get_feature_names_out(cat_cols)
   feature_names = np.concatenate([num_cols, cat_feature_names])

   importances = raw_model.feature_importances_
   fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
   fi.head(15)
   ```

4. **Plot top feature importances**
   ```python
   import matplotlib.pyplot as plt

   top_n = 15
   plt.figure(figsize=(6, 6))
   fi.head(top_n).sort_values().plot(kind="barh")
   plt.xlabel("Importance")
   plt.title("Top Feature Importances")
   plt.show()
   ```

5. **Optional: Permutation importance**
   ```python
   from sklearn.inspection import permutation_importance

   r = permutation_importance(
       best_model, X_test, y_test,
       n_repeats=10,
       random_state=42,
       n_jobs=-1
   )

   pi = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
   pi.head(15)
   ```

6. **Optional: Partial Dependence Plot for 1–2 key features**
   ```python
   from sklearn.inspection import PartialDependenceDisplay

   important_feats = fi.head(2).index.tolist()

   for feat in important_feats:
       fig, ax = plt.subplots(figsize=(5,4))
       PartialDependenceDisplay.from_estimator(
           best_model, X_train, [feat], ax=ax
       )
       plt.title(f"Partial Dependence of {feat}")
       plt.show()
   ```

7. **Mini reflection**
   - Which features are most important? Does that align with domain intuition?
   - From PDPs, how does increasing a feature (e.g., tenure, number of rooms) affect predicted outcome?

**Outcome Day 5**
- You can explain what your model is learning in terms of key features and simple effect plots.

---

## Day 6 – Saving, Loading, and Using the Model (Mini “Deployment”)

**Objectives**
- Learn to serialize (save) your trained pipeline.
- Write a minimal script or function to use the model on new data.

**Tasks**

1. **New notebook**
   - `week10_day6_model_saving_usage.ipynb`.

2. **Save the model**
   ```python
   import joblib

   joblib.dump(best_model, "best_model_pipeline.joblib")
   ```

3. **Load the model**
   ```python
   loaded_model = joblib.load("best_model_pipeline.joblib")
   ```

4. **Create a small “prediction API” function**
   - For example, a function that takes a dict of feature values and returns a prediction:

   ```python
   import pandas as pd

   def predict_single(sample_dict):
       # sample_dict: {"feature1": value1, "feature2": value2, ...}
       sample_df = pd.DataFrame([sample_dict])
       pred = loaded_model.predict(sample_df)[0]
       return pred

   # Example usage (adjust feature names):
   example = {
       "Age": 35,
       "Sex": "female",
       "Pclass": 2,
       # ...
   }
   print("Prediction:", predict_single(example))
   ```

5. **Batch prediction on new data**
   - Simulate “new” data using part of `X_test`:
     ```python
     X_new = X_test.sample(5, random_state=42)
     preds = loaded_model.predict(X_new)
     preds
     ```

6. **Optional: tiny CLI or script**
   - Make a `.py` file (`predict.py`) that:
     - Loads the model.
     - Reads a CSV of new examples.
     - Prints predictions.
   - Even just structuring this in your head or notes is useful.

7. **Mini reflection**
   - Why is it important to save the full pipeline (preprocessing + model) instead of just the model?
   - What could go wrong in deployment if preprocessing is not exactly the same?

**Outcome Day 6**
- You can persist a trained model pipeline and use it on new, raw‑like data—first step toward real deployment.

---

## Day 7 – Week 10 Mini Project: End‑to‑End Report Notebook

**Objectives**
- Combine everything into a polished, narrative notebook suitable for a portfolio piece:
  - Problem → data → EDA → modeling → tuning → interpretation → usage.

**Tasks**

1. **New notebook**
   - `week10_day7_end_to_end_project.ipynb`.

2. **Suggested structure**

   ### 1. Title & Abstract
   - Short title: “Predicting [Target] from [Dataset]”.
   - 3–5 sentence abstract:
     - Problem context.
     - Main approach.
     - Final results headline (e.g., “Achieved AUC of 0.89 using Gradient Boosted Trees”).

   ### 2. Problem Definition
   - Clear description of:
     - Business/real‑world motivation.
     - Target and task type.
     - Which metric matters most and why.

   ### 3. Data Understanding & EDA
   - Loading & initial inspection.
   - Focused EDA:
     - A few key plots/tables.
     - 5–8 bullet points of insights.

   ### 4. Baseline Model
   - Explain baseline approach (simple linear/logistic model).
   - Show:
     - Baseline metric(s) on test set.
   - Brief discussion.

   ### 5. Model Development & Comparison
   - Candidate models (Logistic/Linear, RF, GB, etc.).
   - Cross‑validated comparison table.
   - Justification of chosen model to tune further.

   ### 6. Hyperparameter Tuning & Final Model
   - Description of the search space.
   - Best hyperparameters and CV score.
   - Final test metrics.
   - Simple tables/plots to summarize.

   ### 7. Interpretation
   - Feature importances plot.
   - (Optionally) 1–2 Partial Dependence Plots.
   - 6–8 bullet points of interpretation:
     - Which features matter most.
     - How they influence predictions.
     - Any surprising or domain‑specific insights.

   ### 8. Deployment Considerations
   - Brief description:
     - How you save/load the model (`joblib`).
     - How to call it on new data (pseudo‑API or example function).
   - Potential concerns:
     - Data drift.
     - Missing or new categories.
     - Privacy or fairness issues (if relevant).

   ### 9. Conclusions & Future Work
   - 6–10 bullet points:
     - Summary of what you achieved.
     - Model strengths and limitations.
     - Next steps:
       - Better feature engineering.
       - Trying boosting libraries (XGBoost/LightGBM).
       - Collecting more/cleaner data.
       - Building a simple web app with Streamlit, etc.

3. **Polish**
   - Make sure the notebook runs top‑to‑bottom without errors.
   - Keep plots labeled and legible.
   - Use clear section headings and concise, informative markdown cells.

**Outcome Day 7**
- A complete, end‑to‑end ML project notebook that you could show as part of a portfolio, demonstrating not just individual techniques but a coherent, real‑world workflow.
