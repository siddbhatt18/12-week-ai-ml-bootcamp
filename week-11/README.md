**Week 11 Goal:**  
Work on a **larger, messier project**: handle real‑world data issues (missingness, leakage, feature design), structure your code better, and iterate like a real ML practitioner.

Assume ~1.5–2 hours/day. Ideally you choose a **new dataset** that’s a bit more complex than previous ones (e.g., a Kaggle competition dataset with non‑trivial cleaning).

---

## Day 1 – Choose a “Messy” Dataset & Set Project Structure

**Objectives**
- Pick a substantial dataset for Week 11.
- Set up a clear project structure (folders, notebooks, or scripts).
- Do a quick scan for messiness.

**Tasks**

1. **Pick your Week 11 dataset**
   - Good choices:
     - Kaggle competitions: “Titanic” is okay, but something like:
       - “House Prices: Advanced Regression Techniques”
       - “Telco Customer Churn”
       - Any competition with >20 features and some missing data.
   - Download CSV(s) and place under a folder, e.g. `projects/week11/data/`.

2. **Project structure**
   - Create directories:
     - `projects/week11/data/`
     - `projects/week11/notebooks/`
     - (Optional) `projects/week11/src/` for functions if you like.
   - New notebook: `projects/week11/notebooks/week11_day1_overview.ipynb`.

3. **Load data & quick inspection**
   ```python
   import pandas as pd

   train_df = pd.read_csv("../data/train.csv")  # adjust path/name
   train_df.head()
   train_df.info()
   train_df.describe(include="all")
   train_df.isna().sum().sort_values(ascending=False)
   ```

4. **Problem definition (markdown)**
   - What is the **target** column?
   - Is this **regression** or **classification**?
   - How many rows/columns?
   - Any obvious complications:
     - Multiple files to join?
     - Lots of missing values?
     - Strange encodings (e.g., “?” for missing)?

5. **Quick “messiness” scan**
   - Check:
     - Columns with many missing values (e.g., >40%).
     - Object columns that likely represent numeric categories or dates.
     ```python
     missing_pct = train_df.isna().mean().sort_values(ascending=False)
     missing_pct.head(20)
     ```
     ```python
     obj_cols = train_df.select_dtypes(include=["object"]).columns
     for col in obj_cols[:10]:
         print(col, "unique:", train_df[col].nunique())
         print(train_df[col].value_counts().head(), "\n")
     ```

6. **Mini reflection**
   - List 3–5 main challenges you anticipate (e.g., “many rare categories”, “highly skewed target”, “date/time features”, etc.).

**Outcome Day 1**
- Dataset and goal fixed, and you have a high‑level sense of data complexity and issues.

---

## Day 2 – Deeper Data Cleaning Plan & Guarding Against Leakage

**Objectives**
- Plan a **cleaning + preprocessing strategy** tailored to this dataset.
- Identify places where **data leakage** might easily occur.

**Tasks**

1. **New notebook**
   - `week11_day2_cleaning_plan.ipynb` (or continue Day 1 with clear headings).

2. **Understand target & potential leakage**
   - In markdown, answer:
     - What exactly does each row represent (customer/month, house at sale time, etc.)?
     - When is the target known in time?
     - Are there features that look like they were recorded *after* the outcome (e.g., “days since churn”, “sold price category”, “default_flag”)?
   - In code, inspect 10–20 suspicious columns if any (names like “Outcome”, “Status”, “Flag”, etc.).

3. **Plan cleaning strategy (markdown)**
   For each of these categories, briefly outline your approach:

   - **Numerical missing values**:
     - Median/mean? Domain‑specific fill? Or drop columns with extreme missingness?
   - **Categorical missing**:
     - “Unknown” category? Or infer from related columns?
   - **Outliers**:
     - Winsorization (clip extreme values), log transforms, or leave them?
   - **Dates / temporal features**:
     - Extract year, month, time since event, etc.
   - **High‑cardinality categoricals**:
     - One‑hot encode? Target encode? Group rare categories?

4. **Start implementing simple cleaning utilities**
   - In the notebook or a `src/utils.py` file:
   ```python
   import numpy as np
   import pandas as pd

   def basic_cleaning(df):
       df = df.copy()

       # Example: strip whitespace from string columns
       obj_cols = df.select_dtypes(include=["object"]).columns
       for col in obj_cols:
           df[col] = df[col].astype(str).str.strip()

       # Other generic fixes can go here...

       return df
   ```

5. **Apply to train_df**
   ```python
   clean_train_df = basic_cleaning(train_df)
   clean_train_df.info()
   clean_train_df.isna().sum().sort_values(ascending=False).head(20)
   ```

6. **Mini reflection**
   - Note specific potential leakage risks (e.g., using target‑related info in features).
   - Write 3–4 rules you’ll follow to avoid leakage (e.g., “no imputing using full dataset including test”, “no encoding target in features without proper CV”).

**Outcome Day 2**
- You have a documented cleaning/preprocessing plan and initial cleaning utilities, and you’re explicitly watching for leakage.

---

## Day 3 – Implement a Robust Preprocessing Pipeline for Messy Data

**Objectives**
- Turn your cleaning plan into a **ColumnTransformer + Pipeline** that handles mixed types and missing values.
- Keep it robust and structured.

**Tasks**

1. **New notebook**
   - `week11_day3_preprocessing_pipeline.ipynb`.

2. **Re‑load cleaned data**
   ```python
   train_df = pd.read_csv("../data/train.csv")
   train_df = basic_cleaning(train_df)

   target_col = "..."  # your target
   id_cols = ["..."]   # any IDs
   X = train_df.drop(columns=[target_col] + id_cols)
   y = train_df[target_col]
   ```

3. **Identify final numeric & categorical lists**
   ```python
   num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
   cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
   print("Numeric:", len(num_cols), "Categorical:", len(cat_cols))
   ```

4. **Design tailored transformers**
   - Numeric:
     - Median imputation.
     - Optional scaling (Standard or RobustScaler).
   - Categorical:
     - Frequent category imputation.
     - One‑hot with `handle_unknown="ignore"`.

   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import OneHotEncoder, StandardScaler

   numeric_transformer = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="median")),
       ("scaler", StandardScaler())
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
   ```

5. **Quick sanity check with a simple model**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression, LinearRegression

   X_train, X_valid, y_train, y_valid = train_test_split(
       X, y, test_size=0.2, random_state=42,
       stratify=y if y.nunique() < 20 else None
   )

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

6. **Check transformed feature matrix**
   ```python
   X_train_transformed = model.named_steps["preprocessor"].transform(X_train)
   X_train_transformed.shape
   ```

7. **Evaluate quickly to ensure nothing is broken**
   - Classification: accuracy/F1 on `X_valid`, `y_valid`.
   - Regression: RMSE/R² on `X_valid`, `y_valid`.

8. **Mini reflection**
   - Did any columns cause issues (e.g., all missing, constant, weird types)?
   - Are you comfortable that all cleaning is happening **inside** pipeline/ColumnTransformer where possible?

**Outcome Day 3**
- A robust preprocessing pipeline that can handle your messy dataset and feed any sklearn model.

---

## Day 4 – Feature Engineering Experimentation

**Objectives**
- Design and implement a few **feature engineering ideas**.
- Evaluate if they actually help.

**Tasks**

1. **New notebook**
   - `week11_day4_feature_engineering.ipynb`.

2. **Brainstorm feature ideas (markdown)**
   - 5–10 bullet points of potential features, e.g.:
     - Ratios (e.g., `total_income / loan_amount`).
     - Aggregates (e.g., “total rooms” = bedrooms + bathrooms).
     - Log transforms for highly skewed numeric features (e.g., `log1p(price)`).
     - Bucketization (e.g., “age bands”, “tenure buckets”).
     - Interaction terms (e.g., `rooms * neighborhood_quality`).

3. **Implement a small custom transformer**
   - Example: create a `FeatureEngineer` class that adds new columns:
   ```python
   from sklearn.base import BaseEstimator, TransformerMixin

   class FeatureEngineer(BaseEstimator, TransformerMixin):
       def __init__(self):
           pass

       def fit(self, X, y=None):
           return self

       def transform(self, X):
           X = X.copy()
           # Example: if your dataset has 'GrLivArea' and 'TotalBsmtSF':
           # X["total_living_area"] = X["GrLivArea"] + X["TotalBsmtSF"]
           # Add your own domain‑specific features here
           return X
   ```

4. **Integrate into pipeline before ColumnTransformer**
   ```python
   fe_pipeline = Pipeline(steps=[
       ("feat_eng", FeatureEngineer()),
       ("preprocessor", preprocessor),
       ("model", ... )  # RF, GB, etc.
   ])
   ```

5. **Compare with/without feature engineering**
   - Use cross‑validation:
   ```python
   from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
   from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
   import numpy as np

   if y.nunique() < 20:
       base_model = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", RandomForestClassifier(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ])
       fe_model = Pipeline(steps=[
           ("feat_eng", FeatureEngineer()),
           ("preprocessor", preprocessor),
           ("model", RandomForestClassifier(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ])
       cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
       scoring = "roc_auc"
   else:
       base_model = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", RandomForestRegressor(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ])
       fe_model = Pipeline(steps=[
           ("feat_eng", FeatureEngineer()),
           ("preprocessor", preprocessor),
           ("model", RandomForestRegressor(
               n_estimators=200,
               random_state=42,
               n_jobs=-1
           ))
       ])
       cv = KFold(n_splits=5, shuffle=True, random_state=42)
       scoring = "neg_root_mean_squared_error"

   scores_base = cross_val_score(base_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
   scores_fe = cross_val_score(fe_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

   print("Base model:", scores_base.mean(), scores_base.std())
   print("FE model :", scores_fe.mean(), scores_fe.std())
   ```

6. **Mini reflection**
   - Did your engineered features improve CV performance?
   - If not, why might that be (noisy features, wrong ideas, model already capturing relationships)?

**Outcome Day 4**
- You’ve tried systematic feature engineering and measured its impact via CV.

---

## Day 5 – Handling Real‑World Issues: Outliers, Imbalance, & Robustness

**Objectives**
- Address **outliers**, **class imbalance**, and robustness checks.
- See how these affect model quality.

**Tasks**

1. **New notebook**
   - `week11_day5_real_world_issues.ipynb`.

2. **Outlier inspection (regression or skewed numeric)**
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   import numpy as np

   for col in num_cols[:5]:
       plt.figure(figsize=(4,3))
       sns.boxplot(x=train_df[col])
       plt.title(col)
       plt.show()
   ```

3. **Experiment: target transform for regression**
   - If regression and target is skewed:
     ```python
     import numpy as np

     y_log = np.log1p(y)

     # Adjust pipeline to use y_log; compare RMSE in original and log space.
     ```

4. **Class imbalance (classification)**
   ```python
   if y.nunique() < 20:
       print(y.value_counts(normalize=True))
   ```
   - If heavy imbalance:
     - Try:
       - `class_weight="balanced"` in logistic/RF/GB.
       - Simple oversampling for minority class (e.g., `RandomOverSampler` from imbalanced‑learn, optional if you’re comfortable installing).

5. **Add class weighting in a pipeline**
   ```python
   from sklearn.ensemble import GradientBoostingClassifier

   if y.nunique() == 2:
       gb_imbalanced = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", GradientBoostingClassifier(
               random_state=42
               # GradientBoosting doesn't have class_weight; use for logistic/RF, or use XGBoost scale_pos_weight
           ))
       ])
   ```

   - For logistic or random forest:
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier

   log_bal = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", LogisticRegression(
           max_iter=1000,
           class_weight="balanced"
       ))
   ])

   rf_bal = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", RandomForestClassifier(
           n_estimators=200,
           random_state=42,
           n_jobs=-1,
           class_weight="balanced"
       ))
   ])
   ```

   - Compare CV `roc_auc`/`f1` for balanced vs unbalanced models.

6. **Robustness check: simulate missing / noisy inputs**
   - Take a subset of `X_valid`, introduce some missing values or noise, and see how model handles it. This is more to think about robustness than a strict procedure.

7. **Mini reflection**
   - Did handling imbalance improve recall for minority class?
   - What trade‑offs did you see (e.g., lower overall accuracy but better F1/AUC)?

**Outcome Day 5**
- You have tackled practical issues like outliers and imbalance, and you’ve seen their effect on performance.

---

## Day 6 – Code Organization & Experiment Tracking

**Objectives**
- Start organizing your work more like a real project:
  - Reusable functions.
  - Clear experiment logs.
- Prepare to finalize a substantial project notebook.

**Tasks**

1. **New notebook**
   - `week11_day6_code_structure.ipynb`.

2. **Identify repeated code**
   - Scanning your Week 11 notebooks:
     - Data loading & cleaning.
     - Preprocessor creation.
     - Model building & evaluation.

3. **Extract utilities into functions**
   - Either in the notebook or in a separate `../src/utils.py`:
   ```python
   def load_and_clean_data(path):
       df = pd.read_csv(path)
       df = basic_cleaning(df)
       return df

   def build_preprocessor(num_cols, cat_cols):
       numeric_transformer = Pipeline(steps=[
           ("imputer", SimpleImputer(strategy="median")),
           ("scaler", StandardScaler())
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
       return preprocessor
   ```

4. **Simple experiment log (even just a list/dict)**
   - In markdown or a small CSV/JSON:
     - Record:
       - Model type
       - Preprocessing variant
       - Key hyperparameters
       - CV score(s)
       - Test score
   - Example table structure in notebook:
   ```python
   experiments = [
       {"model": "LogReg", "desc": "baseline", "auc_cv": 0.78, "auc_test": 0.76},
       {"model": "RF", "desc": "200 trees, default", "auc_cv": 0.84, "auc_test": 0.82},
       # add more rows as you try them
   ]
   ```

5. **Decide on final pipeline for Week 11 project**
   - Based on experiments:
     - Pick the best combination of:
       - Preprocessing
       - Feature engineering (on/off)
       - Model family & hyperparameters
   - You’ll formalize this in tomorrow’s “project” notebook.

6. **Mini reflection**
   - How did organizing utilities make your code cleaner?
   - How might you track experiments more systematically in the future (e.g., spreadsheets, MLFlow, Weights & Biases)?

**Outcome Day 6**
- Cleaner, reusable code structure and a clear record of model experiments to date.

---

## Day 7 – Week 11 Mini Project: Messy Data → Clean Model

**Objectives**
- Create a polished, narrative notebook showcasing:
  - Real‑world data cleaning.
  - Robust preprocessing pipeline.
  - Feature engineering attempts.
  - Handling of issues like imbalance.
  - Final tuned model with interpretation.

**Tasks**

1. **New notebook**
   - `week11_day7_messy_data_project.ipynb`.

2. **Structure**

   ### 1. Problem & Dataset
   - 3–5 sentences:
     - Real‑world context.
     - Target, task type.
     - Ruggedness of data (missing values, many categories, etc.).

   ### 2. Data Cleaning & Understanding
   - Show:
     - Original `head()`, `.info()`, and missingness summary.
   - Describe:
     - Main data issues identified (bullet list).
   - Briefly show your `basic_cleaning` steps.

   ### 3. Feature Typing & Preprocessing Pipeline
   - Define:
     - `num_cols`, `cat_cols`, `id_cols`.
   - Show the preprocessing pipeline (ColumnTransformer + imputation + encoding/scaling).
   - Emphasize that all preprocessing is **inside** the pipeline.

   ### 4. Feature Engineering
   - Explain 2–4 engineered features (what & why).
   - Compare:
     - Base model vs feature‑engineered model (CV table with main metric).
   - Keep at least one simple experiment that didn’t help, with a note on why you think it failed.

   ### 5. Handling Real‑World Issues
   - If classification:
     - Discuss class imbalance (show target distribution).
     - Compare model with/without class weighting (or other technique).
   - If regression:
     - Mention outliers and whether you transformed target or features.

   ### 6. Final Model & Evaluation
   - Present your chosen final pipeline:
     - Preprocessing
     - Model type and key hyperparameters
   - Show:
     - Cross‑validated performance.
     - Final test set metrics.
   - Interpret:
     - Feature importances.
     - (Optional) 1–2 partial dependence plots.
   - 6–10 bullet points summarizing what the model learned and how good it is in practical terms.

   ### 7. Lessons Learned & Limitations
   - Discuss:
     - Biggest challenges in cleaning and preprocessing.
     - Limitations of your current approach (e.g., no time‑aware splitting, simplistic imputation).
     - How you might improve:
       - Better feature engineering.
       - Different models (boosting, neural nets).
       - More careful temporal validation if data is time‑based.

3. **Polish**
   - Make sure all code runs top‑to‑bottom.
   - Remove dead/duplicate code cells.
   - Add concise but clear markdown explanations between sections.

**Outcome Day 7**
- A substantial, realistic project showcasing your ability to handle messy data, design robust preprocessing, and iterate thoughtfully—excellent preparation for Week 12, where you’ll focus on polishing your best projects and portfolio.
