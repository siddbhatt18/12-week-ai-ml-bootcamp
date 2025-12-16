**Week 2 Goal:**  
Get comfortable with pandas for real-world data handling and EDA, and understand the idea of train/test splitting for ML.

Assumption: ~1.5–2 hours/day. Reuse the same dataset across multiple days where possible (e.g., Titanic, Ames Housing, California Housing, or any tabular dataset with both numeric and categorical features).

---

## Day 1 – Deeper pandas: Indexing, Filtering, Sorting

**Objectives**
- Learn to select and filter rows/columns efficiently.
- Get comfortable with sorting and basic DataFrame operations.

**Tasks**
1. **Setup**
   - Open a new notebook: `week2_day1_indexing_filtering.ipynb`.
   - Load your chosen dataset:
     ```python
     import pandas as pd

     df = pd.read_csv("data/your_dataset.csv")  # adjust path
     df.head()
     ```

2. **Column selection recap**
   ```python
   # single and multiple columns
   df["some_numeric_column"].head()
   df[["col1", "col2"]].head()
   ```

3. **Row selection with loc/iloc**
   ```python
   # by integer position
   df.iloc[0]         # first row
   df.iloc[0:5]       # first 5 rows

   # by label (loc)
   df.loc[0]          # row with index label 0 (often same as first row)
   df.loc[0:10, ["col1", "col2"]]  # slice rows 0–10, specific columns
   ```

4. **Filtering rows with conditions**
   ```python
   # simple condition
   df[df["some_numeric_column"] > 100]

   # multiple conditions (use & and |, with parentheses)
   df[(df["col1"] > 50) & (df["col2"] == "A")]
   ```

5. **Sorting**
   ```python
   df_sorted = df.sort_values(by="some_numeric_column", ascending=False)
   df_sorted.head()
   ```

6. **Mini exercises**
   - Filter:
     - Rows where a numeric column is above its mean.
       ```python
       mean_val = df["some_numeric_column"].mean()
       high_rows = df[df["some_numeric_column"] > mean_val]
       ```
     - Rows where a categorical column equals a specific category.
   - Sort:
     - By two columns: one ascending, one descending.

**Outcome by end of Day 1**
- Confident selecting and filtering specific subsets of data.
- Know how to sort DataFrames by one or more columns.

---

## Day 2 – GroupBy, Aggregations, and Descriptive Stats

**Objectives**
- Summarize data by groups.
- Compute aggregated statistics and understand their meaning.

**Tasks**
1. **New notebook**
   - `week2_day2_groupby_agg.ipynb`.
   - Load same dataset as Day 1.

2. **Basic groupby**
   ```python
   # Example: group by a categorical column
   grouped = df.groupby("some_category_column")
   grouped["some_numeric_column"].mean()
   ```

3. **Multiple aggregations**
   ```python
   df.groupby("some_category_column")["some_numeric_column"].agg(["mean", "median", "count", "std"])
   ```

4. **Group by multiple columns**
   ```python
   df.groupby(["category1", "category2"])["some_numeric_column"].mean()
   ```

5. **Value counts for categorical variables**
   ```python
   df["some_category_column"].value_counts()
   df["some_category_column"].value_counts(normalize=True)  # proportions
   ```

6. **Mini exercises**
   - For your dataset, answer:
     - What is the average of a key numeric variable for each category in a categorical column?
     - Which category has the highest average of that variable?
   - If applicable:
     - Group by a categorical column (e.g., “class”, “region”, or “species”) and compute:
       - Mean and median of a numeric feature.
       - Count of rows per category.

7. **Short written reflection (markdown)**
   - 3–4 sentences describing:
     - Which groups are largest?
     - Any group that stands out with unusually high/low averages?

**Outcome by end of Day 2**
- Able to slice data by groups and compute meaningful summary statistics.
- More comfortable reading grouped/aggregated tables.

---

## Day 3 – Handling Missing Values & Data Types Properly

**Objectives**
- Detect, understand, and handle missing values more systematically.
- Fix incorrect data types (e.g., numeric stored as string).

**Tasks**
1. **New notebook**
   - `week2_day3_missing_and_dtypes.ipynb`.
   - Load your dataset:
     ```python
     import pandas as pd
     df = pd.read_csv("data/your_dataset.csv")
     ```

2. **Detect missing values**
   ```python
   df.isna().sum()
   percent_missing = df.isna().mean() * 100
   percent_missing.sort_values(ascending=False)
   ```

3. **Inspect dtypes and convert if needed**
   ```python
   df.info()

   # Example: convert a column that should be numeric but is object/string
   df["some_numeric_like_col"] = pd.to_numeric(df["some_numeric_like_col"], errors="coerce")
   ```

4. **Different strategies per column type**
   - For numeric:
     ```python
     df["num_col_filled_mean"] = df["num_col"].fillna(df["num_col"].mean())
     df["num_col_filled_median"] = df["num_col"].fillna(df["num_col"].median())
     ```
   - For categorical:
     ```python
     df["cat_col_filled"] = df["cat_col"].fillna("Unknown")
     ```

5. **Dropping vs imputing**
   ```python
   df_drop = df.dropna()   # be careful: might drop many rows
   print(df.shape, df_drop.shape)
   ```

6. **Mini exercises**
   - Choose:
     - 1 numeric column → fill missing values with median.
     - 1 categorical column → fill missing values with “Unknown”.
   - Compare before/after:
     ```python
     df["num_col"].isna().sum()
     df["num_col_filled"].isna().sum()
     ```

7. **Short written reflection**
   - Which columns had the most missing data?
   - Did you choose to drop or impute, and why?

**Outcome by end of Day 3**
- Systematic way to inspect and handle missing values.
- Comfort manipulating dtypes and filling NaNs appropriately.

---

## Day 4 – Feature Distributions & Relationships (EDA Visuals)

**Objectives**
- Visually explore distributions of features.
- Examine relationships between pairs of variables.

**Tasks**
1. **New notebook**
   - `week2_day4_eda_visuals.ipynb`.
   - Load cleaned version of your dataset (or clean as needed within this notebook).

2. **Distribution plots**
   - Histograms for numeric columns:
     ```python
     import matplotlib.pyplot as plt
     import seaborn as sns

     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
     for col in numeric_cols:
         plt.figure(figsize=(4, 3))
         sns.histplot(df[col].dropna(), kde=True)
         plt.title(col)
         plt.show()
     ```

3. **Boxplots for numeric vs category**
   - If you have a categorical column:
     ```python
     plt.figure(figsize=(6, 4))
     sns.boxplot(data=df, x="some_category_column", y="some_numeric_column")
     plt.xticks(rotation=45)
     plt.show()
     ```

4. **Scatter plots & pairplots**
   - Scatter:
     ```python
     sns.scatterplot(data=df, x="num_col1", y="num_col2")
     plt.show()
     ```
   - Pairplot (small dataset):
     ```python
     sns.pairplot(df[numeric_cols], corner=True)
     plt.show()
     ```

5. **Correlation matrix**
   ```python
   corr = df[numeric_cols].corr()
   plt.figure(figsize=(8, 6))
   sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
   plt.show()
   ```

6. **Mini exercises**
   - Choose:
     - 2–3 numeric features that seem correlated.
     - 1 numeric feature vs 1 categorical feature that looks different across categories.
   - In markdown, write 3–5 bullet points:
     - “Feature A and B appear positively/negatively correlated.”
     - “Category X tends to have higher values of feature Y than category Z.”

**Outcome by end of Day 4**
- Confident in using histograms, boxplots, scatterplots, and correlation heatmaps.
- Starting to “read” data visually and form simple hypotheses.

---

## Day 5 – Train/Test Split & Basic ML Mindset

**Objectives**
- Understand why we split data into train/test sets.
- Perform a train/test split in code (even if we don’t fully train models yet).

**Tasks**
1. **New notebook**
   - `week2_day5_train_test_split.ipynb`.
   - Load your dataset and do minimal cleaning:
     ```python
     import pandas as pd
     from sklearn.model_selection import train_test_split

     df = pd.read_csv("data/your_dataset.csv")
     # apply any simple cleaning from previous days if needed
     ```

2. **Identify target and features**
   - Pick a target column (what you’d like to predict later):
     - Example: price, survival, churn flag, etc.
   ```python
   target_col = "your_target_column"
   X = df.drop(columns=[target_col])
   y = df[target_col]
   ```

3. **Handle obvious non-features**
   - Drop IDs or clearly irrelevant columns:
     ```python
     X = X.drop(columns=["id", "name"], errors="ignore")
     ```

4. **Train/test split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 10 else None
   )

   X_train.shape, X_test.shape
   ```

   - Note: `stratify=y` is useful for classification tasks to keep class balance.

5. **Quick checks**
   ```python
   print("Train target distribution:")
   print(y_train.value_counts(normalize=True) if y_train.nunique() < 20 else y_train.describe())

   print("Test target distribution:")
   print(y_test.value_counts(normalize=True) if y_test.nunique() < 20 else y_test.describe())
   ```

6. **Mini exercises**
   - Try different `test_size` values (e.g., 0.2, 0.3) and see how shapes change.
   - If classification:
     - Compare class proportions in `y_train` vs `y_test` to see if stratification worked.

7. **Short written reflection**
   - Answer in markdown:
     - Why do we need a separate test set?
     - What might go wrong if we trained and evaluated on the same data?

**Outcome by end of Day 5**
- Able to perform a proper train/test split.
- Understand conceptually that test data should not be used during training.

---

## Day 6 – Simple Baseline Models (Preview of Next Week)

**Objectives**
- Use scikit‑learn to train a very simple model as a “baseline”.
- Get a feel for how ML code wraps around the data you prepared.

**Tasks**
1. **New notebook**
   - `week2_day6_baseline_model.ipynb`.
   - Load your dataset, prepare `X`, `y`, do `train_test_split` as on Day 5.

2. **Handle basic preprocessing (just enough to run a model)**
   - For now, keep it simple:
     - Select only numeric columns to avoid dealing with encoding yet:
       ```python
       import numpy as np

       numeric_cols = X_train.select_dtypes(include=[np.number]).columns
       X_train_num = X_train[numeric_cols]
       X_test_num = X_test[numeric_cols]
       ```

3. **Baseline for regression or classification**

   - **If regression (continuous target, e.g., price, amount):**
     ```python
     from sklearn.linear_model import LinearRegression
     from sklearn.metrics import mean_squared_error

     model = LinearRegression()
     model.fit(X_train_num, y_train)

     y_pred = model.predict(X_test_num)
     mse = mean_squared_error(y_test, y_pred)
     print("Test MSE:", mse)
     ```

   - **If classification (discrete target, e.g., 0/1, classes):**
     ```python
     from sklearn.linear_model import LogisticRegression
     from sklearn.metrics import accuracy_score

     model = LogisticRegression(max_iter=1000)
     model.fit(X_train_num, y_train)

     y_pred = model.predict(X_test_num)
     acc = accuracy_score(y_test, y_pred)
     print("Test Accuracy:", acc)
     ```

4. **Compare to simple baseline**
   - Regression:
     ```python
     # Baseline: always predict mean of y_train
     y_mean = y_train.mean()
     y_pred_baseline = [y_mean] * len(y_test)
     mse_baseline = mean_squared_error(y_test, y_pred_baseline)
     print("Baseline MSE:", mse_baseline)
     ```
   - Classification:
     ```python
     # Baseline: always predict most frequent class
     most_common = y_train.value_counts().idxmax()
     y_pred_baseline = [most_common] * len(y_test)
     acc_baseline = accuracy_score(y_test, y_pred_baseline)
     print("Baseline Accuracy:", acc_baseline)
     ```

5. **Mini exercises**
   - Compare model performance vs baseline:
     - Is the ML model meaningfully better?
   - Change:
     - Test size.
     - Random state.
     - See if performance changes slightly.

6. **Short written reflection**
   - In 4–6 sentences:
     - What is your baseline vs model performance?
     - Does your model beat the baseline?
     - What might you do next to improve it (without actually coding it now)?

**Outcome by end of Day 6**
- First end‑to‑end ML experiment, even if very rough.
- Understanding that ML model must be better than naive baseline.

---

## Day 7 – Week 2 Mini Project: EDA + Baseline Notebook

**Objectives**
- Consolidate Week 2 skills into a coherent, well-documented notebook.
- Produce something that looks like the start of a real ML project.

**Tasks**
1. **New notebook**
   - `week2_day7_mini_project.ipynb`.

2. **Structure with markdown headings**
   1. **Title & Objective**
      - Example: “Week 2 Mini Project – Exploring [Dataset Name] and Building a Baseline Model”.
      - 2–3 sentences: describe what the dataset is and what you want to predict.
   2. **Imports & Data Loading**
      ```python
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      import seaborn as sns
      from sklearn.model_selection import train_test_split

      df = pd.read_csv("data/your_dataset.csv")
      ```
   3. **Initial Inspection**
      ```python
      df.head()
      df.info()
      df.describe()
      df.isna().sum()
      ```
   4. **Data Cleaning**
      - Fix dtypes if necessary.
      - Handle missing values (either drop or impute as you decided on Day 3).
   5. **Exploratory Data Analysis**
      - At least:
        - 2 histograms of key numeric variables.
        - 1 boxplot comparing a numeric variable across categories.
        - 1 scatter plot of two numeric variables.
        - 1 correlation heatmap for numeric features.
      - Add markdown cells interpreting key observations.
   6. **Train/Test Split & Baseline Model**
      - Define `X`, `y`.
      - Train/test split.
      - Numeric‑only baseline model (regression or classification as on Day 6).
      - Compare to naive baseline (mean or majority class).
   7. **Conclusions & Next Steps**
      - 5–8 bullet points:
        - What you learned about the data.
        - What your baseline performance is.
        - Which features seem important or promising.
        - What you would do next to improve (feature engineering, better models, etc.).

3. **Polish**
   - Ensure the notebook runs top‑to‑bottom without errors.
   - Use clear variable names and comments.
   - Keep plots readable (labels, titles).

**Outcome by end of Day 7**
- A complete EDA + baseline model notebook.
- You understand the basic ML workflow: clean → explore → split → baseline model.
- You’re ready in Week 3 to go deeper into supervised learning (linear regression, logistic regression) and proper evaluation.

---

If you tell me exactly which dataset you’re using (e.g., Titanic, California Housing, Ames Housing), I can adapt this Week 2 plan with concrete column names and more specific daily exercises.
