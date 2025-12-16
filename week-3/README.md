**Week 3 Goal:**  
Understand and implement core supervised learning with **linear regression (regression tasks)** and **logistic regression (classification tasks)**, plus basic evaluation metrics.

Assumption: ~1.5–2 hours/day. Use:
- One **regression dataset** (e.g., California Housing, Ames Housing, House Prices Kaggle).
- One **classification dataset** (e.g., Titanic, Breast Cancer).

---

## Day 1 – Conceptual Foundations: Supervised Learning & Linear Regression

**Objectives**
- Understand supervised learning: features → target.
- Build intuition for linear regression and loss functions.

**Tasks**

1. **High-level concepts (reading/thinking, ~30–40 min)**
   - In a markdown cell of `week3_day1_concepts.ipynb`, summarize in your own words:
     - What is **supervised learning**?
     - Difference between:
       - **Features (X)**: inputs.
       - **Target (y)**: what we predict.
     - Two main problem types:
       - **Regression** (continuous y).
       - **Classification** (discrete classes).

2. **Linear Regression intuition**
   - Write (in markdown, short bullets):
     - The model form:  
       \[
       \hat{y} = w_0 + w_1 x_1 + \dots + w_n x_n
       \]
     - Interpretation:
       - Each feature has a **weight** (coefficient).
       - Model predicts a value by summing weighted features + bias.
   - Look up “mean squared error” (MSE) and note:
     - MSE measures how far predictions are from actual values.
     - Model training tries to **minimize MSE**.

3. **Toy example in code**
   - New notebook: `week3_day1_toy_linear_regression.ipynb`.
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   # Synthetic data: y = 3x + 5 + noise
   np.random.seed(42)
   X = 2 * np.random.rand(100, 1)
   y = 3 * X[:, 0] + 5 + np.random.randn(100)

   plt.scatter(X, y)
   plt.xlabel("X")
   plt.ylabel("y")
   plt.title("Synthetic linear data")
   plt.show()
   ```

4. **Fit linear regression with scikit-learn**
   ```python
   from sklearn.linear_model import LinearRegression

   model = LinearRegression()
   model.fit(X, y)

   print("Intercept (w0):", model.intercept_)
   print("Coefficient (w1):", model.coef_)

   X_new = np.array([[0], [2]])
   y_pred_line = model.predict(X_new)

   plt.scatter(X, y, alpha=0.5)
   plt.plot(X_new, y_pred_line, "r-", linewidth=2)
   plt.xlabel("X")
   plt.ylabel("y")
   plt.title("Fitted line")
   plt.show()
   ```

5. **Mini reflection (markdown)**
   - How close are the learned parameters to the true ones (3 and 5)?
   - What does the red line represent?

**Outcome Day 1**
- You can explain linear regression conceptually and see it fit a simple line to data.

---

## Day 2 – Real Regression Task: Linear Regression on a Real Dataset

**Objectives**
- Apply linear regression to a real regression dataset.
- Use basic performance metrics: MSE, RMSE, R².

**Tasks**

1. **New notebook**
   - `week3_day2_real_regression.ipynb`.

2. **Choose and load a regression dataset**
   - Option A: Use a built-in dataset (California Housing):
     ```python
     from sklearn.datasets import fetch_california_housing
     import pandas as pd

     data = fetch_california_housing(as_frame=True)
     df = data.frame
     df.head()
     ```
   - Option B: Load your own CSV with a continuous target (e.g., house prices).

3. **Define features and target**
   - Suppose target column is `"MedHouseVal"` for California Housing:
     ```python
     target_col = "MedHouseVal"
     X = df.drop(columns=[target_col])
     y = df[target_col]
     ```

4. **Train/test split**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

5. **Train linear regression**
   ```python
   from sklearn.linear_model import LinearRegression

   model = LinearRegression()
   model.fit(X_train, y_train)

   print("Intercept:", model.intercept_)
   print("Coefficients:", model.coef_)
   print("Feature names:", X_train.columns.tolist())
   ```

6. **Evaluate with metrics**
   ```python
   from sklearn.metrics import mean_squared_error, r2_score
   import numpy as np

   y_pred_train = model.predict(X_train)
   y_pred_test = model.predict(X_test)

   mse_train = mean_squared_error(y_train, y_pred_train)
   mse_test = mean_squared_error(y_test, y_pred_test)

   rmse_train = np.sqrt(mse_train)
   rmse_test = np.sqrt(mse_test)

   r2_train = r2_score(y_train, y_pred_train)
   r2_test = r2_score(y_test, y_pred_test)

   print("Train RMSE:", rmse_train, "R²:", r2_train)
   print("Test  RMSE:", rmse_test, "R²:", r2_test)
   ```

7. **Mini exercises**
   - Compare train vs test:
     - Is test performance close to train? If not, what might that indicate (over/underfitting)?
   - Print top 3 features with largest absolute coefficient values:
     ```python
     coefs = pd.Series(model.coef_, index=X_train.columns)
     print(coefs.sort_values(key=abs, ascending=False).head(3))
     ```

**Outcome Day 2**
- You can train and evaluate a linear regression model on real data and interpret RMSE/R².

---

## Day 3 – Overfitting, Underfitting & Regularization (Intro)

**Objectives**
- Understand overfitting vs underfitting in regression.
- See how **Ridge** and **Lasso** regularization work in practice.

**Tasks**

1. **Concepts in markdown (short)**
   - In `week3_day3_regularization.ipynb`, define:
     - Underfitting: model too simple, high error on train and test.
     - Overfitting: model too complex, low train error but high test error.
     - Regularization: penalizing large weights to control complexity.

2. **Ridge & Lasso overview**
   - Ridge (L2): penalize sum of squared coefficients.
   - Lasso (L1): penalize sum of absolute coefficients, can drive some to 0 (feature selection).

3. **Apply Ridge/Lasso on your regression dataset**
   ```python
   from sklearn.linear_model import Ridge, Lasso
   from sklearn.metrics import mean_squared_error, r2_score
   import numpy as np
   import pandas as pd

   # Assuming X_train, X_test, y_train, y_test already prepared (from Day 2)
   alphas = [0.01, 0.1, 1.0, 10.0]

   results = []

   for alpha in alphas:
       ridge = Ridge(alpha=alpha)
       ridge.fit(X_train, y_train)
       y_pred_test = ridge.predict(X_test)
       rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
       r2_test = r2_score(y_test, y_pred_test)
       results.append(("Ridge", alpha, rmse_test, r2_test))

       lasso = Lasso(alpha=alpha, max_iter=10000)
       lasso.fit(X_train, y_train)
       y_pred_test = lasso.predict(X_test)
       rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
       r2_test = r2_score(y_test, y_pred_test)
       results.append(("Lasso", alpha, rmse_test, r2_test))

   pd.DataFrame(results, columns=["Model", "alpha", "RMSE_test", "R2_test"])
   ```

4. **Coefficient inspection for Lasso**
   ```python
   best_lasso = Lasso(alpha=0.1, max_iter=10000)  # pick one alpha
   best_lasso.fit(X_train, y_train)
   coefs = pd.Series(best_lasso.coef_, index=X_train.columns)
   print(coefs.sort_values())
   print("Number of non-zero coefficients:", (coefs != 0).sum())
   ```

5. **Mini reflection**
   - Do some alphas improve test RMSE vs plain LinearRegression?
   - How does increasing alpha affect:
     - Performance?
     - Coefficients (magnitude and sparsity)?

**Outcome Day 3**
- Intuition for how regularization prevents overfitting and can simplify models.

---

## Day 4 – Logistic Regression: Concepts & Toy Example

**Objectives**
- Understand logistic regression as a classification model.
- Practice with a simple toy binary classification dataset.

**Tasks**

1. **Concepts in markdown**
   - In `week3_day4_logistic_concepts.ipynb`, briefly define:
     - Logistic regression predicts **probability** of class 1.
     - Uses the logistic (sigmoid) function:
       \[
       \sigma(z) = \frac{1}{1 + e^{-z}}
       \]
     - Decision threshold: default 0.5 (if P(y=1) > 0.5 → predict 1).

2. **Toy dataset with sklearn**
   ```python
   from sklearn.datasets import make_classification
   import matplotlib.pyplot as plt
   import seaborn as sns
   import numpy as np

   X, y = make_classification(
       n_samples=300, n_features=2, n_redundant=0, n_clusters_per_class=1,
       class_sep=1.5, random_state=42
   )

   plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.7)
   plt.xlabel("Feature 1")
   plt.ylabel("Feature 2")
   plt.title("Toy binary classification dataset")
   plt.show()
   ```

3. **Fit logistic regression**
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   log_reg = LogisticRegression()
   log_reg.fit(X_train, y_train)

   y_pred = log_reg.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   print("Test accuracy:", acc)
   ```

4. **Probabilities and decision boundary (optional but useful)**
   ```python
   # predict_proba: probability for each class
   y_proba = log_reg.predict_proba(X_test[:5])
   print("Predicted probabilities for first 5:", y_proba)

   # (Optional) visualize decision boundary
   x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))
   Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)

   plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
   plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", alpha=0.7)
   plt.title("Logistic Regression Decision Boundary")
   plt.show()
   ```

5. **Mini reflection**
   - What does `predict_proba` tell you that `predict` does not?
   - How would changing the classification threshold (0.5) affect accuracy vs other metrics?

**Outcome Day 4**
- You understand logistic regression’s role and can train it on a simple 2D dataset.

---

## Day 5 – Real Binary Classification: Logistic Regression on a Real Dataset

**Objectives**
- Apply logistic regression to a real classification dataset.
- Use basic classification metrics: accuracy, confusion matrix, precision, recall, F1.

**Tasks**

1. **New notebook**
   - `week3_day5_real_classification.ipynb`.

2. **Choose a dataset**
   - Option A: Built-in Breast Cancer dataset:
     ```python
     from sklearn.datasets import load_breast_cancer
     import pandas as pd

     data = load_breast_cancer(as_frame=True)
     df = data.frame  # includes features + target
     df.head()

     X = df.drop(columns=["target"])
     y = df["target"]
     ```
   - Option B: Titanic dataset (Kaggle). Preprocess minimally:
     - Encode “Sex”, drop highly missing columns, etc. (if you’re not comfortable yet, use Breast Cancer to avoid feature engineering overload).

3. **Train/test split**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

4. **Train logistic regression**
   ```python
   from sklearn.linear_model import LogisticRegression

   log_reg = LogisticRegression(max_iter=1000)
   log_reg.fit(X_train, y_train)
   ```

5. **Evaluate performance**
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

   y_pred = log_reg.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   cm = confusion_matrix(y_test, y_pred)

   print("Accuracy:", acc)
   print("Confusion matrix:\n", cm)
   print("Classification report:\n", classification_report(y_test, y_pred))
   ```

6. **Mini exercises**
   - Interpret confusion matrix:
     - What are true positives, false positives, false negatives, true negatives?
   - From classification report, note:
     - Precision and recall for each class.
     - Which is more important in your problem (e.g., missing a cancer case vs false alarm)?

**Outcome Day 5**
- You can train logistic regression on a real dataset and interpret accuracy, precision, recall, and F1.

---

## Day 6 – Comparing Linear Models & Feature Importance Insight

**Objectives**
- Compare different linear/regularized models.
- Build intuition about feature importance through coefficients.

**Tasks**

1. **Regression comparison**
   - In `week3_day6_model_comparison.ipynb`, reuse your regression dataset.
   ```python
   from sklearn.linear_model import LinearRegression, Ridge, Lasso
   from sklearn.metrics import mean_squared_error, r2_score
   import numpy as np
   import pandas as pd
   ```

   - Train all three:
     ```python
     models = {
         "Linear": LinearRegression(),
         "Ridge": Ridge(alpha=1.0),
         "Lasso": Lasso(alpha=0.1, max_iter=10000),
     }

     results = []

     for name, m in models.items():
         m.fit(X_train, y_train)
         y_pred = m.predict(X_test)
         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
         r2 = r2_score(y_test, y_pred)
         results.append((name, rmse, r2))

     pd.DataFrame(results, columns=["Model", "RMSE_test", "R2_test"])
     ```

2. **Feature importance via coefficients (regression)**
   ```python
   lin = LinearRegression()
   lin.fit(X_train, y_train)
   coefs_lin = pd.Series(lin.coef_, index=X_train.columns).sort_values(key=abs, ascending=False)
   print("Top Linear Regression coefficients:\n", coefs_lin.head(5))
   ```

   - For Lasso:
     ```python
     lasso = Lasso(alpha=0.1, max_iter=10000)
     lasso.fit(X_train, y_train)
     coefs_lasso = pd.Series(lasso.coef_, index=X_train.columns).sort_values()
     print("Non-zero Lasso coefficients:", (coefs_lasso != 0).sum())
     ```

3. **Classification feature importance (logistic regression)**
   - On your classification dataset:
     ```python
     import numpy as np

     log_reg = LogisticRegression(max_iter=1000)
     log_reg.fit(X_train, y_train)

     coef_series = pd.Series(log_reg.coef_[0], index=X_train.columns)
     print(coef_series.sort_values(key=abs, ascending=False).head(10))
     ```

4. **Mini reflection (markdown)**
   - Which features have the strongest positive correlation with the target?
   - Which features have strongest negative correlation?
   - How might this match domain intuition (e.g., higher “mean radius” increases likelihood of cancer)?

**Outcome Day 6**
- Ability to compare linear models and read coefficient-based feature importance.

---

## Day 7 – Week 3 Mini Project: Regression + Classification Baselines

**Objectives**
- Combine everything from Week 3 into two clean, end-to-end mini pipelines:
  - One **regression** task (linear + regularized comparison).
  - One **classification** task (logistic regression).

**Tasks**

1. **New notebook**
   - `week3_day7_mini_project.ipynb`.

2. **Structure notebook with sections**  
   Use markdown headings:

   ### 1. Overview
   - Briefly describe:
     - Regression problem: which dataset and target?
     - Classification problem: dataset and target?

   ### 2. Regression Task

   1. **Data loading & brief EDA**
      - Load regression dataset.
      - Show `head()`, `info()`, `describe()` for main numeric features.
   2. **Train/test split**
      ```python
      from sklearn.model_selection import train_test_split

      X = df.drop(columns=[reg_target_col])
      y = df[reg_target_col]

      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, random_state=42
      )
      ```
   3. **Models**
      - Fit:
        - LinearRegression
        - Ridge
        - Lasso
      - Compare RMSE and R² in a small table.
   4. **Interpretation**
      - Show top 5 absolute coefficients for one model.
      - 3–5 bullet points summarizing which features matter most.

   ### 3. Classification Task

   1. **Data loading & brief EDA**
      - Load classification dataset.
      - `head()`, `info()`, some `value_counts()` for target.
   2. **Train/test split**
      ```python
      X = df.drop(columns=[clf_target_col])
      y = df[clf_target_col]

      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, random_state=42, stratify=y
      )
      ```
   3. **Logistic regression**
      ```python
      from sklearn.linear_model import LogisticRegression
      from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

      log_reg = LogisticRegression(max_iter=1000)
      log_reg.fit(X_train, y_train)

      y_pred = log_reg.predict(X_test)
      print("Accuracy:", accuracy_score(y_test, y_pred))
      print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
      print("Classification report:\n", classification_report(y_test, y_pred))
      ```
   4. **Interpretation**
      - Display top 5 coefficients in absolute value.
      - Write 3–5 bullet points:
        - What the model is good/bad at (e.g., high recall for positive class?).
        - Which features most strongly influence predictions.

   ### 4. Summary & Next Steps
   - 6–8 bullet points combining lessons from both tasks:
     - What you learned about regression vs classification.
     - When linear models might be too simple.
     - What you would do next to improve (better preprocessing, non-linear models, etc.)

3. **Polish**
   - Ensure notebook runs from top to bottom.
   - Keep code cells short and well commented.
   - Use clear section titles.

**Outcome Day 7**
- Two complete baseline ML workflows using linear and logistic regression.
- Solid foundational understanding of core supervised learning, ready for Week 4 (more classification algorithms and evaluation).
