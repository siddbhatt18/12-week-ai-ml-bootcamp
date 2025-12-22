Below is your 7‑day plan for **Week 3**, where you move from EDA to actually building your **first machine learning models** using **linear regression** (for numeric prediction) and getting a clear understanding of the **core ML workflow**.

Assume ~1.5–2.5 hours/day. Reuse concepts from Weeks 1–2; don’t worry if you have to look things up, but always try to reason before searching.

---

## Overall Week 3 Goals

By the end of this week, you should be able to:

- Frame a **regression problem** (predicting a number).
- Implement the **core ML workflow**:
  1. Define problem and target.
  2. Split data into train/test.
  3. Train a baseline model.
  4. Evaluate with regression metrics (MAE, MSE, RMSE, R²).
  5. Inspect and iterate.
- Use **scikit‑learn** for basic modeling with **LinearRegression**.
- Do a small **regression mini‑project** end‑to‑end.

---

## Day 1 – Core ML Workflow & Problem Framing

**Objectives:**
- Understand what ML is doing (conceptually).
- Learn the basic **workflow** that you’ll repeat for almost every project.

### 1. Notebook

Create: `week3_day1_ml_workflow.ipynb`.

### 2. Conceptual Overview (Markdown)

In a Markdown cell, write short notes (in your own words) on:

- **Supervised learning**:
  - Input features \(X\), target \(y\).
  - Regression vs classification:
    - Regression: predict a continuous value (price, temperature).
    - Classification: predict a category (spam/not spam).

- **Core ML workflow** (write it out explicitly):
  1. Understand the problem and data.
  2. Choose target \(y\) and features \(X\).
  3. Split data into train and test sets.
  4. Fit a model on the train set.
  5. Evaluate performance on the test set.
  6. Iterate: try better preprocessing, features, models.

You can skim a resource by searching:  
`scikit-learn tutorial supervised learning`  
Read just enough to understand the structure.

### 3. Choose a Regression Dataset

Pick **one** small regression dataset:

Options:
- scikit‑learn’s **California Housing** dataset.
- A Kaggle house prices dataset.

For simplicity, use California Housing from scikit‑learn today:

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame  # pandas DataFrame
df.head()
```

Look at:
- `df.head()`
- `df.info()`
- `df.describe()`

### 4. Identify Features and Target

In Markdown:

- Write which column is the **target** (for California Housing, it’s usually `MedHouseVal` or similar; check `data.feature_names` and `data.target`).
- All other columns are features \(X\).

In code:

```python
X = df.drop(columns=["MedHouseVal"])  # or the actual target name
y = df["MedHouseVal"]
```

(If your dataset uses a different name, adjust accordingly.)

### 5. Visual Check

Quick sanity checks:

- `y.describe()` to see target distribution.
- `X.columns` to see feature list.

You are **not** building a model yet. Today is about understanding the structure and naming \(X, y\) correctly.

### 6. Thinking Challenge

In Markdown, answer:

1. In real life, what kinds of factors would you expect to influence house prices? Which of those appear as columns in your dataset?
2. What would be a **reasonable error range** (e.g., is being off by \$1,000 okay? \$50,000?)  
   You’re building intuition about what “good performance” might mean.

---

## Day 2 – Train/Test Split & Your First Baseline Model

**Objectives:**
- Learn how to split data into train & test.
- Train your **first ML model**: simple linear regression.
- Understand what a “baseline” is.

### 1. Notebook

Create: `week3_day2_first_regression.ipynb`.

Load data as you did on Day 1 (`X`, `y`).

### 2. Train/Test Split

From scikit‑learn:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- Explain in Markdown:
  - Why do we keep a test set?
  - What does `random_state` do?

### 3. Baseline Model (Mean Predictor)

Before using ML, build a trivial baseline: **always predict the mean of the training targets**.

```python
import numpy as np

y_mean = y_train.mean()
y_pred_baseline = np.full_like(y_test, fill_value=y_mean, dtype=float)
```

We’ll evaluate it tomorrow, but just keep `y_pred_baseline` for now.

### 4. First Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

- Print model coefficients and intercept:

```python
model.intercept_, model.coef_
```

**Don’t worry** about deep interpretation yet; just note that each feature gets a weight.

### 5. Predict on Test Set

```python
y_pred_lr = model.predict(X_test)
```

You’ve now done a full fit–predict cycle.

### 6. Thinking Challenge

In Markdown, answer:

1. Conceptually, what is linear regression doing?  
   (Your words, something like: “It finds the line/plane in feature space that best fits the training data by minimizing squared error.”)
2. Why is it important that we **fit on train** and **predict on test**, instead of training on all data?

Don’t compute metrics yet; that’s tomorrow’s focus.

---

## Day 3 – Regression Metrics: MAE, MSE, RMSE, R²

**Objectives:**
- Learn how to evaluate regression models.
- Compare baseline vs linear regression.

### 1. Notebook

Create: `week3_day3_regression_metrics.ipynb`.

Reload or reuse `X_train`, `X_test`, `y_train`, `y_test`, `y_pred_baseline`, `y_pred_lr`.  
(Or just re-run previous code in this notebook.)

### 2. Implement Metrics (By Hand First)

Before using scikit‑learn, implement your own functions:

```python
import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

Test them on simple arrays:

```python
y_true_test = np.array([1, 2, 3])
y_pred_test = np.array([1.1, 1.9, 3.2])
print(mean_absolute_error(y_true_test, y_pred_test))
print(root_mean_squared_error(y_true_test, y_pred_test))
```

### 3. Evaluate Baseline vs Linear Regression

Use your custom metrics:

```python
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = root_mean_squared_error(y_test, y_pred_baseline)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = root_mean_squared_error(y_test, y_pred_lr)

print("Baseline MAE:", mae_baseline)
print("Linear Regression MAE:", mae_lr)
print("Baseline RMSE:", rmse_baseline)
print("Linear Regression RMSE:", rmse_lr)
```

### 4. R² Score (Explained Variance)

Use scikit‑learn’s `r2_score`:

```python
from sklearn.metrics import r2_score

r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression R^2:", r2_lr)
```

Optional: implement R² yourself after reading definition.

### 5. Interpret the Numbers

In Markdown:

- Compare baseline vs linear regression MAE/RMSE.
- Answer:
  - Is the model better than the trivial baseline?
  - How big is the average error relative to typical prices (look at target distribution again)?
  - What does a positive R² mean; what about if it were negative?

### 6. Thinking Challenge

- Suppose your model has an RMSE of 0.5 (if target is in 100k’s of dollars, e.g., 0.5 ≈ \$50,000).  
  Is that acceptable? It depends on context—write 5–6 sentences about **how you would decide** if a model is “good enough” in a real business.

---

## Day 4 – Visualizing Predictions & Error Analysis

**Objectives:**
- See where your model performs well or poorly.
- Start building intuition about **error patterns**.

### 1. Notebook

Create: `week3_day4_error_analysis.ipynb`.

Recompute or load your best linear regression model and predictions.

### 2. Plot Predictions vs True Values

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")  # perfect prediction line
plt.show()
```

Interpret:
- Are points close to the red line?
- Do errors get larger for higher prices?

### 3. Residuals (Errors)

Compute residuals:

```python
residuals = y_test - y_pred_lr
```

Plot histogram:

```python
import seaborn as sns

sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residual (y_true - y_pred)")
plt.title("Residual Distribution")
plt.show()
```

Plot residuals vs predicted values:

```python
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_lr, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()
```

Interpret:
- Are residuals centered around 0?
- Do they form a pattern (e.g., funnel shape)? That might suggest non‑linearity or heteroscedasticity (don’t worry too much about the term yet—just notice patterns).

### 4. Error by Feature

Pick one or two features, e.g., `MedInc` (median income) or a similar column:

```python
df_test = X_test.copy()
df_test["y_true"] = y_test
df_test["y_pred"] = y_pred_lr
df_test["residual"] = df_test["y_true"] - df_test["y_pred"]
```

Now:

```python
sns.scatterplot(data=df_test, x="MedInc", y="residual", alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.show()
```

Interpret:
- Does the model systematically under/over‑predict for low vs high `MedInc`?

### 5. Thinking Challenge

Write a short section in Markdown titled **“Model Limitations I Can See”** and list 5–7 bullet points, such as:

- “Errors are larger for higher‑priced houses.”
- “There seems to be a pattern in residuals vs MedInc, suggesting a non‑linear relationship that linear regression isn’t capturing.”
- Etc.

This is training you to **see** problems, not just numbers.

---

## Day 5 – Feature Scaling, Train/Validation/Test & Over/Underfitting (Conceptual)

**Objectives:**
- Understand why we sometimes need **feature scaling**.
- Learn about train/validation/test splits conceptually.
- Get a basic feel for **overfitting vs underfitting**.

### 1. Notebook

Create: `week3_day5_scaling_splits_overfitting.ipynb`.

### 2. Feature Scaling (Standardization)

Search: `scikit-learn StandardScaler`.

Concept:
- Many models (not always plain linear regression, but especially regularized ones and others) like features on similar scales.
- Standardization: subtract mean, divide by standard deviation.

Steps:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train
X_test_scaled = scaler.transform(X_test)        # transform test
```

Build a new model using `X_train_scaled`:

```python
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
```

Compare RMSE of scaled vs unscaled. In linear regression with these features, they might be similar; the main point is the **process**.

Write in Markdown:
- Why do we **fit** scaler only on the training set?
- What is “data leakage”?

### 3. Train / Validation / Test Split (Conceptual + Simple Code)

Explain in Markdown:

- Why we might want:
  - Train set: to fit models.
  - Validation set: to choose/tune models.
  - Test set: final, untouched performance estimate.

Implement a simple 3‑way split:

```python
X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test2, y_val, y_test2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Result: 70% train, 15% val, 15% test
```

You won’t tune heavily yet, but get used to the idea.

### 4. Overfitting vs Underfitting (Conceptual)

In Markdown, in your own words:

- **Underfitting**: model too simple, can’t capture patterns; poor on train and test.
- **Overfitting**: model too complex, memorizes train; great on train, bad on test.
- **Just right**: good train performance, slightly worse but still good test performance.

Then:

1. Train a model on **very small** training set (e.g., 200 samples) and evaluate.
2. Train another model on a **much larger** training set and compare.  
   Notice how performance changes.

### 5. Thinking Challenge

Imagine you’re given a complex model (e.g., huge neural net) that gets almost zero error on training but very large error on test.

- Write a short paragraph:
  - Why might this be happening?
  - What are 3 strategies you could try to reduce overfitting (without knowing all the technical details yet)?

---

## Day 6 – Small Regression Mini‑Project (Part 1: Setup & Baseline)

**Objectives:**
- Start an **end‑to‑end mini‑project** on regression.
- Practice framing the problem and creating a baseline model.

### 1. Notebook

Create: `week3_regression_mini_project.ipynb`.

### 2. Choose Dataset

You can:
- Continue with California Housing, **or**
- Choose a different regression dataset (e.g., Kaggle “House Prices: Advanced Regression Techniques”).  
  If you choose House Prices, you’ll need a bit more cleaning; that’s okay if you want challenge.

For this week, California Housing is fine if you want to focus on workflow, not heavy cleaning.

### 3. Project Introduction (Markdown)

Write a proper **Introduction** section:

- What is the dataset?
- What target are you predicting?
- Why is this useful in the real world?
- What questions do you want to answer:
  - “How accurate can a simple linear model be?”
  - “Which features seem most important?”

### 4. Data Loading & Basic EDA (Short)

- Load data into `df`.
- Show:
  - `df.head()`
  - `df.info()`
  - `df.describe()`
- Briefly inspect for:
  - NaNs.
  - Obvious anomalies.

You don’t need full Week 2‑style EDA now; just enough to understand the basics.

### 5. Baseline & Linear Regression

- Define `X`, `y`.
- Train/test split.
- Implement:
  - Baseline mean predictor.
  - LinearRegression model.
- Compute MAE, RMSE, R² for both.

Document results in a Markdown section “Baseline Results”.

### 6. Thinking Challenge

Write:
- Is the linear regression giving a **meaningful improvement** over the baseline?
- What are 2–3 things you might try next to improve it?

---

## Day 7 – Small Regression Mini‑Project (Part 2: Improvement & Reflection)

**Objectives:**
- Try simple improvements.
- Practice structured reporting and reflection.

### 1. Continue in `week3_regression_mini_project.ipynb`

Add new sections rather than starting from scratch.

### 2. Try Simple Improvements

Pick 1–3 of the following (don’t try to do everything; focus on depth and understanding):

**a) Feature Scaling**
- Use `StandardScaler` on features.
- Refit linear regression.
- Compare metrics.

**b) Feature Selection**
- Use domain intuition or correlations:
  - Drop features that seem unhelpful.
  - Or try a model with only the 3–5 features you think are most predictive.
- Compare performance vs using all features.

**c) Train/Validation Split for Simple Model Selection**
- Create a small validation set.
- Try:
  - Model A: all features.
  - Model B: subset of features.
- Choose the one with better validation RMSE, then evaluate on test.

### 3. Optional: Interpret Coefficients

For the chosen model, look at:

```python
feature_names = X.columns  # if not scaled, or store names separately
coeffs = model.coef_
for name, coef in zip(feature_names, coeffs):
    print(name, ":", coef)
```

Write:
- Which features have large positive or negative coefficients?
- Does that align with intuition (e.g., higher income → higher house value)?

(If you used scaled features, coefficients are in standardized units—still can give a rough sense of importance.)

### 4. Write a Short Project Report (Markdown)

Structure:

1. **Problem & Data**
   - 3–5 sentences about what you tried to predict and what data you used.

2. **Method**
   - Outline:
     - Features and target.
     - Baseline.
     - Linear regression.
     - Any scaling/feature selection.

3. **Results**
   - Table or bullet list of:
     - Baseline metrics.
     - Final model metrics.
   - Compare and highlight improvements.

4. **Insights & Limitations**
   - 5–10 bullet points:
     - How much better than baseline are you?
     - In what ranges (cheap vs expensive houses) does the model do worse?
     - Which features seem most important?
     - What are key limitations (linearity, missing potential non‑linear interactions, etc.)?

5. **Next Steps**
   - 3–5 bullet points on what you would try next if you had more time:
     - Non‑linear models (e.g., trees, random forests).
     - More feature engineering.
     - Deeper hyperparameter tuning.

### 5. Thinking Challenge

Pretend you’re explaining this project to a non‑technical stakeholder (e.g., a real‑estate manager).  
Write a short **non‑technical summary** (5–8 sentences) that covers:

- What you built.
- How accurate it is (in simple terms, e.g., “On average, we are off by about \$X”).
- What factors matter most.
- Why it’s useful, but also what it **can’t** do well yet.

This is crucial practice for real ML work.

---

## After Week 3: What You Should Be Able to Do

You should now:

- Comfortably:
  - Load data.
  - Define \(X, y\).
  - Split into train/test.
  - Train a simple model.
  - Evaluate it with MAE, RMSE, R².
  - Compare against a baseline.
- Understand, at a conceptual level:
  - Overfitting vs underfitting.
  - Why we split data and sometimes scale features.
- Produce a small **regression project notebook** that reads as a story:  
  Problem → Data → Baseline → Model → Evaluation → Insights.

If you’d like, the next step is a similar 7‑day plan for **Week 4 (classification + logistic regression)**, where you’ll use many of the same ideas but for yes/no or multi‑class labels.
