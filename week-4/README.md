Below is your 7‑day plan for **Week 4**, where you move from regression to **classification**, focusing on **logistic regression** and core classification metrics. You’ll mainly work with the **Titanic dataset** (binary: survived vs not) so you can reuse your Week 2 EDA work.

Assume ~1.5–2.5 hours/day. As before, always try to reason through problems *before* searching or copying code.

---

## Overall Week 4 Goals

By the end of this week, you should be able to:

- Frame and implement a **binary classification** problem.
- Prepare data: handle missing values, encode categorical features.
- Use **logistic regression** in scikit‑learn.
- Evaluate with classification metrics:
  - Accuracy, precision, recall, F1, confusion matrix, ROC AUC.
- Understand **class imbalance** and decision thresholds.
- Build a small **classification mini‑project** end‑to‑end.

---

## Day 1 – Regression vs Classification & Framing Titanic as ML Problem

**Objectives:**
- Understand how classification differs from regression.
- Clearly define Titanic survival as a supervised learning task.

### 1. Notebook

Create: `week4_day1_classification_intro.ipynb`.

Load Titanic data (`train.csv`) into `df` (same as Week 2).

### 2. Conceptual Overview (Markdown)

Write (in your own words):

1. **Regression vs classification**:
   - Regression: predict continuous value (price, temperature).
   - Classification: predict class label (yes/no, cat/dog).
2. **Binary classification**:
   - Target \(y\) has only two classes (e.g., 0/1, survived/not).
   - Logistic regression is a linear model that outputs a **probability** between 0 and 1 for the positive class.

3. **Titanic as ML problem**:
   - Input features \(X\): info about passenger (sex, class, age, fare, etc.).
   - Target \(y\): Survived (0 = no, 1 = yes).

### 3. Inspect the Target

```python
df["Survived"].value_counts()
df["Survived"].value_counts(normalize=True)
```

In Markdown:
- What fraction survived?
- Is one class more common than the other?

### 4. Identify Potential Features

In Markdown, list which columns you think are useful for predicting survival and which you want to ignore (e.g., `PassengerId`, `Name`, `Ticket` may be less directly useful initially).

Suggested starting features:

- Categorical: `Sex`, `Pclass`, `Embarked`.
- Numeric: `Age`, `Fare`, `SibSp`, `Parch`.

You don’t need to be perfect; you’re practicing reasoning.

### 5. Thinking Challenge

Write down:

1. Three **real-world questions** you’d like your classifier to help answer (e.g., “Given a passenger’s info, what’s the chance they survive?”).
2. One potential **ethical concern** with predictive models in such a context (e.g., using personal characteristics to decide resource allocation).

---

## Day 2 – Data Cleaning & Feature Preparation for Classification

**Objectives:**
- Prepare Titanic features \(X\) and target \(y\).
- Handle missing values and encode categorical features.

### 1. Notebook

Create: `week4_day2_titanic_preprocessing.ipynb`.

Load Titanic: `df = pd.read_csv("data/train.csv")`.

### 2. Basic Cleaning

Check:
```python
df.isna().sum()
```

Handle missing values **simply but explicitly**:

- Age: fill with median (or group-specific median if you like challenge).
- Embarked: fill with most common value.
- Cabin: either drop the column entirely or create a “HasCabin” flag if you want extra.

Example:
```python
df["Age"].fillna(df["Age"].median(), inplace=True)

most_common_embarked = df["Embarked"].mode()[0]
df["Embarked"].fillna(most_common_embarked, inplace=True)
```

Decide and document your choices in Markdown.

### 3. Create Features and Target

Set:
```python
target = "Survived"

features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
X = df[features].copy()
y = df[target]
```

### 4. Encode Categorical Features

For now, use pandas’ `get_dummies`:

```python
X_encoded = pd.get_dummies(X, columns=["Sex", "Embarked", "Pclass"], drop_first=True)
X_encoded.head()
```

In Markdown, explain:

- Why we need to encode categorical variables for ML models.
- What `drop_first=True` roughly does (it avoids redundant dummy columns).

### 5. Sanity Check

- Check `X_encoded.info()` and ensure all features are numeric.
- Confirm **no missing values** remain in `X_encoded`:
  ```python
  X_encoded.isna().sum().sum()
  ```

### 6. Thinking Challenge

- Which of these preprocessing steps could cause **data leakage** if applied incorrectly when we introduce train/test splits?  
  Hint: think about computing medians/means on the full dataset vs training set only.

Write a short explanation.

---

## Day 3 – Train/Test Split & First Logistic Regression Model

**Objectives:**
- Perform train/test split for classification.
- Train your first logistic regression model.
- Get basic predictions.

### 1. Notebook

Create: `week4_day3_first_logistic_regression.ipynb`.

Load and preprocess data exactly as Day 2 (or import from a helper module if you made one).

### 2. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
```

Note `stratify=y`:
- In Markdown: explain why stratifying by target can be helpful (preserves class balance in train and test).

### 3. Baseline Classifier

Build a naive baseline: always predict the **most frequent class** in the training set.

```python
most_common_class = y_train.mode()[0]
import numpy as np
y_pred_baseline = np.full_like(y_test, fill_value=most_common_class)
```

You’ll evaluate this tomorrow.

### 4. Train Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
```

Then:

```python
y_pred_logreg = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # probability of class 1 (survived)
```

### 5. Quick Accuracy Check

Just to verify things work:

```python
from sklearn.metrics import accuracy_score

print("Baseline accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Logistic regression accuracy:", accuracy_score(y_test, y_pred_logreg))
```

Don’t stop at accuracy—we’ll go deeper tomorrow.

### 6. Thinking Challenge

In Markdown:

1. Why might **accuracy alone** be misleading? Give a hypothetical example (e.g., disease detection with 99% healthy, 1% sick).
2. What is the difference between:
   - `predict` (returns 0/1 labels)
   - `predict_proba` (returns probabilities)

---

## Day 4 – Classification Metrics: Confusion Matrix, Precision, Recall, F1, ROC AUC

**Objectives:**
- Properly evaluate the classifier beyond accuracy.
- Understand what makes a classification model “good” in different contexts.

### 1. Notebook

Create: `week4_day4_classification_metrics.ipynb`.

Recreate or load `y_test`, `y_pred_baseline`, `y_pred_logreg`, `y_pred_proba`.

### 2. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm_baseline = confusion_matrix(y_test, y_pred_baseline)
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

print("Baseline confusion matrix:\n", cm_baseline)
print("Logistic regression confusion matrix:\n", cm_logreg)
```

Use `ConfusionMatrixDisplay` for nicer plots if you like:

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay(cm_logreg).plot()
plt.show()
```

In Markdown, label the cells:

- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- True Positive (TP)

And interpret for logistic regression:
- How many survivors did we correctly identify?
- How many did we miss (FN)?

### 3. Precision, Recall, F1

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_baseline = precision_score(y_test, y_pred_baseline)
recall_baseline = recall_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline)

precision_lr = precision_score(y_test, y_pred_logreg)
recall_lr = recall_score(y_test, y_pred_logreg)
f1_lr = f1_score(y_test, y_pred_logreg)

print("Baseline - Precision:", precision_baseline,
      "Recall:", recall_baseline, "F1:", f1_baseline)
print("LogReg   - Precision:", precision_lr,
      "Recall:", recall_lr, "F1:", f1_lr)
```

In Markdown:

- Explain precision in plain language.
- Explain recall in plain language.
- Explain F1 as the harmonic mean of precision and recall (why might that be useful?).

### 4. ROC Curve & AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_lr = roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label=f"LogReg (AUC = {auc_lr:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

In Markdown:

- What does a random classifier’s ROC curve look like?
- What does AUC ≈ 0.5 mean vs AUC close to 1?

### 5. Thinking Challenge

Imagine you are designing a **lifeboat allocation** system where:
- **False negatives** (predicting “will not survive” when they actually would) might be worse than some false positives.

1. Should you prefer **higher recall** or **higher precision** for the positive (survived) class?
2. How would your metric choice (precision, recall, F1) reflect that?

Write 5–8 sentences justifying your answers.

---

## Day 5 – Decision Thresholds and Class Imbalance

**Objectives:**
- Understand that 0.5 threshold is arbitrary and can be adjusted.
- See how this affects precision and recall.
- Introduce concept of class imbalance more concretely.

### 1. Notebook

Create: `week4_day5_thresholds_imbalance.ipynb`.

Load `y_test` and `y_pred_proba` from logistic regression.

### 2. Vary Decision Threshold

By default, logistic regression uses threshold 0.5:
```python
y_pred_default = (y_pred_proba >= 0.5).astype(int)
```

Now vary:

```python
import numpy as np
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
from sklearn.metrics import precision_score, recall_score, f1_score

for thr in thresholds:
    y_pred_thr = (y_pred_proba >= thr).astype(int)
    p = precision_score(y_test, y_pred_thr)
    r = recall_score(y_test, y_pred_thr)
    f1 = f1_score(y_test, y_pred_thr)
    print(f"Threshold {thr:.2f} -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
```

In Markdown:
- Describe the trade‑off you observe:
  - Lower threshold → more positives predicted → higher recall, lower precision.
  - Higher threshold → fewer positives → higher precision, lower recall.

### 3. Plot Precision–Recall Curve (Optional but Useful)

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
```

Interpret:
- Why is this especially relevant when positive class is rare?

### 4. Class Imbalance Discussion

Revisit:
```python
df["Survived"].value_counts(normalize=True)
```

Discuss in Markdown:

- Is there significant imbalance? (Probably not extreme, but survival < non‑survival.)
- Why can imbalance make accuracy misleading?
- Why might you prefer precision/recall/F1 or ROC AUC in certain contexts?

### 5. Thinking Challenge

Create a small **thought experiment**:

- Compare two models:
  - Model A: High accuracy but low recall on survivors.
  - Model B: Slightly lower accuracy but much higher recall on survivors.
- In the Titanic context, which would you choose and why?  
  Consider human impact, not just numbers.

Write a 6–10 sentence answer.

---

## Day 6 – Titanic Classification Mini‑Project (Part 1: Setup & Baseline)

**Objectives:**
- Start an end‑to‑end classification project on Titanic.
- Structure your notebook like a small report.
- Get baseline logistic regression performance.

### 1. Notebook

Create: `week4_titanic_classification_project.ipynb`.

### 2. Project Introduction (Markdown)

Include:

- Short description of Titanic dataset.
- Problem statement: predict `Survived` from passenger info.
- Real‑world analog: using limited information to estimate survival chances.
- Goals:
  - Build a baseline logistic regression model.
  - Evaluate with accuracy, precision, recall, F1, ROC AUC.
  - Understand which factors seem most important.

### 3. Data Loading & Preprocessing

Sections:

**a) Load data**

```python
df = pd.read_csv("data/train.csv")
```

**b) Brief EDA recap**

- Show `df.head()` and `df.info()`.

**c) Preprocessing**

- Handle missing values (Age, Embarked, maybe drop Cabin).
- Select features (like in Day 2).
- Encode categoricals with `pd.get_dummies`.
- Confirm no missing values in `X_encoded`.

Explain each step briefly in Markdown: *what* you did and *why*.

### 4. Train/Test Split and Baseline

- Split:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded, y, test_size=0.2, random_state=42, stratify=y
  )
  ```

- Baseline (most frequent class):
  ```python
  most_common_class = y_train.mode()[0]
  y_pred_baseline = np.full_like(y_test, fill_value=most_common_class)
  ```

### 5. Logistic Regression Model

Train & evaluate:

```python
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]
```

Compute and store:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC

Present them in a small table or clean printout.

### 6. Thinking Challenge

Write preliminary conclusions:

- How much better is logistic regression than baseline?
- Is your recall on survivors satisfactory?
- Which metric do you think is most important for this problem and why?

---

## Day 7 – Titanic Classification Mini‑Project (Part 2: Improvement, Interpretation & Reflection)

**Objectives:**
- Try simple improvements.
- Interpret model behavior.
- Write a concise, structured project report.

### 1. Continue in `week4_titanic_classification_project.ipynb`

Add new sections below previous ones.

### 2. Simple Improvements

Choose 1–3 ideas to try (don’t do everything; be deliberate):

**Idea A – Feature Engineering**

- Create `FamilySize = SibSp + Parch + 1`.
- Maybe create an `IsAlone` flag or simple `Child` flag (Age < 16).
- Rebuild `X` including these new features.
- Retrain and reevaluate logistic regression.

**Idea B – Feature Scaling**

For numeric features, scale them:

```python
from sklearn.preprocessing import StandardScaler

num_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]  # adjust if needed
X_train_num = X_train[num_cols]
X_test_num = X_test[num_cols]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train_num)
X_test_scaled[num_cols] = scaler.transform(X_test_num)
```

Retrain on `X_train_scaled`, evaluate on `X_test_scaled`.

**Idea C – Threshold Tuning**

Choose threshold based on your priorities (e.g., higher recall).

- Try 2–3 thresholds (e.g., 0.4, 0.5, 0.6).
- Compare precision, recall, F1.
- Pick one that best matches your problem goals.

### 3. Coefficients Interpretation

Get coefficients:

```python
feature_names = X_train_scaled.columns  # or X_train if not scaled
coeffs = log_reg.coef_[0]

for name, coef in sorted(zip(feature_names, coeffs), key=lambda x: -abs(x[1])):
    print(name, ":", coef)
```

In Markdown:

- Which features have the largest (absolute) coefficients?
- Interpret a few in plain language:
  - Example: “Being male is associated with a lower probability of survival.”
- Note: If you used dummy variables, interpret those carefully (compared to the reference category).

### 4. Final Project Report Structure

In Markdown, add a top‑level **Report** section summarizing:

1. **Problem & Data**
   - Brief explanation of Titanic dataset and goal.

2. **Methods**
   - Features used, preprocessing choices.
   - Logistic regression as the main model.
   - Any feature engineering or scaling.

3. **Results**
   - Baseline vs final model metrics.
   - Confusion matrix and at least one of: ROC AUC, precision/recall/F1 breakdown.
   - Mention of tuned threshold if used.

4. **Insights**
   - Which features contributed most to survival predictions?
   - Any interesting or surprising patterns (e.g., large family size may hurt survival).
   - Limitations of logistic regression (e.g., cannot capture complex non‑linear relationships).

5. **Next Steps**
   - 3–5 bullets: what you would try next (e.g., decision trees, random forests, cross‑validation, more powerful feature engineering).

### 5. Thinking Challenge: Non‑Technical Summary

Write a **non‑technical summary** (as if for a historian or non‑technical stakeholder):

- 6–10 sentences, no math or code.
- Explain:
  - What you predicted.
  - How accurate it is (in intuitive terms: e.g., “We correctly identify about X% of survivors.”).
  - What factors matter most.
  - Limitations and uncertainties.

This mirrors how real ML work is communicated.

---

## After Week 4: What You Should Be Able to Do

You should now be comfortable with:

- Framing and implementing a **binary classification** problem.
- Preprocessing tabular data: missing values, categorical encoding, basic feature engineering.
- Training and evaluating **logistic regression**.
- Interpreting:
  - Confusion matrices.
  - Precision, recall, F1.
  - ROC curves and AUC.
- Thinking critically about metrics and thresholds based on the problem context.

If you’d like, the next week’s plan can focus on **regularization, model comparison (logistic vs trees/forests), and cross‑validation** to deepen your understanding of overfitting and generalization.
