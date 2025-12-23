## Day 3 (Week 1) – Functions & Problem Decomposition (with ML-Flavored Practice)

**Objective for today:**  
By the end of Day 3, you should be able to:

1. Define and use **functions** to organize your code.
2. Understand **parameters**, **return values**, and **scope**.
3. Break a slightly larger task into small, testable pieces.
4. Implement small, ML-flavored utilities: e.g. computing metrics, normalizing features, summarizing tiny datasets.
5. Begin thinking in **reusable components**, a key skill for ML pipelines and experiments.

Target time: ~2–3 hours (core) + optional stretch.

---

## 0. Quick Warm-Up (5–10 minutes)

Open your Day 2 notebook and:

1. Re-run all cells.
2. In a fresh cell, **from memory**:
   - Create a small list of `y_true`, `y_pred`.
   - Use a `for` loop to compute accuracy.
3. If anything feels shaky, quickly revisit that section.

Now create a new notebook for today:

- Name: `day3_functions_and_decomposition.ipynb`
- Top markdown cell:

```markdown
# Day 3 – Functions & Problem Decomposition

Goals:
- Learn to define and use functions.
- Break down ML-style tasks into smaller utilities.
- Implement simple metric and preprocessing functions.
```

---

## 1. Functions – Your Building Blocks (30–40 minutes)

Functions let you encapsulate logic and reuse it. In ML projects, you’ll use them for:

- Data loading and cleaning.
- Feature engineering.
- Model training and evaluation.

### 1.1 Basic Function Syntax

Add section:

```markdown
## 1. Basics of Functions
```

In a code cell:

```python
def add_numbers(a, b):
    result = a + b
    return result

sum_ab = add_numbers(3, 5)
print("Result:", sum_ab)
```

Key ideas:

- `def` starts a function definition.
- `a` and `b` are **parameters** (inputs).
- `return` sends a value back to the caller.
- Without `return`, the function returns `None`.

### 1.2 Functions with No Return (Side Effects)

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alex")
```

Uses:

- Logging progress.
- Printing intermediate summaries in ML workflows.

### 1.3 Functions with Documentation (Docstrings)

```python
def bmi(weight_kg, height_m):
    """
    Compute Body Mass Index (BMI) given weight (kg) and height (m).
    Returns a float.
    """
    return weight_kg / (height_m ** 2)

print("BMI:", round(bmi(72, 1.78), 2))
```

Docstrings are essential for clarity—especially for ML utility functions you’ll reuse.

---

## 2. Functions in Practice – Small Utilities (30–40 minutes)

Now turn yesterday’s patterns into functions you can call repeatedly.

Add:

```markdown
## 2. Turning Repeated Logic into Functions
```

### 2.1 Even/Odd Checker (Simple Example)

```python
def is_even(n):
    """
    Return True if n is even, False otherwise.
    """
    return n % 2 == 0

print(is_even(4))
print(is_even(7))
```

### 2.2 Basic Stats Function for a List

```python
def basic_stats(numbers):
    """
    Given a list of numbers, return a dictionary with min, max, and average.
    """
    n = len(numbers)
    total = 0
    min_val = numbers[0]
    max_val = numbers[0]

    for x in numbers:
        total += x
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x

    avg = total / n
    return {"min": min_val, "max": max_val, "avg": avg}

stats = basic_stats([3, 7, 10, 2, 9])
print(stats)
```

You just reproduced a core pattern: summarizing numeric features.

### 2.3 Normalizing Values (Feature Scaling)

Add:

```python
def min_max_normalize(x, min_val, max_val):
    """
    Normalize x to [0, 1] using provided min and max.
    Assumes max_val > min_val.
    """
    return (x - min_val) / (max_val - min_val)

print(min_max_normalize(30, 10, 50))  # test
```

You’ll later use this idea with NumPy/pandas for feature scaling.

---

## 3. Problem Decomposition (ML-Flavored Example) (30–40 minutes)

Now practice **breaking down** a slightly larger ML-style task into functions.

Goal: given lists of true labels and predicted labels:

- Compute accuracy.
- Compute confusion matrix counts.

Add:

```markdown
## 3. Problem Decomposition – Metric Utilities
```

### 3.1 Step 1 – Accuracy Function

```python
def accuracy_score(y_true, y_pred):
    """
    Compute accuracy given lists of true and predicted labels (0/1 or classes).
    """
    correct = 0
    total = len(y_true)
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / total
```

Test:

```python
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 1, 1, 0, 0]

print("Accuracy:", accuracy_score(y_true, y_pred))
```

This mirrors `sklearn.metrics.accuracy_score` conceptually.

### 3.2 Step 2 – Confusion Matrix Counts

```python
def confusion_counts(y_true, y_pred):
    """
    Compute TP, TN, FP, FN for binary classification (labels 0/1).
    Returns a dictionary.
    """
    tp = tn = fp = fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

print(confusion_counts(y_true, y_pred))
```

You’ve essentially implemented the logic behind many ML diagnostics.

### 3.3 Step 3 – Precision and Recall from Confusion Counts

```python
def precision_recall(counts):
    """
    Given a confusion counts dict with tp, tn, fp, fn,
    return precision and recall.
    """
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    # Handle potential division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall

counts = confusion_counts(y_true, y_pred)
prec, rec = precision_recall(counts)
print("Precision:", prec)
print("Recall:", rec)
```

You’ve now:

- Decomposed the problem into 3 functions.
- Built reusable building blocks resembling scikit-learn’s metric functions.

This is the same thinking you’ll use to nest preprocessing functions, model training functions, and evaluation in larger projects.

---

## 4. Practice Questions (Increasing Difficulty, ML-Flavored)

Add:

```markdown
## 4. Practice Questions
```

Work through levels in order. Try to write each function from scratch without copy-paste.

---

### Level 1 – Direct Use of Functions

1. **Temperature Converter**

   Write a function:

   ```python
   def celsius_to_fahrenheit(c):
       """
       Convert Celsius to Fahrenheit.
       """
   ```

   - Then convert: `0, 25, 100`.

   This warms up the “input → formula → output” pattern.

2. **Reusing Normalization Logic**

   Use your `min_max_normalize` function:

   - Normalize `age = 30` given `min_age = 18`, `max_age = 60`.
   - Normalize `age = 60` and `age = 18`.
   - Confirm:
     - Min maps to 0.
     - Max maps to 1.

3. **Reusable Accuracy Function**

   Using your `accuracy_score` function:

   - Test it on several different pairs of `y_true`, `y_pred`:
     - Perfect predictions.
     - Half correct.
     - All wrong.

   Confirm the outputs match your expectations.

---

### Level 2 – Small ML-Style Utilities

4. **Normalize a List of Values**

   Write a function:

   ```python
   def normalize_list(values):
       """
       Given a list of numbers, return a new list where each value is normalized
       between 0 and 1 using min-max normalization.
       """
   ```

   Steps inside:

   - Find `min_val` and `max_val` using a loop or built-ins.
   - For each value `v` in `values`, compute normalized value using `min_max_normalize` (or inline formula).
   - Return the new list.

   Test on: `[10, 20, 30, 40]`.

   This is essentially a very primitive version of what scalers (like `MinMaxScaler`) do in scikit-learn.

5. **Dataset Summary Function**

   Reuse the “customers” dataset:

   ```python
   customers = [
       {"age": 25, "income": 30000, "churned": 0},
       {"age": 40, "income": 60000, "churned": 1},
       {"age": 35, "income": 50000, "churned": 0},
       {"age": 50, "income": 80000, "churned": 1},
       {"age": 29, "income": 40000, "churned": 0},
   ]
   ```

   Write a function:

   ```python
   def summarize_customers(customers):
       """
       Given a list of customer dicts, return:
       - average age
       - average income
       - churn rate
       in a dictionary.
       """
   ```

   Inside the function:

   - Loop over customers, accumulate totals, and counts.
   - Return e.g.:

   ```python
   {
       "avg_age": ...,
       "avg_income": ...,
       "churn_rate": ...
   }
   ```

   This resembles group-level summaries you’ll compute with pandas.

6. **Rule-Based Classifier as a Function**

   Define a simple rule:

   - If `income > 50000` and `age > 30` → predict `churned = 1`.
   - Else → predict `churned = 0`.

   Write:

   ```python
   def rule_based_churn_prediction(customer):
       """
       Given a customer dict with age and income,
       return a predicted churn value (0 or 1) based on a simple rule.
       """
   ```

   Then:

   - Loop over the `customers` list.
   - For each customer, get prediction from `rule_based_churn_prediction`.
   - Compare with true `churned`.
   - Use `accuracy_score` to compute the rule’s accuracy.

   This mimics a small but meaningful ML pipeline:
   - Rule → Predictions → Evaluation.

---

### Level 3 – Stretch Challenges (Optional, Strongly Recommended)

7. **Reusable Metric Suite**

   Combine earlier ideas into one utility:

   ```python
   def evaluate_binary_classifier(y_true, y_pred):
       """
       Given true and predicted binary labels:
       - compute confusion counts
       - compute accuracy, precision, and recall
       Return a dictionary with all metrics.
       """
   ```

   Inside:

   - Call `confusion_counts` and `precision_recall`.
   - Compute accuracy via `accuracy_score`.
   - Return e.g.:

   ```python
   {
       "tp": ..., "tn": ..., "fp": ..., "fn": ...,
       "accuracy": ...,
       "precision": ...,
       "recall": ...
   }
   ```

   You have now a compact “evaluation report” function—exactly what you’ll want when comparing models later.

8. **Feature Engineering Function**

   For customers, add a new feature: `income_per_age`.

   Write:

   ```python
   def add_income_per_age(customers):
       """
       Given a list of customer dicts, modify each dict in-place
       to add 'income_per_age' = income / age.
       Return the modified list.
       """
   ```

   - Loop over `customers`.
   - For each, compute `c["income_per_age"] = c["income"] / c["age"]`.
   - Return the list.

   Then:

   - Call `add_income_per_age(customers)`.
   - Print `customers` to check.

   This is exactly what you’ll later do with pandas: adding computed columns.

9. **Mini “Experiment Runner” (Thinking Bigger)**

   Simulate a minimal training/evaluation experiment:

   - You have:
     - `y_true` (ground truth labels).
     - Two different prediction lists: `preds_model_a`, `preds_model_b`.
   - Write a function:

     ```python
     def compare_two_models(y_true, y_pred_a, y_pred_b):
         """
         Compute accuracy for both models and print which one performs better.
         """
     ```

   - Inside:
     - Use `accuracy_score` for each model.
     - Print accuracies and a short message like `"Model A is better"` or `"Model B is better"` or `"They are equal"`.

   This anticipates how you’ll later compare logistic regression vs random forest vs gradient boosting on the same dataset.

---

## 5. Wrap-Up & Reflection (10–15 minutes)

At the bottom of your notebook:

```markdown
## End-of-Day Reflection
```

Answer:

1. What are **two concrete advantages** of using functions in ML projects?
   - (e.g., reusability, easier debugging, testing small units)
2. Which function you wrote today do you think you’ll **reuse or adapt** once you start working with real datasets (e.g., normalization, accuracy, evaluation)?
3. If you had to explain to a friend how to **break down** the problem of “evaluating a classifier,” how would you split it into smaller steps?

---

## How Day 3 Fits into Your ML Journey

You now know how to:

- Package logic into **functions**.
- Decompose problems into small, testable utilities.
- Implement core conceptual pieces of ML evaluation and preprocessing by hand.

Over the coming weeks, you’ll replace your manual functions with:

- `sklearn.metrics.accuracy_score`, `precision_score`, `recall_score`.
- `sklearn.preprocessing.StandardScaler` / `MinMaxScaler`.
- Pipelines that integrate all of this automatically.

But because you’ve built these pieces yourself, you’ll understand what’s happening under the hood—making you more confident and better at debugging.

If you’d like next, I can:

- Design a detailed **Day 4** guide (NumPy basics with ML-flavored array operations), or
- Provide a short **checkpoint quiz** that mixes concepts from Days 1–3.
