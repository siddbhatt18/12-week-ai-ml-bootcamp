## Day 2 (Week 1) – Lists, Dictionaries & Control Flow (with ML-Flavored Practice)

**Objective for today:**  
By the end of Day 2, you should be able to:

1. Use **lists** to store and iterate over collections of values.
2. Use **dictionaries** to store structured data (like one row of a dataset).
3. Use **if / elif / else** to make decisions in code.
4. Use **for** loops to process multiple items (e.g., samples, predictions).
5. Apply these to small, ML-flavored tasks: computing basic metrics, summarizing “dataset rows,” and simulating tiny analyses.

Target time: ~2–3 hours (core) + optional stretch.

---

## 0. Quick Warm-Up (5–10 minutes)

Open yesterday’s notebook `day1_intro_python.ipynb` and:

1. Re-run all cells from top to bottom.
2. In a new cell, write from memory:
   - A BMI calculation.
   - A simple accuracy computation for a hard-coded `y_true` and `y_pred`.
3. Fix any errors you encounter, using `print()` if needed.

Then create a **new notebook** for today:

- Name: `day2_collections_control_flow.ipynb`.
- Add a top markdown cell:

```markdown
# Day 2 – Lists, Dictionaries & Control Flow

Goals:
- Understand lists and dictionaries.
- Practice if/elif/else and for loops.
- Apply these to small ML-like problems.
```

---

## 1. Lists – Sequences of Values (30–40 minutes)

Lists represent **ordered collections**. Think:  
- A list of feature values for one sample.  
- A list of predictions.  
- A list of errors.

### 1.1 Creating and Inspecting Lists

Add section:

```markdown
## 1. Lists – Basics
```

In a code cell:

```python
numbers = [3, 7, 10, 2, 9]
print(numbers)
print("Length:", len(numbers))
print("First element:", numbers[0])
print("Last element:", numbers[-1])
```

Concepts:
- `len(list)` gives number of elements.
- Indexing: 0-based (`list[0]` is the first).

### 1.2 Modifying Lists

```python
numbers.append(5)      # add to end
print("After append:", numbers)

last = numbers.pop()   # remove and return last element
print("Popped:", last)
print("After pop:", numbers)

numbers.remove(7)      # remove first occurrence of 7
print("After remove 7:", numbers)
```

### 1.3 Simple Aggregations

```python
numbers = [3, 7, 10, 2, 9]
print("Sum:", sum(numbers))
print("Min:", min(numbers))
print("Max:", max(numbers))
average = sum(numbers) / len(numbers)
print("Average:", average)
```

### 1.4 ML-Flavored Micro-Exercise: List of Errors

Imagine prediction errors:

```python
errors = [0.5, -1.2, 0.3, 0.0, -0.7]

# Compute Mean Absolute Error (MAE) using only built-ins and a loop
total_abs_error = 0
for e in errors:
    total_abs_error += abs(e)

mae = total_abs_error / len(errors)
print("MAE:", mae)
```

You’re already simulating a basic regression metric.

---

## 2. Dictionaries – Structured Records (30–40 minutes)

In ML, a single **sample/row** is often a mapping from feature names to values. Dictionaries are perfect for this.

Add:

```markdown
## 2. Dictionaries – Representing Data Records
```

### 2.1 Basic Dictionary Operations

```python
person = {
    "name": "Alice",
    "age": 25,
    "city": "Paris"
}

print(person)
print("Name:", person["name"])
print("Age:", person["age"])

person["occupation"] = "Data Analyst"  # add
person["age"] = 26                     # update
print("Updated person:", person)

print("Keys:", list(person.keys()))
print("Values:", list(person.values()))
print("Items:", list(person.items()))
```

### 2.2 ML-Style Record Example

Represent a **customer** with features and a label:

```python
customer = {
    "age": 35,
    "income": 55000,
    "country": "US",
    "churned": 0  # 0 = stayed, 1 = churned
}

print(customer["age"], customer["income"], customer["churned"])
```

This is analogous to one row of a dataset in pandas.

### 2.3 List of Dictionaries as a Tiny Dataset

```python
customers = [
    {"age": 25, "income": 30000, "churned": 0},
    {"age": 40, "income": 60000, "churned": 1},
    {"age": 35, "income": 50000, "churned": 0},
    {"age": 50, "income": 80000, "churned": 1},
]
```

We’ll use this in the practice section to simulate dataset operations.

---

## 3. Control Flow – If/Elif/Else & For Loops (40–50 minutes)

Now you’ll combine conditionals and loops with lists/dicts to mimic dataset processing and metric computations.

### 3.1 If / Elif / Else

Add:

```markdown
## 3. Control Flow – Conditions and Loops
```

#### Basics:

```python
grade = 85

if grade >= 90:
    print("A")
elif grade >= 80:
    print("B")
elif grade >= 70:
    print("C")
else:
    print("D")
```

#### ML-Flavored Example: Thresholding a Probability

Imagine a model outputs a probability that a sample is positive:

```python
prob = 0.72
threshold = 0.5

if prob >= threshold:
    prediction = 1
else:
    prediction = 0

print("Probability:", prob)
print("Prediction:", prediction)
```

This is exactly how binary classifiers like logistic regression become 0/1 predictions.

---

### 3.2 For Loops Over Lists

#### Simple Loop:

```python
numbers = [1, 2, 3, 4, 5]
for n in numbers:
    print("Number:", n)
```

#### Computing Average Manually:

```python
numbers = [10, 20, 30]
total = 0
for n in numbers:
    total += n

avg = total / len(numbers)
print("Average:", avg)
```

---

### 3.3 Looping Over a Tiny Dataset

Use the `customers` list from above:

```python
customers = [
    {"age": 25, "income": 30000, "churned": 0},
    {"age": 40, "income": 60000, "churned": 1},
    {"age": 35, "income": 50000, "churned": 0},
    {"age": 50, "income": 80000, "churned": 1},
]
```

Compute the **average income** and **churn rate**:

```python
total_income = 0
churn_count = 0

for c in customers:
    total_income += c["income"]
    if c["churned"] == 1:
        churn_count += 1

avg_income = total_income / len(customers)
churn_rate = churn_count / len(customers)

print("Average income:", avg_income)
print("Churn rate:", churn_rate)
```

You’ve now done manually what pandas `.mean()` and `.value_counts(normalize=True)` will do later.

---

## 4. Practice Questions (Increasing Difficulty, ML-Flavored)

Do these in the same notebook under:

```markdown
## 4. Practice Questions
```

Aim to solve basics first (Level 1), then move up. Use comments and `print()` to clarify your thinking.

---

### Level 1 – Direct Applications

1. **List Aggregation Refresher**

   ```python
   losses = [0.9, 0.7, 0.5, 0.6, 0.4]
   ```

   - Compute min, max, average loss using a loop (don’t use `min` / `max`).
   - Then verify with built-ins: `min(losses)`, `max(losses)`, `sum(losses)/len(losses)`.

2. **Pass/Fail Classification (Using If Statements)**

   Write a function:

   ```python
   def classify_grade(grade):
       # returns "A" if grade >= 90
       # "B" if 80–89
       # "C" if 70–79
       # "D" otherwise
   ```

   - Test it on: `95, 83, 75, 60`.

   This is directly analogous to **bucketing** numeric features into categories.

3. **Converting Probabilities to Predictions**

   Given:

   ```python
   probs = [0.1, 0.6, 0.4, 0.8, 0.52]
   threshold = 0.5
   ```

   - Create a new list `preds` where each element is `1` if the corresponding probability ≥ threshold, else `0`.
   - Print both lists side by side.

---

### Level 2 – Small ML-Like Mini Tasks

4. **Manual Accuracy Computation**

   ```python
   y_true = [1, 0, 1, 1, 0, 1]
   y_pred = [1, 1, 1, 0, 0, 1]
   ```

   - Use a loop to count correct predictions.
   - Compute accuracy = correct / total.
   - Print: `"Accuracy: 0.666..."` (or formatted with `round`).

5. **Counting Confusion Matrix Components (TP, TN, FP, FN)**

   Using the same `y_true` and `y_pred`:

   - Initialize four counters: `tp`, `tn`, `fp`, `fn` to 0.
   - Loop over all pairs `(true, pred)`:
     - If `true == 1` and `pred == 1`: `tp += 1`.
     - If `true == 0` and `pred == 0`: `tn += 1`.
     - If `true == 0` and `pred == 1`: `fp += 1`.
     - If `true == 1` and `pred == 0`: `fn += 1`.
   - Print the four counts.

   This is directly what ML libraries do under the hood to compute metrics.

6. **Dataset Summary with Dictionaries**

   Reuse this dataset:

   ```python
   customers = [
       {"age": 25, "income": 30000, "churned": 0},
       {"age": 40, "income": 60000, "churned": 1},
       {"age": 35, "income": 50000, "churned": 0},
       {"age": 50, "income": 80000, "churned": 1},
       {"age": 29, "income": 40000, "churned": 0},
   ]
   ```

   Tasks:
   - Compute:
     - Average age.
     - Average income.
   - Compute:
     - Churn rate for customers younger than 35.
     - Churn rate for customers aged 35 or older.

   Hints:
   - Use conditionals inside the loop: `if c["age"] < 35: ... else: ...`
   - Track separate counts and churn counts for `< 35` and `>= 35`.

   You’re effectively doing **segmented analysis**, which is a key part of error analysis later.

---

### Level 3 – Stretch Challenges (Optional but Highly Valuable)

7. **Simple Rule-Based Classifier**

   Imagine you want to predict `churned` based on a simple rule:

   - If `income < 50000` → predict `0` (no churn).
   - Else → predict `1` (churn).

   Using the `customers` list above:
   - For each customer, compute a prediction based on this rule.
   - Compare it with the true `churned` value.
   - Compute accuracy of this rule-based “model”.

   You just built a *very simple* model using if/else logic. This mirrors how decision trees conceptually work.

8. **Creating a New Feature: Income per Age**

   For each customer, compute:

   \[
   \text{income\_per\_age} = \frac{\text{income}}{\text{age}}
   \]

   - Loop over `customers`.
   - For each dictionary, add a new key `"income_per_age"` with the computed value.
   - Print the modified `customers` list.

   This is **feature engineering** in its simplest form.

9. **Manual Precision & Recall (Bonus)**

   Using your `tp`, `fp`, `fn` from exercise 5:

   - Compute:
     - Precision = `tp / (tp + fp)` (handle division by zero if no positives predicted).
     - Recall = `tp / (tp + fn)` (handle if no positive examples).
   - Print both values.

   You’ve now manually computed two of the core ML evaluation metrics that you’ll later use from scikit-learn.

---

## 5. Wrap-Up & Reflection (10–15 minutes)

At the bottom of your notebook, add:

```markdown
## End-of-Day Reflection
```

Answer briefly:

1. How comfortable do you now feel with:
   - Lists?
   - Dictionaries?
   - If/else?
   - For loops?

2. Which ML-flavored exercise helped you **see how basic Python connects** to real ML tasks?

3. Is there any snippet of code from today that you would struggle to write from scratch?  
   If yes, copy that task into a fresh cell and re-implement it **without** looking back.

---

## How Day 2 Sets Up Your ML Journey

You’ve now learned:

- **Lists** → how predictions, labels, errors, and samples are often represented at a basic level.
- **Dictionaries** → how a single sample’s features and label can be structured.
- **Control flow (if/loops)** → how to:
  - Turn probabilities into class predictions.
  - Compute evaluation metrics manually.
  - Slice and summarize small “datasets” by conditions.

These are exactly the mental tools you’ll use with pandas and scikit-learn, just abstracted and automated. Tomorrow (Day 3), we’ll wrap these ideas into **functions** and start thinking in reusable building blocks—very similar to how you’ll later design ML pipelines.

If you’d like next, I can:

- Create a detailed **Day 3** guide (functions, problem decomposition, more ML-style practice), or
- Give you a short **self-check quiz** to verify mastery of Day 1–2 concepts before moving on.
