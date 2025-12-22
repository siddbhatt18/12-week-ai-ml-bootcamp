## Day 1 (Week 1) – Foundations for Machine Learning: Environment, Python Basics & Mindset

**Objective for today:**  
By the end of Day 1 you will:

1. Have a working Python + Jupyter environment.
2. Understand how to use notebooks productively (not just run cells).
3. Be comfortable with the **core Python building blocks** ML code relies on:
   - Variables & basic types
   - Arithmetic and comparisons
   - Printing and simple debugging
4. Have done a first set of practice questions that start building ML intuition (without yet doing real ML).

Target time: ~2 hours (extendable to 3–4 with the optional/stretch parts).

---

## 1. Setup & Orientation (20–30 minutes)

### 1.1 Install and Test Your Environment

**Goal:** One command opens Jupyter; you can run basic Python code in a notebook.

1. **Install Anaconda (recommended) or Miniconda**  
   - Search: `download anaconda individual edition`  
   - Download and install for your OS.
2. **Create a working folder** for this course:
   - e.g. `~/ml-journey/week1/` (or `C:\Users\you\ml-journey\week1\`).
3. **Launch Jupyter:**
   - Open **Anaconda Navigator → Jupyter Notebook**  
   **or**
   - Open a terminal / Anaconda Prompt:
     ```bash
     cd path/to/ml-journey/week1
     jupyter notebook
     ```
4. Your browser should open Jupyter. If not, copy the URL from the terminal into your browser.

### 1.2 Create Your First Notebook

1. In Jupyter, click **New → Python 3 (ipykernel)**.
2. Save it as: `day1_intro_python.ipynb`.

### 1.3 Notebook Workflow: How to Think While Using It

Before writing any code, set up a structure:

- Add a **Markdown cell** at the top:
  - Title: `# Day 1 – Python Basics`
  - 2–3 bullet points: what you’ll cover (variables, types, arithmetic).
- Remember:
  - Use **Shift + Enter** to run a cell.
  - Use separate cells for separate ideas (easier to debug and review).
  - Add short text (Markdown) headings for sections.

---

## 2. Python Basics for ML (60–90 minutes)

Everything in ML—from data cleaning to training models—is built on fundamental Python constructs. Today you’ll focus on four pillars:

1. Variables & data types
2. Arithmetic & comparisons
3. Simple input/output (printing, formatting)
4. Very light exposure to errors and debugging

### 2.1 Variables & Basic Types

In your `day1_intro_python.ipynb`, create a section:

```markdown
## 1. Variables and Data Types
```

#### Concepts

- **Variable**: name that refers to a value in memory.
- **Types** you’ll use constantly:
  - `int` – integer numbers (e.g., number of samples)
  - `float` – decimal numbers (e.g., loss value, accuracy)
  - `str` – text (e.g., column names, messages)
  - `bool` – `True` / `False` (e.g., comparison results, conditions)

#### Practice

Create a new code cell and run:

```python
age = 25
height = 1.78
name = "Alex"
is_student = True

print(age, type(age))
print(height, type(height))
print(name, type(name))
print(is_student, type(is_student))
```

**What to notice:**

- `type()` shows what “kind” of value you’re working with.
- ML libraries heavily rely on numeric types (`int`, `float`) and arrays of them.

**Mini Task**

1. Create variables that could conceptually come from an ML problem:

   ```python
   n_samples = 1000         # number of rows in a dataset
   n_features = 20          # number of columns (excluding target)
   learning_rate = 0.01     # hyperparameter for gradient-based algorithms
   model_name = "baseline_linear"
   is_trained = False
   ```

2. Print a sentence using these variables. Example:

   ```python
   print(f"Model {model_name} will be trained on {n_samples} samples with {n_features} features.")
   ```

---

### 2.2 Arithmetic & Comparisons

Add a section:

```markdown
## 2. Arithmetic and Comparisons
```

#### Concepts

- Arithmetic operators: `+ - * / // % **`
- Comparisons: `== != > < >= <=`
- In ML, you’ll use these to:
  - Compute error, accuracy, metrics.
  - Compare predicted vs true values.

#### Practice

In a new cell:

```python
# Arithmetic
a = 10
b = 3

print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)   # float division
print("a // b:", a // b) # integer division
print("a % b:", a % b)   # remainder
print("a ** b:", a ** b) # exponentiation

# Comparisons
print("a > b:", a > b)
print("a == b:", a == b)
print("a != b:", a != b)
```

**Mini Task: BMI & Interpretation (baby step toward feature engineering)**

1. Compute a simple BMI (body mass index):

   ```python
   weight_kg = 72
   height_m = 1.78
   bmi = weight_kg / (height_m ** 2)
   print("BMI:", round(bmi, 2))
   ```

2. In a **new code cell**, add a simple comparison:

   ```python
   is_over_25 = bmi > 25
   print("Is BMI over 25?", is_over_25)
   ```

This is the same kind of reasoning you’ll use later when creating features or filters on a dataset.

---

### 2.3 Printing & Simple Debugging Mindset

Add:

```markdown
## 3. Printing and Quick Debugging
```

#### Concepts

- `print()` is your first, simplest debugging tool.
- Use it to:
  - Check intermediate values.
  - Ensure your understanding of what the code is doing matches reality.

#### Practice

Experiment:

```python
x = 5
y = 2 * x + 3
print("x:", x)
print("y:", y)
```

Now deliberately create a mistake and fix it:

```python
x = 5
y = 2 * x + 3

# Incorrect assumption: suppose you *think* y should be 12
# Use a print to check actual values
print("Expected y ~ 12, got:", y)
```

If reality doesn’t match your expectation, you adjust your code or your understanding. This habit becomes critical when debugging ML pipelines.

---

### 2.4 Working with Errors Intentionally (Very Brief)

Add:

```markdown
## 4. Understanding Errors (Don’t Panic)
```

#### Practice

Run a few wrong lines on purpose to see common error messages:

```python
# 1. NameError
print(unknown_variable)

# 2. TypeError
x = "10"
y = 5
print(x + y)  # string + int
```

Read the error messages. You don’t need to memorize them, but get comfortable with the idea that:

- Errors are expected.
- The messages usually tell you:
  - Type of problem
  - Line where it happened
  - A short description

You’ll see a lot of errors once you start combining pandas, NumPy, and scikit-learn, so Day 1 is about lowering your “fear of errors.”

---

## 3. Practice Questions (Increasing Difficulty)

These are designed to:
- Apply today’s Python basics.
- Start building ML-like thinking (working with “data”, “features”, “metrics”).

Do them in new sections of your notebook such as:

```markdown
## Practice Questions
### Level 1 – Fundamentals
...
```

### Level 1 – Fundamentals (Direct Application)

1. **Time Conversion**

   Write code that converts a given number of **seconds** into:
   - Hours
   - Minutes
   - Remaining seconds

   Example: `3665` → `1 hour 1 minute 5 seconds`.

   Hints:
   - Use `//` and `%`.
   - Start from total seconds → hours; then reduce to remaining seconds.

2. **Simple ML-Style Metric: Accuracy (Conceptual)**

   Imagine you have 10 predictions; 1 means “positive”, 0 means “negative”:

   ```python
   y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
   y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
   ```

   - Count how many predictions are correct.
   - Compute accuracy = correct / total.

   You still don’t know ML formally, but accuracy is one of the most basic metrics you’ll use later.

3. **Basic String Formatting**

   You trained a hypothetical model:

   ```python
   model_name = "logistic_regression"
   accuracy = 0.82
   ```

   Print this exactly:

   > Model logistic_regression achieved 82.0% accuracy.

   Use an f-string and `round`.

---

### Level 2 – Light Abstraction & Mini Problems

4. **Feature Scaling Intuition (Without Libraries)**

   Suppose you have a feature representing **ages** and you want to normalize it between 0 and 1 manually (very common ML preprocessing idea).

   Given:

   ```python
   min_age = 18
   max_age = 60
   age = 30
   ```

   Compute a normalized age:

   \[
   \text{normalized} = \frac{\text{age} - \text{min\_age}}{\text{max\_age} - \text{min\_age}}
   \]

   - Implement this formula.
   - Print the normalized value.
   - Then try a different age (e.g., 60 and 18) and interpret the result:
     - What value corresponds to the minimum?
     - What value corresponds to the maximum?

5. **Mini Evaluation Scenario (Closer to ML)**

   Imagine a binary classification model that predicts whether a patient has a disease (1) or not (0):

   ```python
   y_true = [0, 1, 0, 1, 1, 0, 0, 1]
   y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
   ```

   - Count:
     - True positives (pred = 1, true = 1)
     - True negatives (pred = 0, true = 0)
     - False positives (pred = 1, true = 0)
     - False negatives (pred = 0, true = 1)
   - Use only basic comparisons and increments like `count += 1`.

   You’re already touching the core of precision/recall, which you’ll learn formally later.

6. **Interpreting a Simple “Loss”**

   Suppose you have a model whose error on 5 predictions is:

   ```python
   errors = [1.2, -0.5, 0.3, -1.0, 0.0]  # prediction - true
   ```

   - Compute the **Mean Absolute Error (MAE)** manually:
     \[
     \text{MAE} = \frac{1}{n}\sum |error_i|
     \]
   - Use a loop, `abs()`, and `len()`.

   MAE is a real regression metric you’ll implement using libraries in a few weeks; today you just see it as a simple average of distances.

---

### Level 3 – Stretch (Optional but Very Valuable)

7. **Simple “Data Row” Simulation**

   You have “rows” of a tiny dataset represented as dictionaries:

   ```python
   row1 = {"age": 25, "income": 30000, "bought": 0}
   row2 = {"age": 40, "income": 60000, "bought": 1}
   row3 = {"age": 35, "income": 50000, "bought": 0}
   ```

   - Compute the average age of these “customers”.
   - Compute the proportion who bought (`bought == 1`).

   This mimics what you’ll do with pandas DataFrames later, but at a Python-dictionary level.

8. **Synthetic Feature Creation (Feature Engineering Mindset)**

   For each row above, create a new value called `income_per_age`:

   \[
   \text{income\_per\_age} = \frac{\text{income}}{\text{age}}
   \]

   Do this by:
   - Creating new variables:
     ```python
     row1_income_per_age = row1["income"] / row1["age"]
     # etc.
     ```
   - Printing them.

   Simple, but this is **exactly** what feature engineering is: deriving new informative quantities from existing ones.

9. **First “If-Else” Logic Related to ML (Teaser for Tomorrow)**

   You won’t fully study `if` and loops until Day 2, but try this if you feel comfortable:

   - Choose an error value `e`.
   - Write a small piece of code that:
     - Prints `"Good"` if the absolute error is less than 0.5,
     - `"Okay"` if it’s between 0.5 and 1.0,
     - `"Bad"` if it’s greater than or equal to 1.0.

   Use:

   ```python
   e = 0.7  # try different values

   if abs(e) < 0.5:
       print("Good")
   elif abs(e) < 1.0:
       print("Okay")
   else:
       print("Bad")
   ```

   You’ll use this kind of conditional thinking constantly when analyzing or post-processing model outputs.

---

## 4. Reflection & Wrap-Up (10–15 minutes)

Add a final Markdown section:

```markdown
## End-of-Day Reflection
```

Answer these in text (1–3 sentences each):

1. What **three concrete things** can you now do in Python that you couldn’t this morning?
2. Which error messages did you encounter, and what did you learn from them?
3. Which practice question felt:
   - too easy,
   - comfortably challenging,
   - too hard?

This reflection does two things:
- Strengthens memory.
- Gives you a signal for how much to review tomorrow before moving to Day 2 (lists, dictionaries, control flow).

---

## How This Sets You Up for Machine Learning

Today’s work may feel “basic programming” rather than “ML,” but ML code is built from:

- Variables and types → storing parameters, hyperparameters, data sizes.
- Arithmetic and comparisons → computing metrics, losses, updates.
- Printing and debugging → understanding why models behave a certain way.
- Simple “data row” thinking → foundations for pandas, which you’ll use heavily.

If you’d like next, I can:

- Design **Day 2** in the same structured way (lists, dictionaries, control flow, with ML-flavored examples), or  
- Give you a short **self-check quiz** to verify you’ve internalized Day 1 before moving on.
