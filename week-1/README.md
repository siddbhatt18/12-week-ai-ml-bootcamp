Below is a structured 7-day plan for Week 1, focused on Python basics, NumPy, pandas, and Jupyter. You’ll get concrete tasks, links/keywords to search for (so you practice finding resources), and small “stretch” challenges to push your thinking.

Assume ~1.5–2.5 hours per day. If you have more time, deepen the exercises rather than rushing ahead.

---

## Day 1 – Environment Setup & First Steps in Python

**Objectives:**
- Set up Python, Jupyter, and a clean working folder.
- Run your first Python code and understand how notebooks work.

### 1. Install and Set Up

1. Install **Anaconda** (or Miniconda + Jupyter).
   - Search: `download anaconda individual edition`  
   - Follow official instructions for your OS.
2. Create a dedicated folder, e.g., `ml-learning/week1/`.
3. Launch:
   - Either **Anaconda Navigator → Jupyter Notebook**  
   - Or in terminal: `jupyter notebook` from your `ml-learning/week1/` folder.

### 2. Create Your First Notebook

1. In Jupyter, click **New → Python 3 (ipykernel)**.
2. Name it: `day1_intro_python.ipynb`.

### 3. Basic Python Practice (In Notebook)

In `day1_intro_python.ipynb`, create separate cells for each part and run them.

**Topics:**
- Variables and basic types (`int`, `float`, `str`, `bool`)
- Arithmetic (`+ - * / // % **`)
- Printing and f-strings
- Simple built-in functions: `len`, `type`, `print`, `round`

**Exercises:**

1. **Variables & types**
   - Create variables:
     ```python
     age = 25
     height = 1.75
     name = "Alice"
     is_student = True
     ```
   - Use `type()` to check each variable’s type.
   - Print a sentence like:  
     `"My name is Alice, I am 25 years old, and my height is 1.75 meters."`

2. **Simple math**
   - Compute:
     - Your BMI (weight / height²; pick any weight).
     - The area and perimeter of a rectangle (given width and height).
   - Use `round(value, 2)` to round results.

3. **Mini challenge (thinking)**  
   Write code that:
   - Takes a number of minutes (e.g., `125`) and calculates:
     - Hours (integer)
     - Remaining minutes  
   For example, 125 minutes → 2 hours 5 minutes.  
   Use integer division `//` and modulo `%`.  
   Don’t copy-paste; think it through:
   - How many full hours fit into `total_minutes`?
   - What’s left after those hours?

---

## Day 2 – Lists, Dictionaries & Basic Control Flow

**Objectives:**
- Learn how to store collections of data.
- Use `if` statements and loops.

### 1. Create a New Notebook

- Name it: `day2_collections_control_flow.ipynb`.

### 2. Lists and Dictionaries

Search: `python lists tutorial`, `python dictionaries tutorial`.

**Practice:**

1. **Lists**
   - Create a list of 5 numbers, e.g.:
     ```python
     numbers = [3, 7, 10, 2, 9]
     ```
   - Access elements (`numbers[0]`, `numbers[-1]`).
   - Append and remove elements (`append`, `pop`, `remove`).
   - Compute:
     - Sum (with `sum(numbers)`).
     - Length (`len(numbers)`).
     - Average.

2. **Dictionaries**
   - Create a dictionary:
     ```python
     person = {
         "name": "Alice",
         "age": 25,
         "city": "Paris"
     }
     ```
   - Access values (`person["name"]`).
   - Add a new key `occupation`.
   - Update `age` to a new value.

3. **Mini challenge (thinking)**  
   You have a list of dictionaries representing students:
   ```python
   students = [
       {"name": "Alice", "grade": 85},
       {"name": "Bob", "grade": 72},
       {"name": "Charlie", "grade": 90}
   ]
   ```
   - Compute the **average grade**.
   - Print: `"Alice passed"` if grade ≥ 80, else `"Alice failed"` for each student.
   - Think: how do you iterate over a list? How do you access the dictionary fields inside?

### 3. If Conditions and Loops

**Practice:**

1. Write a function `classify_grade(grade)` that:
   - Returns `"A"` if grade ≥ 90  
   - `"B"` if 80–89  
   - `"C"` if 70–79  
   - `"D"` otherwise
2. Test it on several values.
3. Use a `for` loop to apply `classify_grade` to all students’ grades and print `"Alice: B"`, `"Bob: C"`, etc.

---

## Day 3 – Functions, Writing Clean Code & Problem Breakdown

**Objectives:**
- Learn to define and call functions.
- Practice clear thinking and breaking problems into steps.

### 1. New Notebook

- Name: `day3_functions_and_thinking.ipynb`.

### 2. Functions

Search: `python define function`, `python *args **kwargs` (skim).

**Practice:**

1. Basic function:
   ```python
   def add_numbers(a, b):
       return a + b
   ```
   - Call `add_numbers(3, 5)`.

2. Slightly more complex:
   ```python
   def bmi(weight_kg, height_m):
       return weight_kg / (height_m ** 2)
   ```
   - Print `"Your BMI is X"` using the function.

3. Write:
   ```python
   def is_even(n):
       # returns True if n is even, False otherwise
   ```
   - Test on several numbers.

### 3. Break a Problem Into Steps

**Exercise:**  
Write a function that, given a list of numbers, returns:
- The minimum
- The maximum
- The average

Name it: `stats(numbers)` → returns a dictionary:
```python
{"min": ..., "max": ..., "avg": ...}
```

**Steps:**
1. Describe in plain English what you need to do.
2. Implement using built-ins (`min`, `max`, `sum`, `len`).
3. Then implement again **without** `min` and `max`:
   - Loop and track current `min_val`, `max_val`.

**Mini challenge (thinking)**  
Write `def normalize(numbers):` that returns a new list where each value is scaled between 0 and 1 using:
\[
x_{normalized} = \frac{x - \text{min}}{\text{max} - \text{min}}
\]
- Use your `stats` function to get min/max.
- Consider the edge case when all numbers are the same (what happens if max = min?).

---

## Day 4 – Intro to NumPy: Arrays & Vectorized Operations

**Objectives:**
- Understand what `NumPy` arrays are and why they’re used in ML.
- Learn basic array creation, indexing, and arithmetic.

### 1. Install/Import NumPy

- In a new notebook: `day4_numpy_basics.ipynb`.
- At the top:
  ```python
  import numpy as np
  ```

Search: `numpy tutorial w3schools` or `numpy quickstart tutorial`.

### 2. Basic Arrays

**Practice:**

1. Create arrays:
   ```python
   a = np.array([1, 2, 3, 4])
   b = np.array([10, 20, 30, 40])
   ```
   - Print `a`, `a.shape`, `a.dtype`.

2. Element-wise operations:
   - `a + b`, `a * b`, `a + 10`, `a ** 2`.
   - Compare with doing the same via Python lists (you’ll see the difference).

3. 2D arrays (matrices):
   ```python
   m = np.array([[1, 2, 3],
                 [4, 5, 6]])
   ```
   - Print `m.shape`.
   - Index element `5` (think about `m[row_index, col_index]`).
   - Extract second row, second column.

4. Useful constructors:
   - `np.zeros((3, 3))`, `np.ones((2, 4))`, `np.arange(0, 10, 2)`, `np.linspace(0, 1, 5)`.

### 3. Mini Challenges

1. **Vectorized operations vs loops**  
   Given a Python list of 1 million numbers (e.g., using `range`), try:
   - Compute sum using a loop.
   - Convert to `np.array` and compute sum with `np.sum()`.  
   Note the difference in simplicity (and, if you can, performance).

2. **Mean and standard deviation**  
   - Use `np.mean`, `np.std` on a random array (`np.random.randn(1000)`).
   - Think: in ML, we often normalize features using mean and std.

3. **Thinking challenge**  
   Create a 2D array of shape (5, 5) with numbers from 0 to 24:
   ```python
   arr = np.arange(25).reshape(5, 5)
   ```
   - Extract the center 3x3 block using slicing.
   - Zero out the diagonal elements (set them to 0).  
   Hint: indices where row index == column index.

---

## Day 5 – Intro to pandas: Loading and Inspecting Data

**Objectives:**
- Learn what pandas `DataFrame`s are.
- Load a CSV dataset and inspect its structure.

### 1. New Notebook

- Name: `day5_pandas_intro.ipynb`.
- At top:
  ```python
  import pandas as pd
  ```

Search: `kaggle titanic data download` and create a free Kaggle account if you don’t have one.

### 2. Load a Real Dataset (Titanic Recommended)

1. Download Titanic train CSV from Kaggle (or another small dataset).
2. Place `train.csv` in `ml-learning/week1/data/` (create the `data` folder).
3. In notebook:
   ```python
   df = pd.read_csv("data/train.csv")
   ```

### 3. Basic DataFrame Exploration

**Practice:**

1. Inspect:
   - `df.head()`
   - `df.tail()`
   - `df.shape`
   - `df.columns`
   - `df.info()`
   - `df.describe()` (numerical columns)

2. Access columns:
   - `df["Age"]`, `df["Sex"]`
   - `df[["Age", "Fare"]]`

3. Basic questions:
   - How many rows and columns?
   - What is the average age of passengers? (`df["Age"].mean()`)
   - What is the median fare? (`df["Fare"].median()`)

4. Missing values:
   - `df.isna().sum()`  
   Which columns have missing data?

### 4. Mini Challenges (thinking)

1. Compute:
   - The proportion of passengers who survived:  
     `df["Survived"].mean()`
   - The average age of survivors vs non-survivors:  
     Use `df.groupby("Survived")["Age"].mean()`

2. Think:
   - Does age seem to be related to survival in this simple aggregation?
   - What other columns might be important (e.g., `Sex`, `Pclass`)?

---

## Day 6 – Filtering, Sorting & Basic Plotting

**Objectives:**
- Learn how to select subsets of rows based on conditions.
- Sort data and create simple visualizations.

### 1. New Notebook (or continue Day 5)

- Name: `day6_pandas_filter_plot.ipynb` (or continue in previous).

### 2. Filtering and Sorting

**Practice:**

1. Filtering with conditions:
   - Passengers with age < 18:
     ```python
     kids = df[df["Age"] < 18]
     ```
   - Female passengers in 1st class:
     ```python
     female_first = df[(df["Sex"] == "female") & (df["Pclass"] == 1)]
     ```

2. Sorting:
   - Sort by age:
     ```python
     df.sort_values("Age", ascending=True).head()
     ```
   - Sort by fare descending:
     ```python
     df.sort_values("Fare", ascending=False).head()
     ```

3. Value counts:
   - `df["Sex"].value_counts()`
   - `df["Pclass"].value_counts(normalize=True)` (proportions)

### 3. Basic Plotting (matplotlib/seaborn)

Install/Import:
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

Search: `seaborn countplot`, `seaborn histplot`.

**Practice:**

1. Histogram of age:
   ```python
   sns.histplot(data=df, x="Age", bins=20)
   plt.show()
   ```

2. Survival rate by sex:
   ```python
   sns.countplot(data=df, x="Sex", hue="Survived")
   plt.show()
   ```

3. Boxplot of fare by class:
   ```python
   sns.boxplot(data=df, x="Pclass", y="Fare")
   plt.show()
   ```

### 4. Mini Challenges (thinking)

1. Plot survival vs passenger class in a way that lets you see differences clearly.  
   You might:
   - Use `sns.countplot(x="Pclass", hue="Survived", data=df)`.
   - Or compute survival rate per class and then plot a bar chart.

2. Ask and try to answer 2–3 questions via plots:
   - Did younger passengers tend to survive more?
   - How does fare relate to survival?

Write short text cells with your interpretations.

---

## Day 7 – Week 1 Consolidation Mini-Project

**Objectives:**
- Combine Python + NumPy + pandas + plotting.
- Practice structuring a small “analysis” from start to conclusion.

### 1. New Notebook

- Name: `week1_mini_project_titanic_eda.ipynb`.

### 2. Mini-Project Outline

Your goal: perform a simple exploratory data analysis (EDA) on the Titanic dataset and write a short, coherent story.

**Sections to Include (Use Markdown cells to title them):**

1. **Introduction**
   - 2–3 sentences:  
     - What is this dataset?  
     - What is your goal this week? (e.g., “Understand basic structure and relationships, especially with survival.”)

2. **Data Loading**
   - Load CSV with pandas.
   - Show `df.head()`, `df.info()` briefly.

3. **Data Overview**
   - Summaries:
     - `df.describe()`.
     - Missing values (`df.isna().sum()`).

4. **Univariate Analysis**
   - Distributions of key variables:
     - Age, Fare.
     - `Sex`, `Pclass`, `Embarked` (value counts / bar plots).

5. **Bivariate Analysis with Survival**
   - Survival by:
     - Sex (plot + survival rate numbers).
     - Pclass (plot + survival rate numbers).
     - Possibly Age (e.g., compare age distributions for survivors vs non-survivors).

6. **Simple Numeric Summary**
   - Use `groupby`:
     - `df.groupby("Survived")["Age"].mean()`
     - `df.groupby("Survived")["Fare"].mean()`
     - `df.groupby(["Pclass", "Survived"])["PassengerId"].count()` (for counts).

7. **Conclusions & Next Steps**
   - 5–10 bullet points summarizing what you observed:
     - “Females appear to survive at a much higher rate than males.”
     - “Passengers in 1st class have higher survival rates than those in 3rd class.”
   - One or two thoughts on what features might be useful if you wanted to build a survival prediction model later (foreshadowing ML).

### 3. Thinking Challenges for the Mini-Project

- Avoid copy-pasting everything from Day 5/6; try to re-implement from memory. Use search only when needed.
- When you plot something, always ask:
  - What question does this plot answer?
  - What did I learn from it?
- Try at least one thing you haven’t done before this week (e.g., a `pairplot` on a subset of numeric columns).

Example:
```python
sns.pairplot(df[["Age", "Fare", "Pclass", "Survived"]], hue="Survived")
plt.show()
```
(If this is too slow or messy, pick fewer columns.)

---

## How to Know You’re Ready for Week 2

By the end of Day 7, you should be able to:

- Write simple Python functions and loops without constantly looking things up.
- Use NumPy arrays for basic numeric operations.
- Load a CSV into pandas, explore columns and basic statistics.
- Filter, sort, and aggregate data using pandas.
- Create basic plots and interpret them in plain language.
- Produce a small, self-contained notebook that reads like a story: intro → analysis → insights.

If any of these feel very shaky, spend an extra session redoing or slightly modifying your Day 7 mini-project (for example, switch to a different dataset and repeat the process). Repetition with slight variation builds both confidence and real skill.

If you’d like, next I can:
- Turn Week 2 into a similar day-by-day plan, focusing more deeply on EDA and visualization, or
- Give you a set of “check-yourself” questions/exercises to test your Week 1 mastery.
