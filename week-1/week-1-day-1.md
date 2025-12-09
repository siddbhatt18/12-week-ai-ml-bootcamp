# Week 1, Day 1: NumPy Arrays & Operations - Deep Dive

## Learning Objectives
By the end of today, you will:
- Understand what NumPy is and why it's essential for Machine Learning
- Create and manipulate NumPy arrays
- Perform basic mathematical operations on arrays
- Understand the concept of vectorization
- Build intuition for how data is represented in ML

---

## Session Structure (2-3 hours)

### Part 1: Theory & Context (30 minutes)
### Part 2: Hands-On Practice (90 minutes)
### Part 3: Challenge Exercise (30-60 minutes)

---

## PART 1: THEORY & CONTEXT (30 minutes)

### Why NumPy Matters for Machine Learning

**The Problem:**
Imagine you have 1 million data points and need to multiply each by 2. Using regular Python:

```python
# Slow Python way
regular_list = [1, 2, 3, 4, 5]
result = []
for num in regular_list:
    result.append(num * 2)
```

This works, but for ML with millions of data points, it's **painfully slow**.

**The NumPy Solution:**
```python
import numpy as np
array = np.array([1, 2, 3, 4, 5])
result = array * 2  # All elements multiplied at once!
```

**Speed difference:** NumPy is typically 10-100x faster! 🚀

### Why is NumPy faster?

1. **Written in C:** Low-level, compiled code
2. **Vectorization:** Operations on entire arrays at once
3. **Contiguous memory:** Data stored efficiently
4. **Optimized algorithms:** Uses CPU efficiently

### Mental Model for ML

Think of data in ML as tables (spreadsheets):
- **Each row** = one example/sample (e.g., one house, one customer)
- **Each column** = one feature (e.g., price, age, size)

NumPy arrays are how we represent these tables in code!

---

## PART 2: HANDS-ON PRACTICE (90 minutes)

### Setup (5 minutes)

Open Google Colab (colab.research.google.com) or Jupyter Notebook and create a new notebook.

```python
import numpy as np

# Check your NumPy version
print(f"NumPy version: {np.__version__}")

# This should work without errors
print("✓ NumPy is ready!")
```

---

### Exercise Block 1: Creating Arrays (20 minutes)

#### 1.1 Basic Array Creation

```python
# From Python lists
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)

print("Original list:", list_1d)
print("NumPy array:", array_1d)
print("Type:", type(array_1d))
```

**👉 YOUR TURN:** Create a NumPy array containing the numbers 10 through 20.

<details>
<summary>💡 Solution (try first!)</summary>

```python
my_array = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# OR
my_array = np.array(list(range(10, 21)))
print(my_array)
```
</details>

---

#### 1.2 Multi-Dimensional Arrays (THE CORE CONCEPT)

```python
# 2D array (like a spreadsheet/table)
array_2d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("2D Array:")
print(array_2d)
print(f"\nShape: {array_2d.shape}")  # (rows, columns)
print(f"Dimensions: {array_2d.ndim}")
print(f"Total elements: {array_2d.size}")
```

**🧠 UNDERSTAND THIS:** 
- **Shape (3, 3)** = 3 rows, 3 columns
- In ML: 3 samples (e.g., 3 houses), 3 features each (e.g., size, price, age)

**👉 YOUR TURN:** Create a 2D array representing 4 students with 3 test scores each:
- Student 1: [85, 90, 78]
- Student 2: [92, 88, 84]
- Student 3: [78, 85, 80]
- Student 4: [95, 92, 96]

Then print the shape and total number of scores.

<details>
<summary>💡 Solution</summary>

```python
test_scores = np.array([
    [85, 90, 78],
    [92, 88, 84],
    [78, 85, 80],
    [95, 92, 96]
])

print("Test Scores:")
print(test_scores)
print(f"Shape: {test_scores.shape}")  # Should be (4, 3)
print(f"Total scores: {test_scores.size}")  # Should be 12
```
</details>

---

#### 1.3 Convenient Array Creation Functions

```python
# Array of zeros (useful for initialization)
zeros = np.zeros((3, 4))  # 3 rows, 4 columns
print("Zeros:\n", zeros)

# Array of ones
ones = np.ones((2, 5))
print("\nOnes:\n", ones)

# Range of values
range_array = np.arange(0, 10, 2)  # start, stop, step
print("\nRange (0-10, step 2):", range_array)

# Evenly spaced values
linspace_array = np.linspace(0, 1, 5)  # start, stop, number of points
print("\nLinspace (0-1, 5 points):", linspace_array)

# Random values (VERY common in ML)
random_array = np.random.rand(3, 3)  # uniform distribution [0, 1)
print("\nRandom:\n", random_array)

# Random integers
random_ints = np.random.randint(1, 100, size=(2, 3))  # low, high, size
print("\nRandom integers:\n", random_ints)
```

**👉 YOUR TURN:** Create:
1. An array of 10 zeros
2. An array with values from 5 to 50 with step of 5
3. A 5x5 array of random numbers

<details>
<summary>💡 Solution</summary>

```python
# 1. Ten zeros
ten_zeros = np.zeros(10)
print("Ten zeros:", ten_zeros)

# 2. 5 to 50, step 5
fives = np.arange(5, 51, 5)
print("Fives:", fives)

# 3. 5x5 random
random_5x5 = np.random.rand(5, 5)
print("5x5 Random:\n", random_5x5)
```
</details>

---

### Exercise Block 2: Array Indexing & Slicing (25 minutes)

#### 2.1 1D Array Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single element
print("First element:", arr[0])
print("Last element:", arr[-1])

# Slicing [start:stop:step]
print("First three:", arr[0:3])
print("Last two:", arr[-2:])
print("Every other element:", arr[::2])
```

**👉 YOUR TURN:** Given the array `np.array([5, 10, 15, 20, 25, 30, 35, 40])`:
1. Get elements at positions 2, 3, and 4
2. Get every third element
3. Reverse the array

<details>
<summary>💡 Solution</summary>

```python
arr = np.array([5, 10, 15, 20, 25, 30, 35, 40])

# 1. Positions 2, 3, 4 (remember: 0-indexed!)
print("Positions 2-4:", arr[2:5])

# 2. Every third element
print("Every third:", arr[::3])

# 3. Reverse
print("Reversed:", arr[::-1])
```
</details>

---

#### 2.2 2D Array Indexing (CRITICAL FOR ML)

```python
# Sample data: 3 students, 4 subjects
grades = np.array([
    [85, 92, 78, 88],  # Student 0
    [90, 85, 92, 95],  # Student 1
    [78, 88, 85, 82]   # Student 2
])

# Single element: [row, column]
print("Student 0, Subject 0:", grades[0, 0])  # 85
print("Student 1, Subject 2:", grades[1, 2])  # 92

# Entire row (all subjects for one student)
print("\nAll grades for Student 1:", grades[1, :])

# Entire column (one subject for all students)
print("\nSubject 2 for all students:", grades[:, 2])

# Subarray
print("\nFirst 2 students, first 2 subjects:")
print(grades[0:2, 0:2])
```

**🧠 ML CONTEXT:** 
- `grades[:, 0]` gets Feature 1 for all samples
- `grades[0, :]` gets all features for Sample 1

**👉 YOUR TURN:** Create a 4x3 array representing 4 houses with 3 features (square_feet, bedrooms, price):

```python
houses = np.array([
    [1500, 3, 300000],
    [2000, 4, 450000],
    [1200, 2, 250000],
    [1800, 3, 380000]
])
```

Extract:
1. All prices (last column)
2. Features for the second house
3. Square footage and bedrooms (first 2 columns) for all houses

<details>
<summary>💡 Solution</summary>

```python
houses = np.array([
    [1500, 3, 300000],
    [2000, 4, 450000],
    [1200, 2, 250000],
    [1800, 3, 380000]
])

# 1. All prices
prices = houses[:, 2]
print("All prices:", prices)

# 2. Second house features
house_2 = houses[1, :]
print("House 2:", house_2)

# 3. Sqft and bedrooms for all
sqft_beds = houses[:, 0:2]
print("Square footage and bedrooms:\n", sqft_beds)
```
</details>

---

### Exercise Block 3: Array Operations (25 minutes)

#### 3.1 Arithmetic Operations (Vectorization!)

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

# Element-wise operations
print("Addition:", arr1 + arr2)
print("Subtraction:", arr2 - arr1)
print("Multiplication:", arr1 * arr2)
print("Division:", arr2 / arr1)
print("Power:", arr1 ** 2)

# Operations with scalars (broadcasting)
print("\nMultiply by 10:", arr1 * 10)
print("Add 100:", arr1 + 100)
```

**🚀 THIS IS THE MAGIC:** No loops needed! All elements processed simultaneously.

**👉 YOUR TURN:** 
You have temperatures in Fahrenheit: `[32, 68, 86, 104, 122]`
Convert them to Celsius using the formula: C = (F - 32) * 5/9

<details>
<summary>💡 Solution</summary>

```python
fahrenheit = np.array([32, 68, 86, 104, 122])
celsius = (fahrenheit - 32) * 5/9
print("Celsius:", celsius)
# Should output: [ 0. 20. 30. 40. 50.]
```
</details>

---

#### 3.2 Aggregation Functions (Statistical Operations)

```python
data = np.array([10, 20, 30, 40, 50])

print("Sum:", data.sum())
print("Mean:", data.mean())
print("Standard Deviation:", data.std())
print("Min:", data.min())
print("Max:", data.max())
print("Index of max:", data.argmax())
```

For 2D arrays - specify axis!

```python
grades = np.array([
    [85, 92, 78],
    [90, 85, 92],
    [78, 88, 85]
])

print("Overall mean:", grades.mean())
print("Mean per student (axis=1):", grades.mean(axis=1))
print("Mean per subject (axis=0):", grades.mean(axis=0))
```

**🧠 AXIS UNDERSTANDING:**
- **axis=0**: down the rows (column-wise operation)
- **axis=1**: across the columns (row-wise operation)

**👉 YOUR TURN:** 
Given sales data for 3 stores over 4 days:
```python
sales = np.array([
    [150, 200, 180, 220],  # Store 1
    [165, 190, 175, 210],  # Store 2
    [140, 185, 195, 205]   # Store 3
])
```

Calculate:
1. Total sales for each store
2. Average daily sales across all stores
3. Best performing day (highest total sales)

<details>
<summary>💡 Solution</summary>

```python
sales = np.array([
    [150, 200, 180, 220],
    [165, 190, 175, 210],
    [140, 185, 195, 205]
])

# 1. Total per store
store_totals = sales.sum(axis=1)
print("Total per store:", store_totals)

# 2. Average daily sales across all stores
daily_averages = sales.mean(axis=0)
print("Daily averages:", daily_averages)

# 3. Best performing day
daily_totals = sales.sum(axis=0)
best_day = daily_totals.argmax()
print(f"Best day: Day {best_day + 1} with ${daily_totals[best_day]}")
```
</details>

---

#### 3.3 Boolean Indexing (Super Powerful!)

```python
ages = np.array([25, 30, 35, 40, 45, 50])

# Create boolean mask
mask = ages > 35
print("Mask:", mask)

# Use mask to filter
filtered_ages = ages[mask]
print("Ages > 35:", filtered_ages)

# One-liner
print("Ages > 35 (one-liner):", ages[ages > 35])
```

**👉 YOUR TURN:**
Given test scores: `[45, 67, 82, 91, 58, 73, 88, 95, 62, 77]`
1. Find all passing scores (>= 70)
2. Count how many students passed
3. Find scores in the "B" range (80-89)

<details>
<summary>💡 Solution</summary>

```python
scores = np.array([45, 67, 82, 91, 58, 73, 88, 95, 62, 77])

# 1. Passing scores
passing = scores[scores >= 70]
print("Passing scores:", passing)

# 2. Count
num_passing = (scores >= 70).sum()  # True = 1, False = 0
print("Number passing:", num_passing)

# 3. B range (80-89)
b_grades = scores[(scores >= 80) & (scores < 90)]
print("B grades:", b_grades)
```
</details>

---

### Exercise Block 4: Array Reshaping (20 minutes)

```python
# Original array
arr = np.arange(12)  # [0, 1, 2, ..., 11]
print("Original:", arr)

# Reshape to 2D
reshaped = arr.reshape(3, 4)  # 3 rows, 4 columns
print("\n3x4:\n", reshaped)

# Reshape to 3D
reshaped_3d = arr.reshape(2, 2, 3)
print("\n2x2x3:\n", reshaped_3d)

# Flatten back to 1D
flattened = reshaped.flatten()
print("\nFlattened:", flattened)

# Transpose (flip rows and columns)
transposed = reshaped.T
print("\nTransposed:\n", transposed)
```

**👉 YOUR TURN:**
1. Create an array from 1 to 20
2. Reshape it to 4 rows and 5 columns
3. Get the transpose
4. Flatten it back to 1D

<details>
<summary>💡 Solution</summary>

```python
# 1. Create array
arr = np.arange(1, 21)
print("Original:", arr)

# 2. Reshape to 4x5
matrix = arr.reshape(4, 5)
print("\n4x5:\n", matrix)

# 3. Transpose
transposed = matrix.T
print("\nTransposed (5x4):\n", transposed)

# 4. Flatten
flat = transposed.flatten()
print("\nFlattened:", flat)
```
</details>

---

## PART 3: CHALLENGE EXERCISE (30-60 minutes)

### Real-World Scenario: Analyzing Student Performance

You are analyzing data for 10 students across 5 subjects. Build a complete analysis system.

**Task Breakdown:**

```python
# Step 1: Generate random student scores (50-100) for 10 students, 5 subjects
# Use: np.random.randint()
np.random.seed(42)  # For reproducibility
scores = # YOUR CODE HERE

# Step 2: Print the shape and first 3 students' scores

# Step 3: Calculate each student's average score

# Step 4: Find the overall class average

# Step 5: Find which student has the highest average

# Step 6: Find the average score for each subject

# Step 7: Identify students who are failing (average < 60)

# Step 8: Find the hardest subject (lowest average)

# Step 9: Create a "normalized" version where each score is 
#         the difference from that subject's mean

# Step 10: Count how many scores are above 90
```

**Expected Output Format:**
```
Student scores shape: (10, 5)
First 3 students:
[[...]]

Student averages: [...]
Class average: XX.XX
Top student: Student X with average YY.YY

Subject averages: [...]
Hardest subject: Subject X with average YY.YY

Failing students: [...]
Scores above 90: XX
```

<details>
<summary>💡 Full Solution (try for 30+ minutes first!)</summary>

```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Step 1: Generate scores
scores = np.random.randint(50, 101, size=(10, 5))

# Step 2: Shape and first 3
print("Student scores shape:", scores.shape)
print("\nFirst 3 students:")
print(scores[:3])

# Step 3: Student averages
student_averages = scores.mean(axis=1)
print("\nStudent averages:", student_averages)

# Step 4: Overall average
class_average = scores.mean()
print(f"\nClass average: {class_average:.2f}")

# Step 5: Top student
top_student_idx = student_averages.argmax()
top_student_avg = student_averages[top_student_idx]
print(f"\nTop student: Student {top_student_idx + 1} with average {top_student_avg:.2f}")

# Step 6: Subject averages
subject_averages = scores.mean(axis=0)
print("\nSubject averages:", subject_averages)

# Step 7: Failing students
failing_mask = student_averages < 60
failing_students = np.where(failing_mask)[0] + 1  # +1 for human numbering
if len(failing_students) > 0:
    print(f"\nFailing students: {failing_students}")
else:
    print("\nNo failing students!")

# Step 8: Hardest subject
hardest_subject_idx = subject_averages.argmin()
hardest_subject_avg = subject_averages[hardest_subject_idx]
print(f"\nHardest subject: Subject {hardest_subject_idx + 1} with average {hardest_subject_avg:.2f}")

# Step 9: Normalize scores
normalized_scores = scores - subject_averages
print("\nNormalized scores (first 3 students):")
print(normalized_scores[:3])

# Step 10: Count scores above 90
count_above_90 = (scores > 90).sum()
print(f"\nScores above 90: {count_above_90}")
```
</details>

---

## Bonus Challenge (Optional)

If you finish early, try this:

**Matrix Multiplication Practice:**
```python
# Create two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Tasks:
# 1. Element-wise multiplication (A * B)
# 2. Matrix multiplication (A @ B or np.dot(A, B))
# 3. What's the difference?
# 4. When would you use each in ML?
```

---

## Day 1 Wrap-Up

### Key Concepts Mastered Today ✅
- [ ] Creating NumPy arrays (1D, 2D, and using helper functions)
- [ ] Understanding array shapes and dimensions
- [ ] Indexing and slicing arrays
- [ ] Vectorized operations (the power of NumPy!)
- [ ] Aggregation functions (mean, sum, etc.)
- [ ] Boolean indexing for filtering
- [ ] Reshaping and transposing arrays

### Self-Assessment Questions

Before moving to Day 2, make sure you can answer:

1. **What is the shape of this array and what does it mean?**
   ```python
   arr = np.array([[1,2,3], [4,5,6]])
   ```

2. **How do you select all rows but only the first 2 columns?**

3. **What's the difference between `arr.sum(axis=0)` and `arr.sum(axis=1)`?**

4. **Why is NumPy faster than Python lists for numerical operations?**

5. **How would you find all values greater than 50 in an array?**

<details>
<summary>Answers</summary>

1. Shape is (2, 3) = 2 rows, 3 columns. In ML: 2 samples with 3 features each.

2. `arr[:, :2]` or `arr[:, 0:2]`

3. `axis=0` operates down rows (result has shape matching columns), `axis=1` operates across columns (result has shape matching rows)

4. NumPy uses vectorization with compiled C code and optimized memory layout

5. `arr[arr > 50]`
</details>

---

## Tomorrow's Preview: Day 2 - Advanced NumPy

You'll learn:
- Broadcasting rules
- Universal functions (ufuncs)
- Array copying vs. views
- Advanced indexing techniques
- Working with real datasets

---

## Resources for Tonight (Optional Reading - 20 mins max)

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html) - Review what you learned
- NumPy array visualization tool: [Array Visualizer](https://jalammar.github.io/visual-numpy/)

---

## Journal Prompt

Before ending today, write 3-5 sentences:
1. What concept made the most sense?
2. What concept needs more practice?
3. How do you see NumPy being used in ML? (based on what you learned)

---

**Congratulations on completing Day 1!** 🎉 

You've built the foundation for all ML work ahead. NumPy arrays are how data lives in machine learning—everything you learn from here builds on today's concepts.

**Remember:** If something doesn't click yet, that's normal. You'll use these concepts daily for the next 12 weeks, and they'll become second nature.

See you on Day 2! 💪
