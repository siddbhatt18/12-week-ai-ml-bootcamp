Below is a detailed, step‑by‑step guide for **Day 3, Week 1**. Today you’ll move from plain Python lists to **NumPy arrays**, which are the backbone of numerical computing and machine learning in Python.

---

# Day 3 – Intro to NumPy (Arrays & Basic Operations)

**Goal:**  
Understand what NumPy arrays are, why they’re used for ML, and practice basic operations: creation, shapes, indexing, slicing, and simple statistics.

Approx. time: 1.5–2 hours.

---

## 0. Setup: Open Your Environment

1. **Activate your environment** (in Terminal / Anaconda Prompt):

   ```bash
   conda activate ml
   ```

2. **Go to your working folder** (same as previous days):

   ```bash
   cd path/to/your/ml-learning
   ```

3. **Launch JupyterLab** (or Notebook):

   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

4. **Create a new notebook**:
   - Name it: `day3_numpy_intro.ipynb`.

---

## 1. What is NumPy and Why Use It?

**Concept (read, no coding yet):**

- Python **lists** are flexible but slow for large numerical operations.
- **NumPy arrays** (`ndarray`) are:
  - Stored in **contiguous memory**
  - Optimized for **vectorized operations** (operate on whole arrays at once)
  - Used internally by most ML libraries (pandas, scikit‑learn, TensorFlow, PyTorch).

You’ll be using NumPy throughout the entire ML journey.

---

## 2. Import NumPy and Create Basic Arrays

### 2.1. Import NumPy

In the first cell:

```python
import numpy as np
```

This is the standard alias used everywhere.

### 2.2. Create 1D (vector) arrays

```python
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Type:", type(arr))
print("dtype:", arr.dtype)        # data type of elements
print("Shape:", arr.shape)        # dimensions
print("Number of dimensions:", arr.ndim)
```

### 2.3. Create 2D (matrix) arrays

```python
mat = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Matrix:\n", mat)
print("Shape:", mat.shape)
print("Dimensions:", mat.ndim)
```

**Key idea:**  
- 1D array: shape like `(5,)`
- 2D array: shape like `(rows, columns)`

---

## 3. Creating Arrays with Helper Functions

Learn useful constructors: `arange`, `zeros`, `ones`, `linspace`, `random`.

### 3.1. `np.arange` – ranges

```python
a = np.arange(0, 10)       # 0..9
b = np.arange(0, 10, 2)    # 0,2,4,6,8

print("a:", a)
print("b:", b)
```

### 3.2. `np.zeros` and `np.ones`

```python
zeros = np.zeros((3, 4))   # 3x4 matrix of zeros
ones = np.ones((2, 3))     # 2x3 matrix of ones

print("zeros:\n", zeros)
print("ones:\n", ones)
```

### 3.3. `np.linspace` – evenly spaced numbers

```python
c = np.linspace(0, 1, 5)   # 5 numbers from 0 to 1 inclusive
print("linspace:", c)
```

### 3.4. Random numbers

```python
rand_uniform = np.random.rand(3, 3)    # 3x3 uniform [0,1)
rand_normal = np.random.randn(3, 3)    # 3x3 standard normal

print("Uniform random:\n", rand_uniform)
print("Normal random:\n", rand_normal)
```

---

### Mini Exercise 1 (10 minutes)

Create the following:

1. A 1D array `x` with values from 0 to 20 (inclusive) with step 5 → `[0, 5, 10, 15, 20]`.
2. A 2x5 array of all zeros.
3. A 4x4 array of all ones.
4. A 1D array of 6 values evenly spaced between -1 and 1.

Check shapes with `.shape`.

*Solution (after you try):*

```python
x = np.arange(0, 21, 5)
zeros_2x5 = np.zeros((2, 5))
ones_4x4 = np.ones((4, 4))
lin = np.linspace(-1, 1, 6)

print("x:", x, "shape:", x.shape)
print("zeros_2x5 shape:", zeros_2x5.shape)
print("ones_4x4 shape:", ones_4x4.shape)
print("lin:", lin)
```

---

## 4. Array Attributes: Shape, Reshape, and Dimensions

Understanding shapes and reshaping is crucial for ML.

### 4.1. Check shape and size

```python
arr = np.arange(12)   # 0..11
print("arr:", arr)
print("Shape:", arr.shape)
print("Size:", arr.size)  # total number of elements
print("Dimensions:", arr.ndim)
```

### 4.2. Reshape arrays

```python
mat_3x4 = arr.reshape(3, 4)
print("Reshaped to 3x4:\n", mat_3x4)

mat_2x6 = arr.reshape(2, 6)
print("Reshaped to 2x6:\n", mat_2x6)
```

You must keep the total number of elements the same (here: 12).

### 4.3. Using `-1` in reshape

`-1` tells NumPy to infer that dimension automatically.

```python
mat_auto = arr.reshape(-1, 3)   # 4 rows, 3 columns
print("Reshaped with -1:\n", mat_auto)
print("Shape:", mat_auto.shape)
```

---

### Mini Exercise 2 (10 minutes)

1. Create an array `a` with numbers from 1 to 16.
2. Reshape it to:
   - shape `(4, 4)`
   - shape `(2, 8)`
3. Use `-1` to create a 2D array with 4 columns.

*Solution (after trying):*

```python
a = np.arange(1, 17)
mat_4x4 = a.reshape(4, 4)
mat_2x8 = a.reshape(2, 8)
mat_auto_4cols = a.reshape(-1, 4)

print("4x4:\n", mat_4x4)
print("2x8:\n", mat_2x8)
print("auto with 4 cols:\n", mat_auto_4cols)
print("Shape of auto:", mat_auto_4cols.shape)
```

---

## 5. Basic Arithmetic and Vectorized Operations

NumPy operations apply element‑wise by default.

### 5.1. Element‑wise arithmetic

```python
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])

print("x + y:", x + y)
print("x - y:", x - y)
print("x * y:", x * y)
print("x / y:", x / y)
print("x ** 2:", x ** 2)
```

### 5.2. Scalar operations

```python
print("x + 10:", x + 10)
print("x * 3:", x * 3)
```

### 5.3. Comparison operations

```python
print("x > 2:", x > 2)
print("y == 20:", y == 20)
```

These return boolean arrays.

---

### Mini Exercise 3 (10 minutes)

1. Create an array `temps_c = np.array([0, 10, 20, 30, 40])`.
2. Convert to Fahrenheit: `F = C * 9/5 + 32`.
3. Create a boolean mask `hot = F > 80`.
4. Print `F` and the `hot` mask.

*Solution:*

```python
temps_c = np.array([0, 10, 20, 30, 40])
temps_f = temps_c * 9/5 + 32
hot = temps_f > 80

print("Celsius:", temps_c)
print("Fahrenheit:", temps_f)
print("Is hot (>80F):", hot)
```

---

## 6. Indexing and Slicing

You must be very comfortable with this; it’s how you select rows/columns.

### 6.1. 1D array indexing

```python
a = np.array([10, 20, 30, 40, 50])

print("First element:", a[0])
print("Last element:", a[-1])

print("Elements from index 1 to 3:", a[1:4])   # 20,30,40
print("Every second element:", a[::2])        # 10,30,50
```

### 6.2. 2D array indexing

```python
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("b:\n", b)
print("Element at row 0, col 1:", b[0, 1])    # 2
print("First row:", b[0, :])                 # [1,2,3]
print("Second column:", b[:, 1])             # [2,5,8]
print("Submatrix (rows 0-1, cols 1-2):\n", b[0:2, 1:3])
```

---

### Mini Exercise 4 (10–15 minutes)

Given:

```python
m = np.arange(1, 13).reshape(3, 4)
print(m)
```

1. Print the second row.
2. Print the third column.
3. Print the top‑left 2x2 submatrix.
4. Print all elements in the last column.

*Solution (after trying):*

```python
print("Second row:", m[1, :])
print("Third column:", m[:, 2])
print("Top-left 2x2:\n", m[0:2, 0:2])
print("Last column:", m[:, -1])
```

---

## 7. Basic Statistics and Aggregations

You’ll often need means, sums, etc., over arrays.

### 7.1. Overall statistics

```python
data = np.array([1, 2, 3, 4, 5, 6])

print("Mean:", data.mean())
print("Sum:", data.sum())
print("Min:", data.min(), "Max:", data.max())
print("Std dev:", data.std())
```

### 7.2. Statistics along an axis

```python
mat = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Matrix:\n", mat)

print("Column-wise mean:", mat.mean(axis=0))   # means of each column
print("Row-wise sum:", mat.sum(axis=1))       # sums of each row
```

- `axis=0`: operate **down columns** (across rows).
- `axis=1`: operate **across columns** (per row).

---

### Mini Exercise 5 (10–15 minutes)

1. Create a 4x3 matrix with values from 1 to 12:

   ```python
   m = np.arange(1, 13).reshape(4, 3)
   ```

2. Compute:
   - The mean of each column.
   - The max of each row.
3. Compute:
   - Global mean (all elements).
   - Global standard deviation.

*Solution:*

```python
m = np.arange(1, 13).reshape(4, 3)
print("m:\n", m)

col_means = m.mean(axis=0)
row_maxes = m.max(axis=1)
global_mean = m.mean()
global_std = m.std()

print("Column means:", col_means)
print("Row maxes:", row_maxes)
print("Global mean:", global_mean)
print("Global std:", global_std)
```

---

## 8. Small “Mini Project”: Simulated Height Data

**Objective:** Use NumPy to simulate data and compute simple stats, like you would for a real dataset.

### 8.1. Generate synthetic heights

In your notebook:

```python
np.random.seed(42)  # for reproducibility

# simulate heights (in cm) for 1000 people, normal distribution
heights = np.random.normal(loc=170, scale=10, size=1000)  # mean=170cm, std=10cm
```

### 8.2. Compute statistics

```python
print("Number of people:", heights.size)
print("Mean height:", heights.mean())
print("Median height:", np.median(heights))
print("Min height:", heights.min())
print("Max height:", heights.max())
print("Std dev:", heights.std())
```

### 8.3. Simple selection with boolean masks

```python
tall = heights > 185
short = heights < 160

print("Number taller than 185cm:", tall.sum())
print("Number shorter than 160cm:", short.sum())

# Extract heights for tall people
tall_heights = heights[tall]
print("Mean height of tall group:", tall_heights.mean())
```

### 8.4. (Optional) Simple histogram using Matplotlib

You’ll do more plotting later, but this connects NumPy to plots.

```python
import matplotlib.pyplot as plt

plt.hist(heights, bins=30)
plt.title("Simulated Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Count")
plt.show()
```

---

## 9. Organize & Save Your Notebook

Before finishing:

1. Add **markdown headings** to your notebook:
   - `# Intro to NumPy`
   - `## Creating Arrays`
   - `## Shapes and Reshaping`
   - `## Arithmetic and Vectorized Operations`
   - `## Indexing and Slicing`
   - `## Statistics and Aggregations`
   - `## Mini Project: Simulated Heights`
2. Use **Kernel → Restart & Run All** to ensure:
   - No cells depend on “hidden” state.
   - Everything runs cleanly from top to bottom.
3. Save (`Ctrl + S` / `Cmd + S` or File → Save).

---

## 10. Quick Reflection (3–5 minutes)

In a final markdown cell, answer briefly:

- What’s one clear advantage of NumPy arrays over Python lists?
- Which NumPy concept felt most confusing (reshape, axis, indexing, etc.)?
- Which piece do you want to review tomorrow before moving on?

---

## Day 3 Completion Checklist

By the end of Day 3, you should be able to say “yes” to:

- [ ] I can create 1D and 2D NumPy arrays.
- [ ] I understand what `.shape`, `.ndim`, and `.size` mean.
- [ ] I can reshape arrays (and use `-1` for automatic dimension).
- [ ] I can perform element‑wise arithmetic and scalar operations on arrays.
- [ ] I can index and slice 1D and 2D arrays to select elements, rows, and columns.
- [ ] I can compute basic statistics (mean, sum, std, min, max) overall and along axes.
- [ ] I completed the “simulated heights” mini project using NumPy.

If you’d like, you can paste your “simulated heights” code or any exercise that felt tricky, and I can review it or suggest more targeted practice.
