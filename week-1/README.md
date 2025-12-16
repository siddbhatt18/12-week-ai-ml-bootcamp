**Week 1 Goal:**  
Get your environment ready and become comfortable using Python, Jupyter, NumPy, and pandas for basic data work.

Assumption: ~1.5–2 hours per day. Adjust up/down as needed.

---

## Day 1 – Environment Setup & First Notebook

**Objectives**
- Install Python + tools.
- Learn how to open and use Jupyter Notebook.

**Tasks**
1. **Install Python environment**
   - Install **Anaconda** (recommended) from: https://www.anaconda.com/download
     - During install, add Anaconda to PATH if prompted.
   - Open **Anaconda Prompt** (Windows) or Terminal (macOS/Linux).

2. **Create a dedicated ML environment**
   - In terminal:
     ```bash
     conda create -n ml python=3.11
     conda activate ml
     ```
   - Install key libraries:
     ```bash
     conda install numpy pandas matplotlib seaborn scikit-learn jupyterlab
     ```

3. **Launch JupyterLab / Notebook**
   - In the same terminal (with env activated):
     ```bash
     jupyter lab
     ```
     or
     ```bash
     jupyter notebook
     ```
   - This opens in your browser. Create a new **Python 3** notebook.

4. **First notebook tests**
   - In the first cell:
     ```python
     import sys
     print(sys.version)
     ```
   - In a new cell:
     ```python
     import numpy as np
     import pandas as pd
     import matplotlib.pyplot as plt

     print("NumPy:", np.__version__)
     print("pandas:", pd.__version__)
     ```
   - Save the notebook as `day1_setup.ipynb`.

**Outcome by end of Day 1**
- Working conda environment.
- Able to launch Jupyter and run simple Python code.

---

## Day 2 – Python Refresher & Basic Scripting

**Objectives**
- Refresh core Python: variables, lists, dicts, loops, functions.
- Practice writing and running code in a notebook.

**Tasks**
1. **Quick Python syntax refresher (30–40 min)**
   - In a notebook `day2_python_basics.ipynb`, practice:
     ```python
     # variables and types
     x = 10
     y = 3.5
     name = "Alice"
     is_active = True

     # lists
     nums = [1, 2, 3, 4, 5]
     print(nums[0], nums[-1])
     nums.append(6)

     # dicts
     person = {"name": "Bob", "age": 25}
     print(person["name"])
     person["city"] = "London"
     ```

2. **Control flow**
   ```python
   # if / elif / else
   n = 7
   if n % 2 == 0:
       print("even")
   else:
       print("odd")

   # for loop
   for i in range(5):
       print(i)

   # while loop
   count = 0
   while count < 3:
       print("count:", count)
       count += 1
   ```

3. **Functions & simple debugging**
   ```python
   def square(num):
       return num * num

   print(square(4))

   def describe_person(name, age):
       return f"{name} is {age} years old."

   print(describe_person("Alice", 30))
   ```

4. **Mini exercise (do yourself first, then check)**
   - Write a function `mean_of_list(lst)` that:
     - Takes a list of numbers.
     - Returns the arithmetic mean.
   - Test it on `[1, 2, 3, 4]` and `[10, 20, 30]`.

**Outcome by end of Day 2**
- Comfortable using basic Python constructs in Jupyter.
- Able to write and run simple functions.

---

## Day 3 – Intro to NumPy

**Objectives**
- Understand why NumPy arrays are used.
- Do basic numerical operations with NumPy.

**Tasks**
1. **Create arrays & basic operations**
   - New notebook: `day3_numpy_intro.ipynb`
   ```python
   import numpy as np

   a = np.array([1, 2, 3, 4, 5])
   print("a:", a)
   print("dtype:", a.dtype)

   b = np.array([[1, 2, 3],
                 [4, 5, 6]])
   print("b shape:", b.shape)

   print("a + 10:", a + 10)
   print("a * 2:", a * 2)
   print("a squared:", a ** 2)
   ```

2. **Statistics with NumPy**
   ```python
   print("mean:", a.mean())
   print("std:", a.std())
   print("min:", a.min(), "max:", a.max())
   ```
   - Create a random array:
     ```python
     rand = np.random.randn(1000)  # 1000 samples from normal distribution
     print(rand.mean(), rand.std())
     ```

3. **Indexing & slicing**
   ```python
   print(a[0], a[-1])
   print(a[1:4])    # slice
   print(b[0, 1])   # row 0, column 1
   print(b[:, 1])   # all rows, column 1
   ```

4. **Mini exercise**
   - Create a 3x3 array with values 1–9:
     ```python
     c = np.arange(1, 10).reshape(3, 3)
     ```
   - Compute:
     - Column-wise mean.
     - Row-wise sum.
   - Hint:
     ```python
     c.mean(axis=0)
     c.sum(axis=1)
     ```

**Outcome by end of Day 3**
- Able to create and manipulate NumPy arrays.
- Comfortable computing basic stats and indexing/slicing.

---

## Day 4 – Intro to pandas (Series & DataFrame)

**Objectives**
- Understand pandas `Series` and `DataFrame`.
- Learn basic ways to inspect and select data.

**Tasks**
1. **Create Series & DataFrame from scratch**
   - Notebook: `day4_pandas_intro.ipynb`
   ```python
   import pandas as pd

   s = pd.Series([10, 20, 30], index=["a", "b", "c"])
   print(s)
   print("Index:", s.index)
   print("Values:", s.values)

   data = {
       "name": ["Alice", "Bob", "Charlie"],
       "age": [25, 30, 35],
       "city": ["NY", "LA", "Chicago"]
   }
   df = pd.DataFrame(data)
   print(df)
   ```

2. **Inspect a DataFrame**
   ```python
   print(df.head())       # first rows
   print(df.info())       # dtypes, non-null
   print(df.describe())   # basic stats (numeric)
   print(df["age"])       # select column
   print(df[["name", "city"]])  # select multiple columns
   ```

3. **Row selection**
   ```python
   print(df.loc[0])   # label-based
   print(df.iloc[1])  # position-based
   ```

4. **Mini exercise**
   - Create a DataFrame `students` with columns:
     - `name`, `math_score`, `english_score`
   - Add 4–5 rows.
   - Compute:
     - Average math score.
     - Row where `english_score` is maximum.

   Hint:
   ```python
   students["math_score"].mean()
   students.loc[students["english_score"].idxmax()]
   ```

**Outcome by end of Day 4**
- Able to build and inspect small DataFrames.
- Comfortable with column and row selection basics.

---

## Day 5 – Loading Real Data & Simple Visualizations

**Objectives**
- Load a CSV dataset with pandas.
- Perform simple inspection & plotting with Matplotlib and Seaborn.

**Tasks**
1. **Get a real CSV dataset**
   - Download a simple public dataset, for example:
     - **Iris dataset** CSV from UCI or Kaggle, or
     - Any small “CSV demo” you find (e.g., on Kaggle search “Iris”).
   - Save it into a `data/` folder in your working directory (e.g., `data/iris.csv`).

2. **Load & inspect**
   - Notebook: `day5_real_data_eda.ipynb`
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns

   df = pd.read_csv("data/iris.csv")  # adjust filename if needed
   df.head()
   df.info()
   df.describe()
   ```

3. **Check basic properties**
   - Number of rows/columns: `df.shape`
   - Column names: `df.columns`
   - Missing values:
     ```python
     df.isna().sum()
     ```

4. **Basic plots**
   - Histogram of one numeric column:
     ```python
     df["sepal_length"].hist(bins=20)
     plt.xlabel("Sepal length")
     plt.ylabel("Count")
     plt.show()
     ```
   - Scatter plot of two features:
     ```python
     plt.scatter(df["sepal_length"], df["sepal_width"])
     plt.xlabel("Sepal length")
     plt.ylabel("Sepal width")
     plt.show()
     ```
   - Using seaborn for pairplot (if dataset is small):
     ```python
     sns.pairplot(df, hue="species")  # adjust target column name
     plt.show()
     ```

5. **Mini exercise**
   - Choose:
     - 1 numeric column → draw histogram.
     - 2 numeric columns → scatter plot.
   - Write 2–3 simple observations in a markdown cell, like:
     - “Most sepal lengths are between X and Y.”
     - “There seems to be a positive/negative relationship between A and B.”

**Outcome by end of Day 5**
- Able to load a CSV file with pandas.
- Can create basic plots to start understanding data.

---

## Day 6 – Data Cleaning Basics (Missing Values & Simple Transforms)

**Objectives**
- Learn to detect and handle missing values.
- Practice simple column transformations.

**Tasks**
1. **Continue with the same dataset** (or pick another simple one).
   - Notebook: `day6_data_cleaning.ipynb`
   ```python
   import pandas as pd
   df = pd.read_csv("data/your_dataset.csv")
   ```

2. **Check missing values**
   ```python
   df.isna().sum()
   ```
   - Identify which columns have missing values.

3. **Simple strategies**
   - Dropping rows with missing values:
     ```python
     df_drop = df.dropna()
     print(df.shape, df_drop.shape)
     ```
   - Filling missing numeric values with mean:
     ```python
     df_filled = df.copy()
     numeric_cols = df_filled.select_dtypes(include=["float64", "int64"]).columns
     for col in numeric_cols:
         df_filled[col].fillna(df_filled[col].mean(), inplace=True)
     ```
   - Alternatively, fill a single column:
     ```python
     df_filled["some_numeric_col"].fillna(df_filled["some_numeric_col"].mean(), inplace=True)
     ```

4. **Simple feature creation**
   - Create a new column that is a transformation of another:
     ```python
     df_filled["sepal_area"] = df_filled["sepal_length"] * df_filled["sepal_width"]
     ```
   - Try using `.apply` for a custom function:
     ```python
     def length_category(x):
         return "long" if x > df_filled["sepal_length"].mean() else "short"

     df_filled["length_cat"] = df_filled["sepal_length"].apply(length_category)
     ```

5. **Mini exercise**
   - Pick 1–2 numeric columns with missing values (or artificially create missing values with `df.sample` and set to NaN).
   - Fill them using:
     - Mean for one column.
     - Median for another.
   - Compare:
     - How many rows were affected in each case?

**Outcome by end of Day 6**
- Capable of identifying and handling missing data.
- Able to create simple derived features from existing columns.

---

## Day 7 – Putting it Together: Mini End‑to‑End Data Notebook

**Objectives**
- Combine everything from Week 1 into one clean notebook.
- Practice a tiny “mini project” start-to-finish.

**Tasks**
1. **Choose one dataset** (preferably the same as Day 5–6).
2. **New notebook:** `day7_week1_mini_project.ipynb`

3. **Structure the notebook into sections**
   - Use markdown cells to organize:
     1. Title & Objective  
        - Example: “Exploring the Iris Dataset – Week 1 Mini Project”  
        - One sentence: “Goal is to load, clean, and explore the dataset.”
     2. Imports & Data Loading
        ```python
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_csv("data/iris.csv")  # or your dataset
        ```
     3. Basic Inspection
        ```python
        df.head()
        df.info()
        df.describe()
        df.isna().sum()
        ```
     4. Simple Cleaning
        - Handle missing values (drop or fill).
        - Maybe create one new column.
     5. Exploratory Plots
        - At least:
          - One histogram.
          - One scatter plot.
          - (Optional) Pairplot or boxplot.

4. **Short written summary (in markdown)**
   - 4–6 sentences answering:
     - What does each row represent?
     - What are the main columns/features and their types?
     - Any missing values? How did you handle them?
     - Any interesting patterns from your plots?

5. **Self‑check**
   - Can you rerun the notebook top‑to‑bottom without errors?
   - Is the code reasonably organized and commented?

**Outcome by end of Day 7**
- One small, coherent notebook that:
  - Loads data
  - Cleans it lightly
  - Explores it with simple statistics and plots
- You’re comfortable with the basic tools needed for Week 2 (deeper EDA and train/test splitting).

---

If you’d like, tell me which dataset you picked (or want to pick), and I can give you tailored Day 5–7 exercises around that specific dataset.
