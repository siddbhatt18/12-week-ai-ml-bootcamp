Day 1 is about getting *fully* set up and doing a few sanity checks so you’re ready to learn without friction the rest of the week.

Below is a detailed, structured plan for **Day 1 of Week 1** (target: ~1.5–2.5 hours).  
If your OS or constraints differ (no admin rights, etc.), tell me and I’ll adapt.

---

## High-Level Objectives for Day 1

By the end of Day 1, you should:

1. Have a working **Python + ML environment** (with conda).
2. Be able to **open JupyterLab/Notebook** from that environment.
3. Run a **first notebook** that:
   - Prints Python and library versions.
   - Runs a few simple NumPy and pandas commands.
4. Save your work in a clear folder structure you’ll reuse later.

---

## Prerequisites / Decisions

Before you start:

- **Machine:** Ideally a laptop/desktop with:
  - OS: Windows, macOS, or Linux
  - RAM: ≥ 8 GB (4 GB is still OK for this plan)
- **Admin rights:** Helpful, but not strictly required.
- **Choose an editor:** We’ll use **JupyterLab**; you can add VS Code later.

If any of these are a problem (e.g., Chromebook, no install rights), pause here and let me know; the setup will be a bit different.

---

## Step 0 (5–10 min): Create a Study Folder & Notes File

**Goal:** Organize from day one.

1. Create a main folder somewhere easy to find, e.g.:
   - `C:\ml-journey` (Windows)
   - `/Users/<you>/ml-journey` (macOS)
   - `/home/<you>/ml-journey` (Linux)
2. Inside it, create:
   - `week1/`
   - `data/`
   - `notes/`
3. In `notes/`, create a text or markdown file:
   - `week1_log.md`  
   Add a simple template:

   ```markdown
   # Week 1 Learning Log

   ## Day 1
   - Time spent:
   - What I did:
   - What worked well:
   - What was confusing:
   - Questions to revisit:
   ```

You’ll fill in the Day 1 section at the end.

---

## Step 1 (20–30 min): Install Anaconda or Miniconda

**Goal:** Get a robust Python distribution with package management.

### 1.1 Choose Anaconda vs Miniconda

- **Anaconda**: larger, includes many packages by default; easiest for beginners.
- **Miniconda**: minimal, installs only what you ask for.

If you’re unsure: choose **Anaconda**.

### 1.2 Download & Install

1. Go to:
   - Anaconda: https://www.anaconda.com/download
   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Download the installer for:
   - Your OS (Windows/macOS/Linux)
   - 64-bit version
3. Run the installer:
   - Accept license, default options are fine.
   - If you see **“Add Anaconda to my PATH”**:
     - Recommended: check it if you’re on your own machine.
     - If unsure, leave unchecked; the Anaconda Prompt will still work.

**Checkpoint:**  
You should now be able to open:
- **Windows:** “Anaconda Prompt” from Start menu.
- **macOS/Linux:** A terminal window, then `conda --version` should run without error.

If `conda` is not recognized, tell me your OS and I’ll help troubleshoot.

---

## Step 2 (20–30 min): Create a Dedicated ML Environment

**Goal:** Isolate your ML work in a clean environment (`ml`) so you avoid dependency chaos later.

### 2.1 Open Terminal / Anaconda Prompt

- **Windows:** Start → search “Anaconda Prompt” → open.
- **macOS/Linux:** Open Terminal (e.g., Spotlight → “Terminal”).

### 2.2 Create the Environment

In the terminal:

```bash
conda create -n ml python=3.11
```

You’ll be asked `Proceed ([y]/n)?` → type `y` and press Enter.

### 2.3 Activate the Environment

```bash
conda activate ml
```

You should see `(ml)` appear at the beginning of your prompt line.

**Checkpoint:**  
Run:

```bash
python --version
```

You should see something like `Python 3.11.x`.

If activation or creation fails, copy the error and I’ll help you fix it.

---

## Step 3 (20–30 min): Install Core Libraries

**Goal:** Install the essential ML stack for the entire 12-week plan.

With `(ml)` still active, run:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

- Type `y` when asked to proceed.
- Let it complete; it may take a few minutes.

**Optional but useful later:**

```bash
conda install ipykernel
```

This ensures the `ml` environment shows up as a kernel in Jupyter.

**Checkpoint:**  
Run:

```bash
python -c "import numpy, pandas, sklearn, matplotlib; print('OK')"
```

You should see `OK`. If you see an ImportError, share the exact message.

---

## Step 4 (20–30 min): Launch JupyterLab & Create First Notebook

**Goal:** Confirm that Jupyter works and is using the right environment.

### 4.1 Navigate to Your Project Folder

From the same terminal (with `(ml)` activated):

```bash
cd path/to/ml-journey
```

Example:
- Windows: `cd C:\ml-journey`
- macOS: `cd /Users/<you>/ml-journey`

### 4.2 Launch JupyterLab

```bash
jupyter lab
```

This should open a browser window at something like `http://localhost:8888/lab`.

If it doesn’t open automatically, copy the URL from the terminal into your browser.

### 4.3 Create a Notebook

1. In JupyterLab:
   - Click the “+” or “Launcher” tab.
   - Under “Notebook”, click **Python 3 (ipykernel)** (or similar).
2. Save it immediately:
   - File → Save As…
   - Name: `week1_day1_setup.ipynb`
   - Store it in `week1/` folder inside `ml-journey`.

**Checkpoint:**  
The notebook title at the top should read `week1_day1_setup.ipynb`, and the kernel should be the one from your `ml` env (often just “Python 3”).

If you don’t see the right kernel, we can register it manually later.

---

## Step 5 (20–30 min): Sanity Checks in the First Notebook

**Goal:** Test that Python and key libraries work as expected.

### 5.1 Check Python Version

In the first cell, type:

```python
import sys
print(sys.version)
```

Run the cell (`Shift+Enter`).  
You should see a 3.11.x version.

### 5.2 Import Core Libraries & Print Versions

New cell:

```python
import numpy as np
import pandas as pd
import matplotlib
import sklearn

print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
```

If this runs without errors, your core stack is ready.

### 5.3 Quick NumPy Test

New cell:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", arr.mean())
print("Standard deviation:", arr.std())
```

You should get numeric outputs, not errors.

### 5.4 Quick pandas Test

New cell:

```python
import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
}
df = pd.DataFrame(data)
display(df)
print("Describe:")
display(df.describe())
```

You should see a small table and basic stats for `age`.

### 5.5 Tiny Plot Test (Matplotlib)

New cell:

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 1, 3, 5, 4]

plt.plot(x, y, marker="o")
plt.title("Test Line Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

You should see a simple line chart rendered below the cell.

If the plot doesn’t display, add this line at the top of the cell and run again:

```python
%matplotlib inline
```

**Checkpoint:**  
If all these cells run without errors, your environment is good enough to proceed with the rest of Week 1.

---

## Step 6 (10–15 min): Organize & Add a Tiny Exercise

**Goal:** Create the habit of writing and saving work + a small coding action.

### 6.1 Organize the Notebook

- Add **markdown cells** (Insert → Insert Cell Below → change to Markdown) at the top with headings:

  ```markdown
  # Week 1 – Day 1: Environment Setup

  ## Objectives
  - Install Python + conda
  - Create ML environment
  - Install core libraries
  - Verify Jupyter + basic imports

  ## System Info
  (Python and library versions here)
  ```

- Use markdown headers (e.g. `## NumPy Test`, `## pandas Test`, `## Matplotlib Test`) to separate sections.

### 6.2 Tiny Exercise

Add a new cell with a small task:

```python
import numpy as np
import pandas as pd

# 1. Create a NumPy array of 10 random numbers
rand_arr = np.random.randn(10)

# 2. Convert it into a pandas Series
s = pd.Series(rand_arr, name="random_values")

# 3. Print mean and standard deviation using pandas
print("Mean:", s.mean())
print("Std:", s.std())

s.head()
```

This is your first micro “data science” snippet: generating data, turning it into a Series, computing stats, and printing results.

---

## Step 7 (5–10 min): Reflect & Log Day 1

**Goal:** Reinforce learning and make future review easier.

Open `notes/week1_log.md` and fill in for **Day 1**:

Example:

```markdown
## Day 1
- Time spent: 2 hours
- What I did:
  - Installed Anaconda and created an `ml` environment with Python 3.11.
  - Installed numpy, pandas, matplotlib, seaborn, scikit-learn, jupyterlab.
  - Launched JupyterLab and created my first notebook.
  - Tested imports, basic NumPy/pandas operations, and a simple plot.
- What worked well:
  - Conda environment creation and package installation were straightforward.
  - JupyterLab interface feels intuitive.
- What was confusing:
  - (Write anything that felt unclear, e.g., environment activation, where files are saved, etc.)
- Questions to revisit:
  - (e.g., What’s the difference between Anaconda and Miniconda? What exactly is a “kernel”?)
```

This reflection habit will help you notice patterns in what confuses you and where you’re improving.

---

## Summary: End-of-Day 1 Checklist

You are ready for Day 2 if you can:

- [ ] Open a terminal / Anaconda Prompt and run `conda activate ml`.
- [ ] Launch JupyterLab with `jupyter lab` from your `ml-journey` folder.
- [ ] Create and save a notebook in `week1/`.
- [ ] Import `numpy`, `pandas`, `matplotlib`, `sklearn` with no errors.
- [ ] Create a small NumPy array, compute mean/std.
- [ ] Create a small pandas DataFrame and view it.
- [ ] Draw a simple line plot in the notebook.

If any box is not checked or something failed, tell me:
- Your OS (Windows/macOS/Linux),
- The exact command you ran,
- The error message (copy/paste),

and I’ll help you debug before you move on to Day 2.
