Day 1 is about getting fully set up so that from tomorrow onward you can focus on *learning ML*, not fighting with your tools.

Below is a structured, step‑by‑step guide you can follow. You don’t need to do everything in one sitting, but try to complete all steps before moving to Day 2.

---

# Day 1 – Environment Setup & First Notebook

**Goal:**  
Have a working Python + ML environment and be able to open a Jupyter notebook, run code, and save your work.

Approx. time: 1.5–2 hours (can be split into shorter blocks).

---

## 1. Choose Where You’ll Work

Decide:

- **Primary machine**: Windows / macOS / Linux laptop or desktop.
- **Working folder**: e.g., `~/ml-learning` (macOS/Linux) or `C:\Users\YourName\ml-learning` (Windows).

Create this folder now; you’ll store your notebooks and data here.

---

## 2. Install Anaconda (Python Distribution)

Anaconda bundles Python, many data libraries, and Jupyter. That makes your life easier.

1. Go to:  
   https://www.anaconda.com/download

2. Download:
   - **Windows**: 64‑bit Installer (Anaconda Distribution) for Python 3.x.
   - **macOS**: Intel or Apple Silicon version depending on your Mac.
   - **Linux**: 64‑bit installer (.sh).

3. Run installer:
   - When asked:
     - “Install for”: choose “Just Me” (default).
     - Destination folder: accept default.
     - **Windows**: If it asks to “Add Anaconda to PATH”, it is generally safe to leave it OFF (default). Anaconda Navigator / Prompt handle it for you.
   - Finish installation.

4. Verify installation:
   - **Windows**: Open “Anaconda Prompt” from Start menu.
   - **macOS/Linux**: Open Terminal and type:
     ```bash
     conda --version
     ```
   - You should see something like: `conda 24.3.0` (version number may differ).

**Checkpoint:** You can open a shell (Anaconda Prompt or Terminal) and `conda` is recognized.

---

## 3. Create a Dedicated ML Environment

You’ll use a separate environment called `ml` so experiments don’t break your base install.

1. In Anaconda Prompt / Terminal:

   ```bash
   conda create -n ml python=3.11
   ```

   - When asked `Proceed ([y]/n)?`, type `y` and press Enter.

2. Activate the environment:

   ```bash
   conda activate ml
   ```

   - You should see `(ml)` appear at the beginning of your prompt.

3. Install core libraries:

   ```bash
   conda install numpy pandas matplotlib seaborn scikit-learn jupyterlab
   ```

   - Confirm with `y` when prompted.

**Checkpoint:**  
Run:

```bash
python -V
```

You should see something like `Python 3.11.x`.  
Your prompt should start with `(ml)`.

---

## 4. Launch JupyterLab (or Classic Notebook)

You’ll use Jupyter for most of your learning.

1. Make sure you are in your chosen working folder:

   ```bash
   cd path/to/your/ml-learning
   ```
   Examples:
   - Windows: `cd C:\Users\YourName\ml-learning`
   - macOS/Linux: `cd ~/ml-learning`

2. Launch JupyterLab (recommended):

   ```bash
   jupyter lab
   ```

   - A browser window should open at a URL like `http://localhost:8888/lab`.

   If JupyterLab feels heavy or you prefer classic:

   ```bash
   jupyter notebook
   ```

3. Keep this terminal window open; it’s running the Jupyter server.  
   - To stop it later: go to the terminal and press `Ctrl + C`, then confirm with `y`.

**Checkpoint:** You see the Jupyter interface in your browser, showing your working folder.

---

## 5. Create Your First Notebook

Inside Jupyter:

1. In JupyterLab:
   - Click the **Python 3 (ipykernel)** icon under “Notebook” in the Launcher.
2. In classic Notebook:
   - Click **New → Python 3** (top right).

3. Immediately rename the notebook:
   - Top left: click on `Untitled.ipynb` → rename to `day1_setup.ipynb`.

---

## 6. Run Basic Python & Environment Checks

### 6.1. Check Python Version from Notebook

In the first cell:

```python
import sys
print(sys.version)
```

- Run the cell:
  - JupyterLab: Press `Shift + Enter`.
  - Classic Notebook: Same.

You should see something like: `3.11.x (default, ...)`.

### 6.2. Import Core Libraries and Check Versions

In a new cell:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("NumPy version:", np.__version__)
print("pandas version:", pd.__version__)
```

- If there are no errors, your core ML stack is installed correctly.

**Checkpoint:**  
- No `ModuleNotFoundError`.
- You see printed versions for NumPy and pandas.

If you see an error like `ModuleNotFoundError: No module named 'numpy'`:
- Confirm you installed libs in the **ml** environment:
  - Close Jupyter, go back to terminal, run:
    ```bash
    conda activate ml
    conda install numpy pandas matplotlib seaborn scikit-learn jupyterlab
    jupyter lab
    ```

---

## 7. First Simple NumPy Experiment

Goal: get a feel for numerical computation within a notebook.

In a new cell:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

print("Mean:", arr.mean())
print("Standard deviation:", arr.std())
print("Min:", arr.min(), "Max:", arr.max())
```

Then, try creating a random array:

```python
rand_arr = np.random.randn(1000)  # 1000 random numbers from normal distribution
print("Random mean:", rand_arr.mean())
print("Random std dev:", rand_arr.std())
```

You should see a mean close to 0 and standard deviation close to 1.

**Mini Exercise (5–10 minutes)**  
Without copying, in a new cell:

1. Create a NumPy array with values 10, 20, 30, 40, 50.
2. Compute and print:
   - The mean.
   - The sum.
   - The array multiplied by 2.

Example solution (only check after you try):

```python
arr2 = np.array([10, 20, 30, 40, 50])
print("Mean:", arr2.mean())
print("Sum:", arr2.sum())
print("Times 2:", arr2 * 2)
```

---

## 8. First Tiny pandas Experiment (Optional but Recommended)

You’ll go deeper on pandas in the next few days; for now, just ensure it works.

In a new cell:

```python
import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
}

df = pd.DataFrame(data)
print(df)
print("\nInfo:")
print(df.info())
```

- You should see a small table printed and summary info.

**Checkpoint:** pandas works and you see a simple DataFrame.

---

## 9. Try a Simple Plot

You’ll use plotting extensively for EDA, so verify it works.

In a new cell:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
```

You should see a simple sine curve plotted below the cell.

If you get a warning about interactive backends, you can also run at the top of the notebook:

```python
%matplotlib inline
```

(Only needed in some configurations; harmless otherwise.)

---

## 10. Save & Close Properly

1. In Jupyter:
   - Click **File → Save Notebook** (or `Ctrl + S` / `Cmd + S`).
2. Close the notebook tab.
3. Stop Jupyter:
   - Go back to the terminal running Jupyter.
   - Press `Ctrl + C`, then `y` to confirm.
4. (Optional) Deactivate the environment:

   ```bash
   conda deactivate
   ```

---

## 11. Day 1 Checklist

By the end of Day 1, you should be able to say “yes” to:

- [ ] I installed Anaconda.
- [ ] I created an environment `ml` and can activate it with `conda activate ml`.
- [ ] I installed `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyterlab` in that environment.
- [ ] I can run `jupyter lab` (or `jupyter notebook`) and see it open in my browser.
- [ ] I created `day1_setup.ipynb` and:
  - [ ] Printed Python and library versions.
  - [ ] Created a NumPy array and computed basic stats.
  - [ ] Created a simple pandas DataFrame and printed it.
  - [ ] Plotted a basic line chart.

If any box is **not** checked, that’s the thing to fix before moving to Day 2.

---

## 12. Optional: Small Reflection (5 minutes)

In a text/markdown cell at the bottom of `day1_setup.ipynb` (or in a separate notes file), write:

- One thing that was easy today.
- One thing that was confusing or took time (e.g., PATH issues, Jupyter not starting).
- Anything you want to revisit (e.g., conda environments, how Jupyter notebooks are stored).

This will help you track progress over the 12 weeks.

---

If you tell me your operating system (Windows/macOS/Linux) and whether anything fails during Day 1 (e.g., specific error messages), I can give you very targeted troubleshooting steps.
