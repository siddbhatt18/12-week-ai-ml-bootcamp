# Week 1: Python & Essential Math Foundations - Daily Study Plan

## **Learning Objectives for Week 1**
- Master Python data science libraries (NumPy, Pandas, Matplotlib)
- Understand core linear algebra concepts for ML
- Build statistical intuition with hands-on coding
- Develop mathematical thinking for gradient descent

---

## **📅 Day 1 (Monday): Python & NumPy Fundamentals**
**Total Time: 3 hours**

### **Hour 1: Environment Setup & Python Review (Theory)**
- [ ] Set up development environment
  - Install Anaconda or Miniconda
  - Create virtual environment for ML projects
  - Install essential packages: `numpy`, `pandas`, `matplotlib`, `jupyter`, `scikit-learn`
- [ ] Python refresher:
  - List comprehensions
  - Lambda functions
  - Classes and objects basics
  - File I/O operations

### **Hour 2: NumPy Deep Dive (Coding)**
- [ ] Work through NumPy basics:
  ```python
  # Create notebook: "01_numpy_fundamentals.ipynb"
  # Topics to cover:
  - Array creation (np.array, np.zeros, np.ones, np.arange, np.linspace)
  - Array indexing and slicing
  - Array reshaping and transposition
  - Mathematical operations (element-wise)
  ```
- [ ] Complete exercises:
  - Create a 5x5 matrix with values 1-25
  - Extract all odd numbers from an array
  - Replace all values > 30 with 30 in an array
  - Compute running mean of 1D array

### **Hour 3: Practice & Documentation (Application)**
- [ ] Implement from scratch:
  ```python
  # Create: "day1_practice.py"
  def matrix_multiply(A, B):
      """Implement matrix multiplication without np.dot"""
      pass
  
  def normalize_array(arr):
      """Normalize array to 0-1 range"""
      pass
  ```
- [ ] Create personal notes document with key NumPy operations
- [ ] **Mini-project:** Create a simple image manipulation script using NumPy (load image as array, apply filters)

### **📊 Day 1 Deliverables:**
- Working ML environment
- Completed NumPy fundamentals notebook
- 5+ NumPy practice problems solved

---

## **📅 Day 2 (Tuesday): Advanced NumPy & Linear Algebra Basics**
**Total Time: 3 hours**

### **Hour 1: Linear Algebra Concepts (Theory)**
- [ ] Watch 3Blue1Brown's "Essence of Linear Algebra" Episodes 1-3
- [ ] Study concepts:
  - Vectors as points in space
  - Vector addition and scalar multiplication
  - Dot product geometric interpretation
  - Matrix as linear transformation
- [ ] Read: Chapter 2.1-2.3 of "Mathematics for Machine Learning" (free PDF)

### **Hour 2: NumPy Linear Algebra (Coding)**
- [ ] Create notebook: "02_linalg_with_numpy.ipynb"
  ```python
  # Implement and visualize:
  - Vector operations (addition, scalar multiplication)
  - Dot product and its properties
  - Matrix multiplication step-by-step
  - Computing matrix determinant and inverse
  - Solving linear equations with NumPy
  ```
- [ ] Coding exercises:
  - Calculate angle between two vectors
  - Project one vector onto another
  - Verify matrix multiplication properties (associative, distributive)

### **Hour 3: Visualization & Application (Practice)**
- [ ] Create visualizations:
  ```python
  # Create: "linalg_visualizations.ipynb"
  - Plot 2D vectors
  - Visualize vector addition
  - Show linear transformation of unit circle
  - Animate rotation matrices
  ```
- [ ] **Mini-project:** Build a simple 2D transformation tool
  - Apply rotation, scaling, translation to shapes
  - Visualize the transformations

### **📊 Day 2 Deliverables:**
- Linear algebra operations notebook
- 5+ vector/matrix exercises completed
- Visualization of at least 3 linear algebra concepts

---

## **📅 Day 3 (Wednesday): Pandas Mastery**
**Total Time: 3 hours**

### **Hour 1: Pandas Fundamentals (Theory)**
- [ ] Study Pandas core concepts:
  - Series vs DataFrame
  - Index and MultiIndex
  - Data types and memory usage
  - Method chaining philosophy
- [ ] Read: "Python Data Science Handbook" Chapter 3 (first half)

### **Hour 2: Data Manipulation (Coding)**
- [ ] Create notebook: "03_pandas_operations.ipynb"
  ```python
  # Master these operations:
  - Loading data (CSV, JSON, Excel)
  - Selecting and filtering data
  - Handling missing values (dropna, fillna)
  - GroupBy operations
  - Merge, join, and concatenate
  - Pivot tables and cross-tabulation
  ```
- [ ] Work with real dataset:
  - Download Titanic dataset from Kaggle
  - Answer 10 questions using Pandas (e.g., survival rate by class)

### **Hour 3: Advanced Pandas (Application)**
- [ ] Advanced operations:
  ```python
  # Create: "pandas_advanced.ipynb"
  - Apply and applymap functions
  - Window functions (rolling, expanding)
  - Time series basics
  - Memory optimization techniques
  - Custom aggregation functions
  ```
- [ ] **Mini-project:** Create a data analysis pipeline
  - Load messy dataset
  - Clean and transform
  - Generate summary statistics
  - Export cleaned data

### **📊 Day 3 Deliverables:**
- Complete Pandas operations notebook
- Titanic dataset analysis (10 insights)
- Reusable data cleaning function library

---

## **📅 Day 4 (Thursday): Statistics Foundations**
**Total Time: 3 hours**

### **Hour 1: Descriptive Statistics (Theory)**
- [ ] Study core concepts:
  - Measures of central tendency (mean, median, mode)
  - Measures of spread (variance, std dev, IQR)
  - Skewness and kurtosis
  - Correlation vs causation
- [ ] Understand distributions:
  - Normal distribution properties
  - Binomial, Poisson basics
  - Central Limit Theorem intuition

### **Hour 2: Statistical Implementation (Coding)**
- [ ] Create notebook: "04_statistics_from_scratch.ipynb"
  ```python
  # Implement without using libraries:
  def calculate_mean(data):
      pass
  
  def calculate_variance(data):
      pass
  
  def calculate_correlation(x, y):
      pass
  
  def confidence_interval(data, confidence=0.95):
      pass
  ```
- [ ] Verify implementations against NumPy/SciPy

### **Hour 3: Statistical Analysis (Application)**
- [ ] Real data analysis:
  ```python
  # Create: "statistical_analysis.ipynb"
  - Load a real dataset (e.g., Boston Housing)
  - Calculate all descriptive statistics
  - Test for normality (QQ plots, Shapiro-Wilk)
  - Identify outliers (IQR, Z-score methods)
  - Correlation analysis with heatmap
  ```
- [ ] **Mini-project:** Statistical report generator
  - Automated EDA function
  - Generate PDF report with findings

### **📊 Day 4 Deliverables:**
- Statistical functions implemented from scratch
- Complete statistical analysis of one dataset
- Automated EDA report generator

---

## **📅 Day 5 (Friday): Matplotlib & Data Visualization**
**Total Time: 3 hours**

### **Hour 1: Matplotlib Fundamentals (Theory)**
- [ ] Understanding the architecture:
  - Figure, Axes, and Artists
  - Object-oriented vs pyplot interface
  - Customization options
- [ ] Study visualization best practices:
  - Choosing right chart types
  - Color theory for data viz
  - Avoiding chartjunk

### **Hour 2: Creating Visualizations (Coding)**
- [ ] Create notebook: "05_matplotlib_mastery.ipynb"
  ```python
  # Master these plot types:
  - Line plots with multiple series
  - Scatter plots with color/size encoding
  - Histograms and density plots
  - Box plots and violin plots
  - Heatmaps and contour plots
  - Subplots and complex layouts
  ```
- [ ] Customization practice:
  - Custom color schemes
  - Annotations and text
  - Dual axes plots
  - Save high-quality figures

### **Hour 3: Advanced Visualization (Application)**
- [ ] Create interactive visualizations:
  ```python
  # Create: "advanced_viz.ipynb"
  - 3D plots (surface, wireframe)
  - Animated plots
  - Statistical plots with confidence intervals
  - Custom dashboard layout
  ```
- [ ] **Mini-project:** Build a visualization dashboard
  - 6-panel dashboard for dataset exploration
  - Include statistical overlays
  - Export as PDF/PNG

### **📊 Day 5 Deliverables:**
- 10+ different visualization types created
- Custom visualization style guide
- Complete visualization dashboard

---

## **📅 Day 6 (Saturday): Calculus Intuition & Gradient Descent**
**Total Time: 4 hours**

### **Hours 1-2: Calculus for ML (Theory + Visual)**
- [ ] Watch 3Blue1Brown's "Essence of Calculus" Episodes 1-3
- [ ] Understand key concepts:
  - Derivatives as rate of change
  - Partial derivatives
  - Chain rule importance in ML
  - Gradient vector meaning
- [ ] Read blog post: "Calculus on Computational Graphs" (colah.github.io)
- [ ] Create visual notes with diagrams

### **Hour 3: Implementing Gradient Descent (Coding)**
- [ ] Create notebook: "06_gradient_descent.ipynb"
  ```python
  # Implement from scratch:
  def compute_gradient(x, y, theta):
      """Compute gradient for linear regression"""
      pass
  
  def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
      """Full gradient descent implementation"""
      pass
  
  def stochastic_gradient_descent(X, y, learning_rate=0.01):
      """SGD implementation"""
      pass
  ```
- [ ] Visualize the optimization process:
  - Loss surface in 3D
  - Gradient descent path
  - Effect of learning rate

### **Hour 4: Experimentation (Application)**
- [ ] Experiments with gradient descent:
  ```python
  # Create: "gradient_experiments.ipynb"
  - Compare different learning rates
  - Implement momentum
  - Visualize convergence
  - Test on non-convex functions
  ```
- [ ] **Mini-project:** Interactive gradient descent visualizer
  - Adjustable parameters
  - Real-time updates
  - Multiple optimization algorithms

### **📊 Day 6 Deliverables:**
- Working gradient descent implementation
- Visualization of optimization process
- Comparison of 3+ learning rates

---

## **📅 Day 7 (Sunday): Integration & First ML Model**
**Total Time: 4 hours**

### **Hours 1-2: Review & Consolidation (Theory + Practice)**
- [ ] Review week's notebooks
- [ ] Complete any unfinished exercises
- [ ] Create a "Week 1 Cheat Sheet" with:
  - Essential NumPy operations
  - Key Pandas methods
  - Statistical formulas
  - Linear algebra operations
- [ ] Organize code into reusable modules

### **Hour 3: Build Your First ML Model (Coding)**
- [ ] Create notebook: "07_first_ml_model.ipynb"
  ```python
  # Build linear regression from scratch:
  class LinearRegression:
      def __init__(self, learning_rate=0.01):
          pass
      
      def fit(self, X, y):
          """Training using gradient descent"""
          pass
      
      def predict(self, X):
          """Make predictions"""
          pass
      
      def score(self, X, y):
          """Calculate R-squared"""
          pass
  ```
- [ ] Test on synthetic data
- [ ] Visualize predictions vs actual

### **Hour 4: Week 1 Capstone Project (Application)**
- [ ] **Capstone:** Complete data science pipeline
  ```python
  # Create: "week1_capstone.ipynb"
  1. Load California housing dataset
  2. Perform complete EDA with visualizations
  3. Implement linear regression from scratch
  4. Train model using gradient descent
  5. Evaluate and visualize results
  6. Create summary report
  ```
- [ ] Reflection:
  - Document what you learned
  - Identify areas for deeper study
  - Plan improvements for Week 2

### **📊 Day 7 Deliverables:**
- Complete linear regression implementation
- Working model with 75%+ accuracy
- Week 1 summary document
- Organized code repository

---

## **📝 Week 1 Success Metrics**

### **Technical Skills Checklist:**
- [ ] Can manipulate arrays with NumPy efficiently
- [ ] Can perform complex data operations with Pandas
- [ ] Can create publication-quality visualizations
- [ ] Understand gradient descent mathematically and programmatically
- [ ] Can implement basic ML algorithm from scratch

### **Completed Deliverables:**
- [ ] 7 completed Jupyter notebooks
- [ ] 3+ mini-projects
- [ ] 1 working ML model
- [ ] Personal notes and cheat sheets
- [ ] 30+ coding exercises solved

### **Knowledge Check Questions:**
1. Can you explain why gradient descent works?
2. Can you implement matrix multiplication without NumPy?
3. Can you clean and analyze a messy dataset?
4. Can you visualize multi-dimensional data effectively?
5. Can you calculate statistics from scratch?

---

## **🎯 Daily Habits for Success**

### **Morning Routine (30 min):**
- Review yesterday's notes
- Set clear goals for the day
- Warm-up with one coding problem

### **Evening Routine (30 min):**
- Summarize key learnings
- Push code to GitHub
- Prepare questions for next day
- Update learning journal

### **Throughout the Day:**
- Take breaks every 45-50 minutes
- Document errors and solutions
- Share one learning on social media/blog

---

## **🚀 Bonus Challenges (Optional)**

1. **Performance Challenge:** Optimize your NumPy operations to be 10x faster
2. **Visualization Challenge:** Recreate a complex visualization from a research paper
3. **Math Challenge:** Derive the gradient descent update rule for logistic regression
4. **Data Challenge:** Find and analyze a unique dataset from your area of interest
5. **Teaching Challenge:** Explain one concept to someone else (blog post, video, or peer)

---

## **📚 Additional Resources for Week 1**

### **Quick References:**
- NumPy Documentation: [numpy.org](https://numpy.org/doc/stable/)
- Pandas Cheat Sheet: [Pandas DataFrame Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- Matplotlib Gallery: [matplotlib.org/gallery](https://matplotlib.org/stable/gallery/index.html)

### **Practice Problems:**
- NumPy 100 Exercises (GitHub)
- Pandas Practice Problems (Kaggle Learn)
- Project Euler (for programming practice)

### **Community Support:**
- Stack Overflow tags: [numpy], [pandas], [matplotlib]
- Reddit: r/learnmachinelearning
- Discord: Python Discord #data-science channel

Remember: **Quality over quantity**. It's better to deeply understand fewer concepts than to superficially cover everything. If you need more time on a topic, adjust the schedule accordingly.
