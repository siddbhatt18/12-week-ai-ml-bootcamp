Below is a 12‑week, 80/20‑focused machine learning study plan plus 5 projects (from beginner to advanced). The idea is:

- Focus on the core 20%: basic math intuition, Python, data handling, supervised learning (regression/classification), model evaluation, and a bit of deep learning.
- Constant coding: you’ll write code every week, not just read theory.
- Projects: you’ll start simple and gradually combine more concepts.

Assumptions:
- You know basic programming concepts but are not yet comfortable with ML/Python libraries.
- You can study ~8–10 hours/week (adjust up/down as needed).

---

## Overview of Core Concepts (the “20%”)

These will appear again and again in the plan:

1. **Python + ML Stack**: NumPy, pandas, Matplotlib/Seaborn, scikit‑learn.
2. **Data workflow**: loading, cleaning, splitting into train/validation/test.
3. **Core algorithms**:
   - Linear & logistic regression
   - k‑Nearest Neighbors
   - Decision trees & Random Forest
   - Gradient boosted trees (e.g., XGBoost / LightGBM / GradientBoosting in sklearn)
   - Basic neural networks (with Keras/PyTorch)
4. **Metrics & evaluation**:
   - Regression: MSE/RMSE, MAE, R²
   - Classification: accuracy, precision, recall, F1, ROC‑AUC
   - Cross‑validation, overfitting, train/test leakage
5. **Feature engineering & preprocessing**:
   - Handling missing data
   - Encoding categorical variables
   - Scaling/normalization
6. **Model improvement**:
   - Regularization (L1/L2)
   - Hyperparameter tuning
   - Bias–variance trade‑off
7. **Basic deployment mindset**:
   - Saving/loading models
   - Using a trained model in a script or simple app

---

## 12‑Week Plan (Week by Week)

For each week:
- **Topics** – what you learn
- **Practice** – specific activities
- **Outcome** – what you should be able to do

### Week 1 – Python & ML Environment Setup

**Topics**
- Install Python, Jupyter Lab/Notebook (or VS Code), and core libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- Python for data science refresh:
  - Lists, dicts, loops, functions
  - Using `pip` / `conda`
- Intro to Jupyter notebooks

**Practice**
- Install Anaconda or Miniconda.
- Create a new environment: `conda create -n ml python=3.11`
- Install libraries in that env.
- Write a notebook that:
  - Imports `numpy`, `pandas`, `matplotlib.pyplot`
  - Creates a small NumPy array and computes mean/standard deviation.
  - Loads a tiny CSV (e.g., from Kaggle or a sample dataset) using `pandas.read_csv`.
  - Plots a simple line chart and histogram.

**Outcome**
- Comfort running code in Jupyter.
- Able to load, inspect, and visualize simple tabular data.

---

### Week 2 – Data Handling & Exploratory Data Analysis (EDA)

**Topics**
- pandas essentials:
  - `DataFrame`, `Series`, indexing, filtering (`loc`, `iloc`)
  - Handling missing values: `isna`, `fillna`, `dropna`
  - Basic transformations: `apply`, `groupby`, `merge`
- EDA basics:
  - Summary statistics: `.describe()`
  - Plotting distributions, scatter plots, correlation matrix
- Train/test split concept

**Practice**
- Download a real dataset (e.g., **California Housing**, **Titanic**, or **Ames Housing**).
- In a notebook:
  - Inspect columns, data types, missing values, target variable.
  - Visualize:
    - Histograms of main features
    - Scatter plots of key features vs target
    - Correlation heatmap
- Write a small function to:
  - Clean basic issues (e.g., drop impossible values, simple imputations).

**Outcome**
- Able to take a raw CSV and understand it: structure, basic patterns, and issues.
- Understand why splitting into train/test matters (no peeking at test data).

---

### Week 3 – Supervised Learning Basics: Linear Regression

**Topics**
- Supervised learning concept: input features → target.
- Linear regression intuition:
  - Best fit line, minimizing squared error.
  - Overfitting vs underfitting (at a basic level).
- Using scikit‑learn:
  - `fit`, `predict`, `score`
  - `train_test_split`
- Regression metrics: MSE, RMSE, MAE, R²

**Practice**
- On a regression dataset (e.g., California Housing, Ames Housing):
  - Split into train/test.
  - Train `LinearRegression` from sklearn.
  - Evaluate with RMSE and R² on train and test.
- Play with:
  - Dropping/adding features and seeing performance changes.
  - Manually normalizing some numerical features and seeing if it affects performance.

**Outcome**
- Able to train and evaluate a simple regression model.
- Understand what loss/error means and why lower is better.

---

### Week 4 – Classification Basics: Logistic Regression & k‑NN

**Topics**
- Classification vs regression.
- Logistic regression intuition:
  - Outputs probability; decision boundary.
- k‑Nearest Neighbors (k‑NN) intuition:
  - Instance‑based; choice of k.
- Classification metrics:
  - Accuracy, confusion matrix, precision, recall, F1 score
  - Basic idea of ROC and AUC
- Class imbalance (only conceptually for now)

**Practice**
- Use a binary classification dataset:
  - e.g., Titanic survival or Breast Cancer dataset (`sklearn.datasets.load_breast_cancer`).
- Train:
  - `LogisticRegression`
  - `KNeighborsClassifier` with different k values
- Evaluate:
  - Accuracy, confusion matrix
  - Precision, recall, F1 using `classification_report`
- Compare models and think:
  - Which does better? Why might that be?

**Outcome**
- Able to build simple classification models and understand basic metric trade‑offs.

---

### Week 5 – Feature Engineering & Preprocessing Pipelines

**Topics**
- Types of features:
  - Numerical vs categorical.
- Preprocessing techniques:
  - Scaling (StandardScaler, MinMaxScaler).
  - One‑hot encoding (`OneHotEncoder`).
  - Handling missing values with `SimpleImputer`.
- Using `ColumnTransformer` and `Pipeline` in scikit‑learn:
  - Keeping preprocessing steps linked with the model.

**Practice**
- Pick either a regression or classification dataset with both numeric and categorical features (Titanic or Ames Housing is ideal).
- Build a `Pipeline` that:
  - Imputes missing values.
  - Scales numeric features.
  - One‑hot encodes categorical features.
  - Trains a model (e.g., LogisticRegression or RandomForestRegressor).
- Compare performance with/without a proper pipeline.

**Outcome**
- Comfortable building a single pipeline that goes from raw data to predictions.
- Understand that proper preprocessing is crucial and should be done only with training data (no data leakage).

---

### Week 6 – Trees, Random Forests & Overfitting

**Topics**
- Decision trees:
  - Splitting criteria, depth, overfitting.
- Random forests:
  - Ensemble of trees, bagging, variance reduction.
- Overfitting in more depth:
  - High training accuracy, poor test accuracy.
  - Controlling tree depth, min samples per split/leaf.

**Practice**
- On both a regression and classification dataset:
  - Train `DecisionTreeRegressor` / `DecisionTreeClassifier`.
  - Check training vs test performance.
  - Increase/decrease `max_depth` and see what happens.
- Train `RandomForestRegressor` / `RandomForestClassifier`:
  - Compare to single tree.
  - Tune basic hyperparameters: `n_estimators`, `max_depth`.

**Outcome**
- Understand trees’ flexibility and why ensembles like random forests often perform better.
- Recognize overfitting from metrics.

---

### Week 7 – Gradient Boosting & Intro to Hyperparameter Tuning

**Topics**
- Gradient boosting intuition:
  - Sequential trees correcting previous errors.
- Using:
  - `GradientBoostingClassifier/Regressor` or `XGBoost` / `LightGBM` if you install them.
- Hyperparameter tuning:
  - Manual grid search.
  - `GridSearchCV` and `RandomizedSearchCV`.

**Practice**
- On one dataset you now know well:
  - Train a gradient boosting model.
  - Use `RandomizedSearchCV` to tune a few hyperparameters:
    - learning_rate
    - n_estimators
    - max_depth
- Compare tuned model to default and to random forest.

**Outcome**
- Know how to improve performance through structured hyperparameter search.
- Have intuition that boosting is often a top performance baseline in tabular data.

---

### Week 8 – Model Evaluation in Depth & Cross‑Validation

**Topics**
- Train/validation/test splitting strategy.
- Cross‑validation:
  - k‑fold CV
  - Why it gives more robust estimates.
- Bias–variance tradeoff:
  - High bias vs high variance.
- Basic model comparison:
  - How to fairly compare models using same train/test splits.

**Practice**
- Take 1 regression and 1 classification problem:
  - Use `cross_val_score` to compare:
    - Linear regression vs random forest vs gradient boosting (regression).
    - Logistic regression vs random forest vs gradient boosting (classification).
- Plot CV scores (e.g., boxplots or just mean ± std).
- Reflect on:
  - Which model is more stable?
  - Any sign of overfitting during cross‑validation?

**Outcome**
- Able to systematically evaluate and compare several models.
- Better feel for how to choose a “good enough” model.

---

### Week 9 – Intro to Neural Networks (for Tabular or Simple Image Data)

**Topics**
- Neural network basics:
  - Perceptron, hidden layers, activation functions (ReLU, sigmoid).
  - Loss function and gradient descent (high level).
- Framework:
  - Keras (TensorFlow) or PyTorch (pick one).
- Applying a small MLP (multi‑layer perceptron) to:
  - Tabular data, or
  - MNIST digits (simple image classification).

**Practice**
- Install TensorFlow or PyTorch.
- Using Keras (for example):
  - Build a simple sequential model:
    - Input → Dense(64, relu) → Dense(64, relu) → Dense(1 or #classes, appropriate activation).
  - Train on:
    - Either: your favorite tabular dataset (classification).
    - Or: MNIST (handwritten digits, built into Keras).
  - Track training and validation accuracy/loss.
- Try:
  - Different numbers of layers/neurons.
  - Early stopping callback to reduce overfitting.

**Outcome**
- Know how to define, train, and evaluate a basic neural network.
- Conceptual understanding of how deep learning fits into the ML toolkit.

---

### Week 10 – End‑to‑End Workflow & Model Interpretability

**Topics**
- End‑to‑end ML workflow:
  - Problem definition, EDA, baseline, model iterations, evaluation, and simple deployment plan.
- Basic interpretability:
  - Feature importances (trees).
  - Permutation importance.
  - Partial dependence plots (if time).
- Saving and loading models.

**Practice**
- Take one dataset you haven’t used yet (new Kaggle dataset).
- Perform:
  1. Problem framing (what is the target? what metric matters?).
  2. EDA.
  3. Baseline model (e.g., logistic regression or random forest).
  4. Improved model (boosted trees or tuned forest).
  5. Interpretability:
     - Plot feature importances.
     - Describe top 3–5 features and how they influence predictions.
  6. Save model (e.g., `joblib.dump`) and reload in a new script to run predictions.

**Outcome**
- Able to execute a full ML project from raw data to a usable model with explanations.

---

### Week 11 – Working on a Larger Project & Handling Real‑World Issues

**Topics**
- Messy data:
  - More missing values, weird encodings, outliers.
- Data leakage examples and how to avoid them.
- Documentation and versioning of experiments.

**Practice**
- Start your **4th project** (see below – an intermediate/advanced tabular ML project).
- Focus on:
  - Careful data cleaning.
  - Creating a robust pipeline.
  - Doing proper train/validation/test splits.
  - Documenting what you try, what works, what doesn’t.
- Track:
  - Best metrics achieved.
  - Which design choices helped.

**Outcome**
- Experience dealing with more realistic data and writing cleaner, reusable code.

---

### Week 12 – Putting It All Together & Portfolio Prep

**Topics**
- Reviewing what you’ve learned.
- Writing up project reports or blog posts.
- Optional: simple deployment:
  - Using a model in a small script or minimal web app (e.g., Flask/FastAPI/Streamlit).

**Practice**
- Polish 2–3 of your best projects:
  - Clean up notebooks/code.
  - Add README with:
    - Problem statement
    - Data description
    - Approach
    - Results (with metrics and plots)
    - Future work or limitations
- Optional:
  - Use Streamlit to create a simple UI where you can input features and get a prediction.

**Outcome**
- A small portfolio that demonstrates applied ML skills.
- Solid foundation to keep learning more advanced topics independently.

---

## 5 Projects (Beginner → Advanced)

You can spread these over the 12 weeks as indicated.

---

### Project 1 (Beginner) – House Price Prediction (Linear Regression)

**When:** Weeks 3–4

**Description**
Use a housing dataset (Kaggle “House Prices: Advanced Regression Techniques” or simpler California Housing). Train a linear regression model to predict house prices based on features like square footage, number of rooms, etc.

**Steps (outline, not full code)**
1. Load dataset, inspect columns.
2. Perform basic EDA: distributions, correlations.
3. Handle missing values simply (drop or basic impute).
4. Encode categorical variables (one‑hot encoding).
5. Train/test split.
6. Train `LinearRegression`.
7. Evaluate using RMSE and R².
8. Write a short summary: which features seem most important? How good is the performance?

**Key Concepts Reinforced**
- Data loading and EDA
- Train/test split
- Regression model training and basic evaluation
- One‑hot encoding and handling missing values

---

### Project 2 (Beginner–Intermediate) – Titanic Survival Classification

**When:** Weeks 4–5

**Description**
Use the classic Titanic dataset from Kaggle to predict whether a passenger survived. Start with logistic regression, then try k‑NN or a tree‑based model.

**Steps**
1. Load data; understand each feature (Passenger class, Age, Sex, etc.).
2. Clean:
   - Impute missing ages and fares.
   - Convert categorical features with one‑hot encoding.
3. Create preprocessing + model `Pipeline`.
4. Train logistic regression, evaluate with:
   - Accuracy
   - Confusion matrix
   - Precision, recall, F1
5. Try:
   - A `RandomForestClassifier` in the same pipeline.
   - Compare metrics to logistic regression.

**Key Concepts Reinforced**
- Classification vs regression
- Logistic regression and random forests
- Pipelines and preprocessing
- Classification metrics and trade‑offs

---

### Project 3 (Intermediate) – Customer Churn Prediction (Classification with Imbalanced Data)

**When:** Weeks 6–8

**Description**
Pick a customer churn dataset (e.g., “Telco Customer Churn” on Kaggle). Predict whether a customer will leave the service. Churn datasets are often imbalanced, so you must think carefully about metrics beyond accuracy.

**Steps**
1. Understand features (contract type, monthly charges, tenure, etc.).
2. Clean and preprocess:
   - Handle missing values.
   - Encode categoricals.
   - Scale numeric features if needed.
3. Use `train_test_split` (or train/val/test split).
4. Train a baseline logistic regression.
5. Check class balance:
   - How many churn vs non‑churn?
   - Use accuracy, precision, recall, F1, ROC‑AUC.
6. Try:
   - Random forest
   - Gradient boosting (e.g., `XGBClassifier` or `GradientBoostingClassifier`)
   - Compare metrics using cross‑validation.
7. (Optional) Handle imbalance:
   - Class weights (`class_weight='balanced'`) or simple resampling.

**Key Concepts Reinforced**
- Handling imbalanced datasets
- More practical feature engineering and preprocessing
- Tree‑based models and gradient boosting
- Cross‑validation and model comparison

---

### Project 4 (Intermediate–Advanced) – Tabular ML Competition-Style Project

**When:** Weeks 9–11

**Description**
Pick a more complex tabular dataset from Kaggle (e.g., a competition that’s now finished). The goal is not to rank highly but to simulate a real ML workflow and push your skills: robust pipelines, hyperparameter tuning, interpretation.

**Steps**
1. Read the competition description:
   - What’s the target?
   - Which metric is used (RMSE, AUC, log loss, etc.)?
2. EDA:
   - Identifying data types, missingness, outliers.
   - Domain understanding from feature descriptions.
3. Build a strong baseline:
   - Random forest or gradient boosting with default params.
4. Improve:
   - Proper `ColumnTransformer`+`Pipeline`.
   - Hyperparameter tuning with `RandomizedSearchCV`.
   - Cross‑validation to select the best model.
5. Interpret:
   - Feature importances.
   - Simple permutation importance.
6. Document:
   - Decisions, experiments, final results.

**Key Concepts Reinforced**
- End‑to‑end ML project flow
- Cross‑validation and hyperparameter tuning in a structured way
- Interpreting models and explaining results
- Thinking about the problem’s metric and business impact

---

### Project 5 (Advanced) – Simple Neural Network Project (MNIST or Tabular NN)

**When:** Weeks 9–12

**Option A: MNIST Handwritten Digit Classification**

**Description**
Use the MNIST dataset (28x28 grayscale images of digits 0–9) to train a neural network (or simple CNN) and classify digits.

**Steps**
1. Load MNIST via Keras datasets.
2. Preprocess:
   - Normalize pixel values to [0,1].
   - One‑hot encode labels.
3. Define a Keras model:
   - Simple dense network: flatten → Dense(128, relu) → Dense(10, softmax).
   - Or small CNN (Conv2D → MaxPooling2D → Dense).
4. Train with train/validation split; use early stopping.
5. Evaluate test accuracy and confusion matrix.
6. Visualize:
   - Some misclassified examples.
   - Training vs validation loss/accuracy curves.

**Option B: Neural Network for Tabular Churn or House Prices**

Same idea, but with tabular input instead of images.

**Key Concepts Reinforced**
- Neural network architectures and training loop
- Overfitting and regularization via early stopping, dropout
- Interpreting training curves and validation performance
- Extending ML skills beyond scikit‑learn into deep learning frameworks

---

## How to Use This Plan Effectively

- **Week structure**: Aim for a cycle of:
  - 40% reading/watching tutorials
  - 60% coding and experimenting
- **Struggle deliberately**:
  - Before looking up an answer, try to debug and experiment on your own.
  - When something breaks, ask: “What does this error really mean?”
- **Maintain a learning log**:
  - After each week, write a short note:
    - What you learned
    - What you struggled with
    - One thing you want to revise later

If you tell me your current math/programming comfort level and how many hours/week you realistically have, I can adjust this plan to be slightly lighter or heavier and suggest specific resources (courses/books) to pair with each week.
