Below is a 7-day plan for Week 2, focused on **deeper EDA, visualization, and basic feature handling** on real datasets (mainly Titanic, but you can swap in another tabular dataset). The aim is to move from “I can use pandas” to “I can investigate questions with data.”

Assume ~1.5–2.5 hours/day. If you have more time, expand the exercises rather than skipping ahead.

You’ll reuse skills from Week 1 and push yourself to:
- Ask questions about the data.
- Choose and build visualizations to answer them.
- Interpret and communicate your findings.

---

## Day 1 – EDA Mindset & Question-Driven Analysis

**Objectives:**
- Move from random plotting to **question-driven** EDA.
- Practice framing EDA around goals, not tools.

### 1. Setup

- Create notebook: `week2_day1_eda_questions.ipynb`.
- Load Titanic (or another dataset you like) as `df`.

### 2. Define Questions Before Plotting

In a **Markdown cell**, write down 5–10 questions about Titanic. Examples (don’t just copy; adjust or invent your own):

- Does gender affect survival?
- Does passenger class (Pclass) affect survival?
- Is there a relationship between age and survival?
- Do families (people with siblings/spouses or parents/children aboard) have different survival rates from solo travelers?
- Did people who paid higher fares tend to survive more?

Try to distinguish:
- **Univariate** questions (“What’s the distribution of Age?”).
- **Bivariate** questions (“How does survival vary with Age?”).
- **Multivariate** questions (“How does survival vary with both Pclass and Sex?”).

### 3. Plan Your Approach (Thinking Exercise)

For each question, in Markdown:
- List:
  - Which columns do you need?
  - Are they numeric or categorical?
  - What simple plots/tables might help? (histogram? bar chart? groupby summary?)

Don’t code yet; think:
- For survival vs sex → groupby + bar plot.
- For age distribution → histogram/boxplot.
- For Pclass & survival → stacked bar or grouped bar.

This is **deliberate planning**, which is an important habit in ML.

### 4. Implement 2–3 Questions

For each selected question:
1. Write the question as a Markdown heading.
2. Implement your chosen analysis:
   - `groupby`, `value_counts`, etc.
   - 1–2 plots that address it.
3. After each plot, write **2–4 sentences** interpreting it.

**Mini Challenge (thinking)**  
For one question, build **two different** visualizations that address it (e.g., countplot and line plot; or bar chart of rates and boxplot) and compare which one communicates the insight better, and why.

---

## Day 2 – Categorical Data, Groupby, and Aggregations

**Objectives:**
- Get comfortable with `groupby` (crucial for thinking in terms of distributions and segments).
- Explore categorical features in depth.

### 1. New Notebook

- Name: `week2_day2_groupby_categorical.ipynb`.
- Load `df` again.

### 2. Review Key Methods

Search (skim, don’t deep read everything):
- `pandas groupby tutorial`
- `pandas value_counts normalize`

### 3. Practice with Groupby

Use Titanic examples (or your dataset):

1. Survival rate by sex:
   ```python
   df.groupby("Sex")["Survived"].mean()
   ```
2. Survival rate by Pclass:
   ```python
   df.groupby("Pclass")["Survived"].mean()
   ```
3. Combined grouping:
   ```python
   df.groupby(["Pclass", "Sex"])["Survived"].mean()
   ```
4. Average age by Pclass and Sex:
   ```python
   df.groupby(["Pclass", "Sex"])["Age"].mean()
   ```

Write short text interpretations.

### 4. Value Counts and Normalized Frequencies

1. Frequency of each port of embarkation:
   ```python
   df["Embarked"].value_counts()
   df["Embarked"].value_counts(normalize=True)
   ```
2. For each Pclass, what fraction of passengers embarked at each port?  
   Hint:
   ```python
   df.groupby("Pclass")["Embarked"].value_counts(normalize=True)
   ```

Interpretation questions:
- Does a particular class tend to embark more from a specific port?

### 5. Mini Challenges (Thinking)

1. For each `Sex`, compute **average fare** and **average age**:
   ```python
   df.groupby("Sex")[["Fare", "Age"]].mean()
   ```
   - Explain any patterns you see.

2. Create a small **summary table** combining multiple aggregations:
   ```python
   df.groupby("Pclass").agg(
       avg_age=("Age", "mean"),
       avg_fare=("Fare", "mean"),
       surv_rate=("Survived", "mean")
   )
   ```
   - Interpret: how do class, age, fare, and survival rate relate?

3. Invent at least **one** groupby question not listed here and answer it.

---

## Day 3 – Numerical Distributions & Outliers (Histograms, Boxplots)

**Objectives:**
- Understand how to inspect numeric distributions.
- Get a feel for skewness, outliers, and transformations.

### 1. Notebook

- Name: `week2_day3_numerical_distributions.ipynb`.

### 2. Histograms & KDE Plots

Imports:
```python
import seaborn as sns
import matplotlib.pyplot as plt
```

**Practice:**

1. Age distribution:
   ```python
   sns.histplot(df["Age"], bins=30, kde=True)
   plt.show()
   ```
   - Are there missing values? (`df["Age"].isna().sum()`)

2. Fare distribution:
   ```python
   sns.histplot(df["Fare"], bins=40, kde=True)
   plt.show()
   ```
   - Is it skewed? Are there very large values?

3. Try log-transform for Fare (avoid log(0) or negatives):
   ```python
   import numpy as np
   df["Fare_log"] = np.log1p(df["Fare"])  # log(1+Fare)
   sns.histplot(df["Fare_log"], bins=40, kde=True)
   plt.show()
   ```
   - Compare shapes before and after log.

### 3. Boxplots & Outliers

1. Boxplot for `Fare`:
   ```python
   sns.boxplot(x=df["Fare"])
   plt.show()
   ```

2. Boxplot of `Fare` across `Pclass`:
   ```python
   sns.boxplot(data=df, x="Pclass", y="Fare")
   plt.show()
   ```
   - What differences do you see?

3. Boxplot for Age vs Survived:
   ```python
   sns.boxplot(data=df, x="Survived", y="Age")
   plt.show()
   ```
   - Any visible difference in age distribution?

### 4. Mini Challenges (Thinking)

1. For each `Pclass`, calculate:
   - Median fare
   - 25th and 75th percentiles
   ```python
   df.groupby("Pclass")["Fare"].describe()
   ```
   - How do these summary stats relate to your boxplots?

2. Identify **possible outliers** in Fare:
   - Decide a cutoff, e.g., Fare > 200.
   ```python
   high_fare = df[df["Fare"] > 200]
   high_fare[["Fare", "Pclass", "Survived"]]
   ```
   - What kind of passengers are these?

3. Think: If you were to build a model later, would you:
   - Keep extreme Fare values as-is?
   - Transform (log)?  
   - Cap/clip them?

Write a short paragraph with your reasoning.

---

## Day 4 – Relationships Between Variables (Scatter, Pairplots, Heatmaps)

**Objectives:**
- Visualize **relationships** between numerical features.
- Use correlation matrices, scatter plots, and pairplots.

### 1. Notebook

- Name: `week2_day4_relationships.ipynb`.

### 2. Correlation Matrix

1. Select numeric columns:
   ```python
   numeric_cols = ["Age", "Fare", "Survived", "Pclass", "SibSp", "Parch"]
   corr = df[numeric_cols].corr()
   corr
   ```
2. Visualize:
   ```python
   plt.figure(figsize=(8, 6))
   sns.heatmap(corr, annot=True, cmap="coolwarm")
   plt.show()
   ```

Interpretation:
- Which pairs have strong positive/negative correlation?
- Is Survived correlated strongly with any numeric feature?

### 3. Scatter Plots and Joint Plots

1. Scatter Age vs Fare:
   ```python
   sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived")
   plt.show()
   ```
2. Joint distribution (if dataset size permits):
   ```python
   sns.jointplot(data=df, x="Age", y="Fare", kind="hex")
   plt.show()
   ```

Think:
- Do Age and Fare have any clear pattern?
- Do survivors cluster differently in this plot?

### 4. Pairplots (Carefully)

To avoid clutter, select a subset:
```python
sns.pairplot(df[["Age", "Fare", "Pclass", "Survived"]], hue="Survived")
plt.show()
```
If it’s slow or unclear, limit further, e.g., only Age, Fare, Survived.

### 5. Mini Challenges (Thinking)

1. Choose **two numeric features** that appear somewhat correlated or interesting from the heatmap and:
   - Make specific scatter plots, maybe separated by a third variable (e.g., hue = Sex or Pclass).
   - Interpret any visible trend.

2. Think of a scenario: If Survived is the target variable, which **two or three features** seem most promising to use, based on:
   - Correlations.
   - Visualizations you’ve made so far.

Explain your reasoning in 5–10 sentences.

---

## Day 5 – Categorical vs Categorical & Feature Combinations

**Objectives:**
- Explore interactions between **multiple categorical features**.
- Start to think in terms of **feature combinations** that may be predictive.

### 1. Notebook

- Name: `week2_day5_categorical_interactions.ipynb`.

### 2. Crosstabs and Pivot Tables

1. Cross-tab between Sex and Survived:
   ```python
   pd.crosstab(df["Sex"], df["Survived"])
   ```
2. Proportions:
   ```python
   pd.crosstab(df["Sex"], df["Survived"], normalize="index")
   ```
   - Interpret survival rates by sex.

3. Extend to Pclass and Sex:
   ```python
   pd.crosstab([df["Pclass"], df["Sex"]], df["Survived"], normalize="index")
   ```
   - Which combination has highest survival?

### 3. Visualizing Categorical Interactions

1. Countplot Pclass vs Survived:
   ```python
   sns.countplot(data=df, x="Pclass", hue="Survived")
   plt.show()
   ```

2. Survival by Sex and Pclass together:
   - Option 1: `catplot`:
     ```python
     sns.catplot(data=df, x="Pclass", hue="Sex", col="Survived", kind="count")
     plt.show()
     ```
   - Option 2: barplot with `estimator=np.mean`:
     ```python
     import numpy as np
     sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", estimator=np.mean)
     plt.show()
     ```

Interpretation:
- Where are survival rates highest/lowest?
- Do sex and class interact (e.g., female 1st class vs male 3rd class)?

### 4. Simple Feature Engineering Idea (Conceptual)

Without coding too much yet, think about:
- Combining SibSp and Parch into a **family_size** feature:
  ```python
  df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
  ```
- Compare survival rates by FamilySize:
  ```python
  df.groupby("FamilySize")["Survived"].mean()
  ```

Think:
- Are very small families (1 person) or very large families at an advantage or disadvantage?

### 5. Mini Challenges (Thinking)

1. Invent and compute at least **one new derived feature** (e.g., Fare per person, Age bucket like Child/Adult/Senior).
2. Create a plot showing how Survived varies with this new feature.
3. Write a short paragraph: **why might this feature be useful later in a prediction model?**

---

## Day 6 – Handling Missing Data & Simple Imputation

**Objectives:**
- Understand how to **detect and handle missing values**.
- Practice simple imputation strategies and see how they change distributions.

### 1. Notebook

- Name: `week2_day6_missing_values.ipynb`.

### 2. Identify Missing Values

1. Recompute:
   ```python
   df.isna().sum()
   ```
2. Focus on columns with missing data (Age, Cabin, Embarked in Titanic).

### 3. Simple Imputation

For Age (as an example):

1. Compare distributions:
   - Drop missing Age:
     ```python
     age_no_na = df["Age"].dropna()
     sns.histplot(age_no_na, kde=True)
     plt.show()
     ```
2. Mean imputation:
   ```python
   df["Age_mean_imputed"] = df["Age"].fillna(df["Age"].mean())
   sns.histplot(df["Age_mean_imputed"], kde=True)
   plt.show()
   ```
3. Median imputation:
   ```python
   df["Age_median_imputed"] = df["Age"].fillna(df["Age"].median())
   sns.histplot(df["Age_median_imputed"], kde=True)
   plt.show()
   ```

Compare shapes and think about pros/cons.

### 4. Categorical Missing and Modes

1. For Embarked:
   ```python
   df["Embarked"].value_counts(dropna=False)
   ```
2. Fill missing Embarked with mode:
   ```python
   most_common_port = df["Embarked"].mode()[0]
   df["Embarked_imputed"] = df["Embarked"].fillna(most_common_port)
   ```

### 5. Mini Challenges (Thinking)

1. For Age, implement **group-specific imputation**:
   - For each Pclass, compute mean Age and impute missing Age with the mean of that Pclass.
   ```python
   df["Age_group_imputed"] = df["Age"]
   for pclass, group in df.groupby("Pclass"):
       mean_age = group["Age"].mean()
       mask = (df["Pclass"] == pclass) & (df["Age_group_imputed"].isna())
       df.loc[mask, "Age_group_imputed"] = mean_age
   ```
   - Compare histograms across imputation strategies.

2. In Markdown, reflect:  
   If you later build a model:
   - Why is it important to handle missing values?
   - What risks come from poor imputation choices?

---

## Day 7 – Week 2 Mini-Project: Structured EDA Report

**Objectives:**
- Practice building a **cohesive EDA report**, not just disjointed code.
- Combine question-driven analysis, visualization, and basic feature transformations.

### 1. Notebook

- Name: `week2_eda_report_titanic.ipynb` (or use another dataset if you prefer, but Titanic is fine).

### 2. Structure Your Report

Use Markdown headings to create this structure:

1. **Introduction & Objectives**
   - Describe dataset and your EDA goals this week:
     - Understand patterns related to survival.
     - Identify potentially useful features for modeling.

2. **Data Loading & Overview**
   - Load CSV.
   - Show `df.head()`, `df.info()`, missing values summary.
   - 2–3 bullet points summarizing dataset shape and obvious issues (missing values, etc.).

3. **Univariate Analysis**
   - Key numeric: Age, Fare, FamilySize (if created).
   - Key categorical: Sex, Pclass, Embarked.
   - Summaries:
     - Histograms/boxplots for numeric.
     - Value counts/plots for categorical.
   - 1–2 sentences after each group of plots.

4. **Bivariate & Multivariate Analysis**
   - Survival vs Sex, Pclass, Age, Fare, and at least one **derived feature** (e.g., FamilySize, or Age bucket).
   - Use:
     - `groupby` summary tables.
     - Barplots, boxplots, and maybe one pairplot.
   - A couple of crosstabs (e.g., Pclass & Sex vs Survived).

5. **Missing Data Handling Experiment**
   - Briefly show:
     - Where missing values are.
     - One simple imputation strategy for Age or Embarked.
   - Show how the distribution changes (at least one “before vs after” plot).
   - Very short reflection on which strategy you prefer and why.

6. **Key Findings & Hypotheses for Modeling**
   - 8–12 bullet points summarizing:
     - Which features strongly relate to survival?
     - What new features you created and why you think they might help.
     - Any surprising or counterintuitive patterns.
   - A short paragraph imagining:
     - If you had to build a `Survived` prediction model tomorrow, which 5–10 features would you start with, and why?

### 3. Thinking/Stretch Tasks

1. Try to **limit yourself** to only the most useful plots:
   - Do not spam visualizations.
   - Focus on ones that directly answer your questions.

2. Pretend your audience is:
   - A colleague who knows basic statistics but not the dataset.
   - They should be able to read your notebook top-to-bottom and understand what you did and what you found.

3. If you have extra time:
   - Repeat the same structured EDA on **a different small dataset** (e.g., another Kaggle tabular dataset).
   - You’ll build “EDA muscle memory” and generalize beyond Titanic.

---

## How to Know You’re Ready for Week 3

By the end of Week 2 you should be able to:

- Start EDA by **writing questions** first, not code.
- Use `groupby`, `value_counts`, crosstabs, and aggregations comfortably.
- Interpret histograms, boxplots, scatter plots, heatmaps, and countplots.
- Reason about missing values and basic imputation choices.
- Create simple derived features and investigate whether they matter.
- Produce a coherent EDA notebook that tells a story and leads naturally to the modeling step.

If any of these feel weak, you can:
- Take your Week 7 mini-project and redo it on a new dataset.
- Or pick 3 of your Week 2 questions and deliberately try to answer them with **different** plots or tables than you used the first time.

If you’d like next, I can produce a similarly structured daily plan for **Week 3 (core ML workflow + linear regression)**, including a small regression mini-project.
