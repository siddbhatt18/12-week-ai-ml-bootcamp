Here is your 7‑day plan for **Week 11**, focused on:

- **Error analysis** (finding where and why your models fail)
- **Model comparison** (beyond “which metric is higher?”)
- **Interpretation & communication** (turning work into portfolio‑ready case studies)

You’ll mostly reuse your strongest **classification** and **regression** projects (e.g., Titanic + California Housing or your Week 7 Kaggle dataset). Assume ~1.5–2.5 hours/day.

---

## Overall Week 11 Goals

By the end of this week, you should be able to:

- Systematically analyze **errors** by segment (feature slices) and instance.
- Compare models along multiple axes: performance, stability, interpretability, cost.
- Use simple tools for **interpretability** (feature importance, partial dependence, simple SHAP intro if you want).
- Turn one or two of your projects into a **clean, story‑driven case study**.

---

## Day 1 – Framing Error Analysis & Choosing Focus Projects

**Objectives:**
- Understand what **error analysis** is and why it matters.
- Choose 1–2 projects to focus on for the rest of the week.

### 1. Notebook

Create: `week11_day1_error_analysis_overview.ipynb`.

### 2. Conceptual Notes (Markdown)

In your words, define:

1. **Error analysis**
   - Systematic inspection of where and how the model fails.
   - Goes beyond a single global accuracy/RMSE number.
2. Why it matters:
   - Helps you improve the model (targeted fixes).
   - Prevents misleading conclusions (e.g., great overall but terrible for some group).
   - Essential for trust, fairness, and deployment decisions.

3. Dimensions of error analysis:
   - By **feature slices** (e.g., age group, price range, region).
   - By **instance type** (edge cases, noisy data).
   - By **model comparison** (where models disagree).

### 3. Choose Your Focus Projects

Pick:

- **One classification project** (likely best: Titanic or your Week 7 classification dataset, with tuned pipeline).
- **One regression project** (e.g., California Housing or house price project with Ridge/Forest/GB).

In Markdown, for each project, briefly note:

- Dataset name + task (e.g., “Titanic survival classification”).
- Which **final model** you’re focusing on (e.g., tuned RandomForest, tuned GradientBoosting, or best pipeline from Week 10).
- One or two **baseline/comparison models** (e.g., logistic vs RF vs GB, linear vs RF).

### 4. Load Best Models & Data

In code, for each project:
- Load data (X, y).
- Recreate or load the **best pipeline/model** you trained previously.
- Split into train/test as you used before.

You want:

- `X_train`, `X_test`, `y_train`, `y_test`
- `best_model` (pipeline)
- Optional: `other_model` for comparison.

### 5. Quick Sanity Check

Re‑compute key metrics for best_model on test:

- Classification: accuracy, precision, recall, F1, ROC AUC.
- Regression: RMSE, MAE, R².

Store them in variables; you’ll refer back to these throughout the week.

### 6. Thinking Challenge

In Markdown:

- For each project:
  - What are you **most worried about** in terms of errors? (e.g., misclassifying high‑risk loans, underestimating expensive houses).
  - Which **user/stakeholder** would be hurt most by bad predictions?
- If you could only improve performance in one segment (e.g., young passengers, expensive houses), which would it be and why?

Write 8–12 sentences total.

---

## Day 2 – Classification Error Analysis: Slices & Confusion‑Driven Insights

**Objectives:**
- Dive deep into classification errors on your chosen classification project.
- Analyze metrics by **subgroup** and interpret confusion matrices.

### 1. Notebook

Create: `week11_day2_classification_error_analysis.ipynb`.

Load:

- `X_test`, `y_test`
- `best_model` (classification)
- Possibly some original `df_test` with raw columns (to slice by interpretable features).

### 2. Get Predictions & Base Metrics

```python
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm
```

Optional: Plot confusion matrix.

### 3. Attach Predictions Back to a DataFrame

If `X_test` is from `df`, reconstruct test DataFrame with original features:

```python
# If you have an index or split mask, recreate df_test accordingly.
df_test = ...  # DataFrame containing same rows as X_test, with original columns

df_test = df_test.copy()
df_test["y_true"] = y_test.values
df_test["y_pred"] = y_pred
df_test["correct"] = (df_test["y_true"] == df_test["y_pred"]).astype(int)
```

### 4. Slice‑Wise Metrics

Pick 2–3 key features to slice by (e.g., for Titanic: Sex, Pclass, AgeGroup).

Example:

```python
def slice_metrics(df_slice):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    if df_slice["y_true"].nunique() < 2:
        return None  # can't compute some metrics with one class only
    return {
        "n": len(df_slice),
        "accuracy": accuracy_score(df_slice["y_true"], df_slice["y_pred"]),
        "precision": precision_score(df_slice["y_true"], df_slice["y_pred"]),
        "recall": recall_score(df_slice["y_true"], df_slice["y_pred"]),
        "f1": f1_score(df_slice["y_true"], df_slice["y_pred"]),
    }
```

Then, for example:

```python
groups = df_test.groupby("Sex")
metrics_by_sex = {name: slice_metrics(group) for name, group in groups}
metrics_by_sex
```

Repeat for:

- `Pclass`
- Simple age bins (e.g., `AgeBin` you made earlier).

Compare each slice’s metrics vs the **overall** metrics.

### 5. Inspect Misclassified Cases

Filter:

```python
false_negatives = df_test[(df_test["y_true"] == 1) & (df_test["y_pred"] == 0)]
false_positives = df_test[(df_test["y_true"] == 0) & (df_test["y_pred"] == 1)]
```

Look at first 10–20 rows for each; note any patterns in features (e.g., certain age/class combos).

### 6. Thinking Challenge

In Markdown:

- Which subgroups have **much worse** performance than the overall?
- Any evidence that the model is biased towards/against particular slices (e.g., certain classes, age groups)?
- From false positives/negatives:
  - Do you see any systematic pattern (e.g., consistently misclassifying some borderline group)?
- Suggest 2–3 **concrete ideas** to address at least one systematic error (e.g., add feature, adjust threshold, re‑weight classes).

Write 12–18 sentences.

---

## Day 3 – Regression Error Analysis: Residuals & Segment Performance

**Objectives:**
- Analyze **residuals** for your regression project.
- Investigate error patterns by feature ranges and target magnitude.

### 1. Notebook

Create: `week11_day3_regression_error_analysis.ipynb`.

Load:

- `Xr_test`, `yr_test`
- `best_reg_model` (e.g., RF, GB, or Ridge pipeline).
- Any original `df_reg_test` with raw features.

### 2. Predictions & Basic Residual Analysis

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

yr_pred = best_reg_model.predict(Xr_test)
residuals = yr_test - yr_pred

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
mae, rmse, r2
```

Plot residual distribution:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(residuals, bins=40, kde=True)
plt.xlabel("Residual (y_true - y_pred)")
plt.title("Residual Distribution")
plt.show()
```

Residuals vs predicted:

```python
plt.figure(figsize=(6, 4))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.show()
```

Interpret visually.

### 3. Attach Residuals to DataFrame

```python
df_reg_test = ...  # reconstruct original test set features

df_reg_test = df_reg_test.copy()
df_reg_test["y_true"] = yr_test.values
df_reg_test["y_pred"] = yr_pred
df_reg_test["residual"] = df_reg_test["y_true"] - df_reg_test["y_pred"]
df_reg_test["abs_error"] = df_reg_test["residual"].abs()
```

### 4. Error by Target Magnitude

Bin target into quantiles or ranges:

```python
df_reg_test["target_bin"] = pd.qcut(df_reg_test["y_true"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
df_reg_test.groupby("target_bin")["abs_error"].agg(["mean", "median", "count"])
```

Visualize mean abs error by bin.

Check whether the model struggles more with high vs low target values.

### 5. Error by Key Features

Pick 1–3 key features (e.g., for housing: `MedInc`, `Latitude`, `Longitude`, etc.)

For each, either:

- Plot residuals vs feature:

```python
sns.scatterplot(data=df_reg_test, x="MedInc", y="residual", alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.show()
```

- Or group into bins and aggregate errors.

```python
df_reg_test["MedInc_bin"] = pd.qcut(df_reg_test["MedInc"], q=4)
df_reg_test.groupby("MedInc_bin")["abs_error"].mean()
```

### 6. Inspect Largest Errors

```python
largest_errors = df_reg_test.sort_values("abs_error", ascending=False).head(20)
largest_errors
```

Look if outliers or data issues exist; maybe plot some of them geographically if location features exist.

### 7. Thinking Challenge

In Markdown:

- Where does your model perform **worst** (by target magnitude and feature ranges)?
- Are errors symmetric (over/underestimation) or biased one way?
- Are large errors associated with:
  - Rare feature combinations?
  - Extreme target values?
  - Missing or noisy data?

Suggest 2–3 targeted improvements (e.g., special treatment for high‑value cases, log transform, more data cleaning).

Write 12–18 sentences.

---

## Day 4 – Model Comparison: Beyond a Single Metric

**Objectives:**
- Compare multiple models on each project along several dimensions.
- Practice reasoned model selection.

### 1. Notebook

Create: `week11_day4_model_comparison.ipynb`.

### 2. Collect Models & Metrics

For each project (classification + regression):

- Gather at least 2–3 models you have trained:

Classification example:
- LogisticRegression pipeline (tuned).
- RandomForest pipeline (tuned).
- GradientBoosting pipeline.

Regression example:
- Ridge or LinearRegression pipeline.
- RandomForestRegressor pipeline.
- GradientBoostingRegressor pipeline.

Re‑evaluate each on the same test split and store metrics in a DataFrame.

Classification metrics:
- Accuracy, Precision, Recall, F1, ROC AUC.

Regression metrics:
- MAE, RMSE, R².

Example:

```python
results_cls = []
def eval_classifier(name, model, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred = model.predict(X_test)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        row["roc_auc"] = roc_auc_score(y_test, y_proba)
    return row
```

Do similarly for regression.

### 3. Add Non‑Metric Considerations

For each model, create a small table including:

- Performance (your metrics).
- **Training time/complexity** (rough notes: “fast”, “medium”, “slow”).
- **Prediction speed** (esp. for large scale).
- **Interpretability** (high, medium, low).
- **Implementation complexity** (easy/hard; number of hyperparameters).

This doesn’t need exact times; approximate is fine.

### 4. Choose “Best” Model with Justification

In Markdown, for each project:

- State which model you would choose to deploy **today**.
- Justify by weighing:
  - Performance differences.
  - Interpretability.
  - Simplicity/stability.
  - Any domain constraints (latency, fairness, explainability needs).

### 5. Thinking Challenge

- Suppose your best performing model is complex, black‑boxy, and very sensitive to hyperparameters.
- Another model is slightly worse numerically but:
  - More interpretable.
  - More stable across CV folds.
  - Easier to implement and maintain.

In 10–15 sentences, argue for or against choosing the slightly worse but simpler model, given:
- A safety‑critical setting (e.g., medical).
- A marketing segmentation setting (low stakes).

---

## Day 5 – Interpretability Tools (Feature Importance, Partial Dependence, SHAP Intro)

**Objectives:**
- Use basic interpretability techniques.
- Build intuition about how your model uses features.

You do not need to master SHAP; just get a taste.

### 1. Notebook

Create: `week11_day5_interpretability.ipynb`.

Choose one project (preferably classification with tree/forest/boosting model).

### 2. Global Feature Importance (Review)

From best tree/ensemble model:

```python
import pandas as pd
import matplotlib.pyplot as plt

final_model = best_rf_or_gb_pipeline  # pipeline
model_only = final_model.named_steps["model"]  # assuming name "model"

importances = model_only.feature_importances_
feature_names = final_model.named_steps["preprocess"].get_feature_names_out()

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.head(20).plot(kind="barh", figsize=(6, 8))
plt.gca().invert_yaxis()
plt.title("Feature Importances")
plt.show()
```

(Adjust `get_feature_names_out()` depending on your preprocessor.)

Interpret top features.

### 3. Partial Dependence Plots (PDP)

For 1–2 important features, use scikit‑learn’s partial dependence:

```python
from sklearn.inspection import PartialDependenceDisplay

features_for_pdp = [0, 1]  # or names/indices of important features after preprocessor

PartialDependenceDisplay.from_estimator(
    final_model, X, features=[features_for_pdp[0]]
)
plt.show()

PartialDependenceDisplay.from_estimator(
    final_model, X, features=[features_for_pdp[1]]
)
plt.show()
```

If your scikit‑learn version requires names or indices differently, adapt accordingly.

Interpret:

- As feature X increases, how does predicted probability or output change?

### 4. Optional SHAP Intro (High‑Level)

If you’re comfortable installing:

```bash
pip install shap
```

Then:

```python
import shap

# For tree-based models, TreeExplainer is efficient
explainer = shap.TreeExplainer(model_only)
X_sample = final_model.named_steps["preprocess"].transform(X.sample(200, random_state=42))
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
```

Treat this as a visual tool; don’t worry about all details.

### 5. Thinking Challenge

In Markdown:

- From feature importance + PD plots:
  - Which features increase predicted probability the most?
  - Any non‑intuitive relationships?
- How can interpretability:
  - Increase stakeholder trust?
  - Help you spot data leakage or spurious correlations?
- When might you **reject** a model even if metrics are strong, based on interpretability insights?

Write 12–18 sentences.

---

## Day 6 – Turning One Project into a Polished Case Study

**Objectives:**
- Take your strongest project (classification or regression).
- Turn it into a **clean, narrative‑driven** notebook or draft report.

### 1. Choose Project

Pick the one you think best showcases your skills and interest (e.g., Week 7 Kaggle tabular classification or house price regression).

### 2. Notebook

Create: `week11_case_study_project.ipynb` (or copy your final Week 10 project and refine).

### 3. Case Study Structure

Use clear sections:

1. **Problem & Context**
   - Business/problem description.
   - Why prediction matters.
   - What decisions could be informed by this model.

2. **Data Understanding**
   - Describe the dataset:
     - Rows, columns, target.
   - Brief, focused EDA:
     - Key distributions.
     - Class balance or target range.
     - Obvious data quality issues.

3. **Modeling Approach**
   - Explain:
     - Features used (including any engineered ones).
     - Preprocessing pipeline (scaling, encoding).
     - Candidate models (e.g., Logistic vs RF vs GB).
     - How you tuned hyperparameters (CV, Grid/Random search).

4. **Results**
   - Present final metrics (train, CV, test).
   - Compare major models in a table.
   - Highlight your chosen model and why.

5. **Error Analysis & Fairness**
   - Summarize what you discovered in Week 11 error analysis:
     - Key failure modes.
     - Segment‑wise performance.
   - Discuss fairness/ethical considerations relevant to the task.

6. **Interpretability**
   - Show key feature importances.
   - Provide 1–2 PD plots or SHAP insights.
   - Explain what these imply for the domain.

7. **Conclusions & Future Work**
   - Final takeaways.
   - Limitations (data, model, evaluation).
   - Concrete future steps.

Format everything so a reader can follow without jumping around; use markdown text to explain each step.

### 4. Thinking Challenge

Re‑read your notebook *as if you were someone else* (a recruiter or teammate):

- Is the **story** clear?
- Are choices justified (not just “I tried this because…” but “I chose X over Y because…”)?
- Does error analysis and interpretability appear as a natural part of the process, not an afterthought?

Write a short list (8–10 bullet points) in Markdown of things you **improved** and things you still want to polish.

---

## Day 7 – Communication Practice & Portfolio Planning

**Objectives:**
- Practice **communicating** your work in different formats.
- Plan how to present your projects in a portfolio.

### 1. Notebook or Markdown Doc

Create: `week11_communication_and_portfolio.md` (Markdown file) or a notebook with mostly text.

### 2. Write a One‑Page Project Summary

For your main case study project, write a concise, non‑technical one‑pager:

- 3–4 sentences: Problem & context.
- 3–4 sentences: Data & methods.
- 3–4 sentences: Results & impact.
- 2–3 sentences: Limitations & future work.

Keep jargon minimal; imagine a manager or recruiter skimming it.

### 3. Write a Technical Summary (For an ML‑Savvy Reader)

- 2–3 paragraphs focusing on:
  - Data shape, feature engineering.
  - Model families tried.
  - Evaluation protocol (train/val/test, CV, metrics).
  - Key results and error analysis highlights.
  - Any interesting modeling decisions (e.g., threshold tuning, handling imbalance).

### 4. Portfolio Planning

In Markdown, draft a plan:

- List **2–3 projects** you’d like to showcase publicly (GitHub, portfolio site).
- For each:
  - Title.
  - Short description (2–3 lines).
  - Status (e.g., “need to refactor code”, “need to add README and visuals”).
- Decide:
  - Which will be your “flagship” project (most polished).
  - Which ones will be secondary examples.

### 5. Checklist for Each Portfolio Project

Make a simple checklist you’ll follow later:

- [ ] Clean, well‑commented notebook or `.py` scripts.
- [ ] Clear README:
  - Problem statement.
  - Data description.
  - Approach.
  - Results.
  - How to run.
- [ ] Requirements (`requirements.txt` or `environment.yml`).
- [ ] Plots and tables embedded for key results.
- [ ] Short error analysis section.
- [ ] Short interpretability section (if applicable).

### 6. Thinking Challenge

Reflect in 10–15 sentences:

- Before this week, how did you think about “model performance”?  
  How has your view changed now that you’ve:
  - Done error analysis?
  - Compared models across multiple criteria?
  - Thought about interpretability and communication?
- What would you do differently now if you started a brand‑new ML project?

---

## After Week 11: What You Should Be Able to Do

You should now:

- Systematically analyze **where** and **why** your models fail.
- Slice performance by features and inspect residuals/confusion patterns.
- Compare models with a view to:
  - Metrics.
  - Interpretability.
  - Stability.
  - Practical constraints.
- Use basic interpretability tools (feature importance, PD plots, simple SHAP).
- Turn an ML project into a **coherent, portfolio‑ready case study** with clear communication.

For Week 12, the natural focus is:

- Final **polish and documentation**.
- Setting up GitHub repos, READMEs, and possibly a simple portfolio page.
- Planning what to learn next (e.g., deeper math, advanced deep learning, or deployment/MLOps).
