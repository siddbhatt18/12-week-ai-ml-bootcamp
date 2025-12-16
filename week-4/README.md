**Week 4 Goal:**  
Strengthen your understanding of **classification** by working with logistic regression, k‑Nearest Neighbors (k‑NN), and core evaluation metrics (accuracy, precision, recall, F1, ROC‑AUC). You’ll also practice comparing models and thinking about class imbalance.

Assumption: ~1.5–2 hours/day. Use **one main classification dataset** all week (e.g., Breast Cancer, Titanic, or another binary classification dataset).

---

## Day 1 – Solidifying Logistic Regression & Metrics

**Objectives**
- Refresh logistic regression on your chosen real dataset.
- Get very comfortable with accuracy, confusion matrix, precision, recall, F1.

**Tasks**

1. **New notebook**
   - `week4_day1_logreg_metrics.ipynb`.

2. **Load your dataset**
   - Example (Breast Cancer built‑in):
     ```python
     from sklearn.datasets import load_breast_cancer
     import pandas as pd

     data = load_breast_cancer(as_frame=True)
     df = data.frame
     df.head()
     ```

   - Choose target and features:
     ```python
     X = df.drop(columns=["target"])
     y = df["target"]
     ```

3. **Train/test split**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

4. **Train logistic regression**
   ```python
   from sklearn.linear_model import LogisticRegression

   log_reg = LogisticRegression(max_iter=1000)
   log_reg.fit(X_train, y_train)

   y_pred = log_reg.predict(X_test)
   ```

5. **Evaluate with metrics**
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

   acc = accuracy_score(y_test, y_pred)
   cm = confusion_matrix(y_test, y_pred)

   print("Accuracy:", acc)
   print("Confusion matrix:\n", cm)
   print("Classification report:\n", classification_report(y_test, y_pred))
   ```

6. **Manual metric interpretation**
   - In a markdown cell:
     - Label confusion matrix entries:
       - TN, FP, FN, TP.
     - Write the formula for:
       - Precision = TP / (TP + FP)
       - Recall = TP / (TP + FN)
       - F1 = 2 * (precision * recall) / (precision + recall)

7. **Mini reflection**
   - Are precision and recall similar, or is one much lower?
   - For your problem, does **missing positives** or **raising false alarms** matter more?

**Outcome Day 1**
- Fully comfortable running logistic regression and reading standard classification metrics.

---

## Day 2 – Understanding and Implementing k‑Nearest Neighbors (k‑NN)

**Objectives**
- Learn the intuition of k‑NN.
- Train k‑NN on your dataset and see how k affects performance.

**Tasks**

1. **New notebook**
   - `week4_day2_knn_basics.ipynb`.
   - Reload or reuse `X_train`, `X_test`, `y_train`, `y_test` from Day 1 (or copy code).

2. **Concepts in markdown**
   - Briefly explain k‑NN:
     - Stores training examples.
     - To classify a new point, finds the **k closest training points** (neighbors).
     - Uses majority vote among neighbors’ labels.
     - Sensitive to feature scales and irrelevant features.

3. **Select numeric features only (to avoid encoding complexity for now)**
   ```python
   import numpy as np

   numeric_cols = X_train.select_dtypes(include=[np.number]).columns
   X_train_num = X_train[numeric_cols]
   X_test_num = X_test[numeric_cols]
   ```

4. **Scale features (very important for k‑NN)**
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train_num)
   X_test_scaled = scaler.transform(X_test_num)
   ```

5. **Train k‑NN with different k**
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.metrics import accuracy_score

   ks = [1, 3, 5, 7, 11, 15]
   results = []

   for k in ks:
       knn = KNeighborsClassifier(n_neighbors=k)
       knn.fit(X_train_scaled, y_train)
       y_pred = knn.predict(X_test_scaled)
       acc = accuracy_score(y_test, y_pred)
       results.append((k, acc))

   import pandas as pd
   pd.DataFrame(results, columns=["k", "accuracy"])
   ```

6. **Plot accuracy vs k (optional but helpful)**
   ```python
   import matplotlib.pyplot as plt

   ks_list = [r[0] for r in results]
   acc_list = [r[1] for r in results]

   plt.plot(ks_list, acc_list, marker="o")
   plt.xlabel("k (number of neighbors)")
   plt.ylabel("Test accuracy")
   plt.title("k-NN performance vs k")
   plt.show()
   ```

7. **Mini reflection**
   - Which k gives best accuracy?
   - What happens at k=1 vs large k conceptually (variance vs bias)?

**Outcome Day 2**
- You can fit k‑NN, understand why scaling matters, and see how k affects performance.

---

## Day 3 – Comparing Logistic Regression and k‑NN + Cross‑Validation

**Objectives**
- Fairly compare logistic regression vs k‑NN using cross‑validation.
- Start using `cross_val_score` to get more robust performance estimates.

**Tasks**

1. **New notebook**
   - `week4_day3_logreg_vs_knn_cv.ipynb`.

2. **Reuse prepared data**
   - Use `X`, `y` from your dataset (full data, not split yet).
   - For k‑NN, you’ll scale features inside a pipeline.

3. **Create pipelines for both models**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from sklearn.neighbors import KNeighborsClassifier

   pipe_log_reg = Pipeline([
       ("scaler", StandardScaler()),
       ("model", LogisticRegression(max_iter=1000))
   ])

   pipe_knn = Pipeline([
       ("scaler", StandardScaler()),
       ("model", KNeighborsClassifier(n_neighbors=5))
   ])
   ```

4. **Cross‑validation comparison**
   ```python
   from sklearn.model_selection import cross_val_score

   for name, pipe in [("Logistic Regression", pipe_log_reg),
                      ("k-NN (k=5)", pipe_knn)]:
       scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
       print(name)
       print("  Mean accuracy:", scores.mean())
       print("  Std:", scores.std())
   ```

5. **Try a few k values for k‑NN with CV**
   ```python
   ks = [3, 5, 7, 11]
   for k in ks:
       pipe_knn = Pipeline([
           ("scaler", StandardScaler()),
           ("model", KNeighborsClassifier(n_neighbors=k))
       ])
       scores = cross_val_score(pipe_knn, X, y, cv=5, scoring="accuracy")
       print(f"k-NN (k={k}) -> mean acc: {scores.mean():.3f}, std: {scores.std():.3f}")
   ```

6. **Mini reflection (markdown)**
   - Across CV folds, which model is:
     - More accurate on average?
     - More stable (lower std)?
   - Do you expect that to generalize to unseen test data?

**Outcome Day 3**
- You can use pipelines + cross‑validation to fairly compare models.

---

## Day 4 – ROC Curves, AUC, and Thresholds

**Objectives**
- Understand ROC curves and AUC for binary classification.
- See how changing the decision threshold affects precision/recall trade‑off.

**Tasks**

1. **New notebook**
   - `week4_day4_roc_auc_thresholds.ipynb`.

2. **Train logistic regression again (with scaling in pipeline)**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   pipe_log_reg = Pipeline([
       ("scaler", StandardScaler()),
       ("model", LogisticRegression(max_iter=1000))
   ])

   pipe_log_reg.fit(X_train, y_train)
   ```

3. **Obtain predicted probabilities**
   ```python
   # Probabilities for class 1
   y_proba = pipe_log_reg.predict_proba(X_test)[:, 1]
   ```

4. **ROC curve and AUC**
   ```python
   from sklearn.metrics import roc_curve, roc_auc_score
   import matplotlib.pyplot as plt

   fpr, tpr, thresholds = roc_curve(y_test, y_proba)
   auc = roc_auc_score(y_test, y_proba)

   plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
   plt.plot([0, 1], [0, 1], "k--")  # random baseline
   plt.xlabel("False Positive Rate")
   plt.ylabel("True Positive Rate (Recall)")
   plt.title("ROC Curve - Logistic Regression")
   plt.legend()
   plt.show()
   ```

5. **Threshold tuning demo**
   ```python
   import numpy as np
   from sklearn.metrics import precision_score, recall_score

   for thresh in [0.3, 0.5, 0.7]:
       y_pred_thresh = (y_proba >= thresh).astype(int)
       prec = precision_score(y_test, y_pred_thresh)
       rec = recall_score(y_test, y_pred_thresh)
       print(f"Threshold {thresh}: precision={prec:.3f}, recall={rec:.3f}")
   ```

6. **Mini reflection**
   - How does increasing threshold affect precision and recall?
   - In what situations would you:
     - Prefer high recall (catch almost all positives)?
     - Prefer high precision (avoid false positives)?

**Outcome Day 4**
- You can generate ROC curves, compute AUC, and reason about classification thresholds.

---

## Day 5 – Class Imbalance & Baselines

**Objectives**
- Recognize class imbalance.
- Compare model to simple baselines (majority class).
- Introduce class weights as a simple handling technique.

**Tasks**

1. **New notebook**
   - `week4_day5_class_imbalance.ipynb`.

2. **Check class distribution**
   ```python
   import pandas as pd

   y.value_counts()
   y.value_counts(normalize=True)
   ```

   - If your dataset is *not* very imbalanced (e.g., Breast Cancer ~ 37/63), still follow steps, but imagine a more imbalanced scenario.
   - If you have a more imbalanced dataset (like a real churn or fraud dataset), even better.

3. **Majority class baseline**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   majority_class = y_train.value_counts().idxmax()
   y_pred_majority = [majority_class] * len(y_test)
   acc_baseline = accuracy_score(y_test, y_pred_majority)
   print("Majority class baseline accuracy:", acc_baseline)
   ```

4. **Logistic regression vs baseline**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import classification_report

   pipe_log_reg = Pipeline([
       ("scaler", StandardScaler()),
       ("model", LogisticRegression(max_iter=1000))
   ])
   pipe_log_reg.fit(X_train, y_train)
   y_pred = pipe_log_reg.predict(X_test)

   print("LogReg accuracy:", accuracy_score(y_test, y_pred))
   print("Classification report:\n", classification_report(y_test, y_pred))
   ```

5. **Class weights with logistic regression**
   ```python
   pipe_log_reg_weighted = Pipeline([
       ("scaler", StandardScaler()),
       ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
   ])
   pipe_log_reg_weighted.fit(X_train, y_train)
   y_pred_w = pipe_log_reg_weighted.predict(X_test)

   print("Weighted LogReg accuracy:", accuracy_score(y_test, y_pred_w))
   print("Weighted classification report:\n", classification_report(y_test, y_pred_w))
   ```

6. **Mini reflection**
   - Does the weighted model improve recall for the minority class?
   - Did overall accuracy go up or down?
   - Why might sacrificing a bit of accuracy still be good in imbalanced problems?

**Outcome Day 5**
- You understand class imbalance, majority baselines, and class weighting in logistic regression.

---

## Day 6 – Small Model Selection Exercise (LogReg vs k‑NN) with Metrics

**Objectives**
- Practice a small model selection process:
  - Compare models using accuracy, F1, and AUC.
- Use cross‑validation metrics beyond simple accuracy.

**Tasks**

1. **New notebook**
   - `week4_day6_model_selection.ipynb`.

2. **Define candidate models (with pipelines)**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from sklearn.neighbors import KNeighborsClassifier

   models = {
       "LogReg": Pipeline([
           ("scaler", StandardScaler()),
           ("model", LogisticRegression(max_iter=1000))
       ]),
       "LogReg_balanced": Pipeline([
           ("scaler", StandardScaler()),
           ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
       ]),
       "kNN_k5": Pipeline([
           ("scaler", StandardScaler()),
           ("model", KNeighborsClassifier(n_neighbors=5))
       ]),
       "kNN_k11": Pipeline([
           ("scaler", StandardScaler()),
           ("model", KNeighborsClassifier(n_neighbors=11))
       ]),
   }
   ```

3. **Evaluate with cross‑validation on multiple metrics**
   ```python
   from sklearn.model_selection import cross_val_score

   scoring_metrics = ["accuracy", "f1", "roc_auc"]

   import pandas as pd
   rows = []

   for name, pipe in models.items():
       row = {"model": name}
       for metric in scoring_metrics:
           scores = cross_val_score(pipe, X, y, cv=5, scoring=metric)
           row[f"{metric}_mean"] = scores.mean()
           row[f"{metric}_std"] = scores.std()
       rows.append(row)

   results_df = pd.DataFrame(rows)
   results_df
   ```

4. **Interpret results**
   - Sort `results_df` by `roc_auc_mean` or `f1_mean`:
     ```python
     results_df.sort_values(by="roc_auc_mean", ascending=False)
     ```

5. **Mini reflection**
   - Does the “best” model change depending on metric (accuracy vs F1 vs AUC)?
   - If dataset is imbalanced, which metric is more informative and why?

**Outcome Day 6**
- You can run a basic model selection experiment across multiple metrics.

---

## Day 7 – Week 4 Mini Project: Classification Deep Dive

**Objectives**
- Build one well‑structured classification notebook that:
  - Performs EDA relevant to classification.
  - Compares logistic regression and k‑NN.
  - Uses multiple metrics and discusses trade‑offs.

**Tasks**

1. **New notebook**
   - `week4_day7_mini_project.ipynb`.

2. **Structure with markdown sections**

   ### 1. Problem & Data Description
   - Dataset name and a 2–3 sentence description.
   - What is the target you’re trying to predict?
   - Are classes balanced or imbalanced?

   ### 2. Data Loading & Basic EDA
   - Load dataset, show `head()`, `info()`, `value_counts()` of target.
   - Show:
     - Distribution of 2–3 key features (histograms).
     - 1–2 boxplots of numeric feature vs target class (if target is 0/1).
   - Write 3–5 bullet points about patterns you see.

   ### 3. Train/Test Split & Preprocessing
   - Define `X`, `y`.
   - Train/test split with stratification.
   - Scaling numeric features via `StandardScaler` in pipelines.

   ### 4. Model Training & Evaluation
   - Train:
     - Logistic Regression (plain).
     - Logistic Regression (`class_weight="balanced"` if relevant).
     - k‑NN with two k values (e.g., 5, 11).
   - For each model:
     - Report test accuracy.
     - Show confusion matrix and classification report.
   - For your best model:
     - Plot ROC curve and report AUC.
     - Show how precision/recall change at 2–3 different thresholds.

   ### 5. Comparison & Discussion
   - In markdown, 8–10 bullet points:
     - Which model worked best and under which metric?
     - Did class imbalance matter?
     - Where does the model perform poorly (e.g., many FNs or FPs)?
     - Which features appear most important (from logistic regression coefficients)?
     - Any clear trade‑offs between precision and recall?

   ### 6. Next Steps
   - 4–6 bullet points of potential improvements:
     - Feature engineering ideas.
     - Trying other models (decision trees, random forests).
     - Better hyperparameter tuning, etc.

3. **Polish**
   - Ensure the notebook runs top‑to‑bottom.
   - Keep sections clear and add short explanations before/after key code blocks.

**Outcome Day 7**
- A strong, self‑contained classification project notebook comparing logistic regression and k‑NN with proper metrics and discussion.
- You are ready to move into Week 5: richer preprocessing (pipelines + column transformers) and more powerful models like decision trees and random forests.
