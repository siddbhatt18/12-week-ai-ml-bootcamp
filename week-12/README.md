**Week 12 Goal:**  
Consolidate everything you’ve learned into a small **portfolio**: polish 2–3 projects, write clear explanations, and (optionally) add a minimal interface or script that uses your models.

Assume ~1.5–2 hours/day. This week is more “meta”: refining, documenting, and presenting your work rather than learning new algorithms.

---

## Day 1 – Choose Portfolio Projects & Audit Their State

**Objectives**
- Decide which projects will represent you.
- Review their current state and list what needs fixing.

**Tasks**

1. **Select 2–3 projects** (markdown list in `week12_day1_portfolio_plan.md` or a notebook)
   - Good candidates:
     - Week 10 end‑to‑end project (clean dataset, strong pipeline, tuned model).
     - Week 11 messy data project.
     - One NN project (Week 9) or an earlier Kaggle/tabular project.
   - For each, note:
     - Dataset.
     - Problem type (classification/regression).
     - Best model & key metric (e.g., “RandomForest, AUC=0.87”).

2. **Create a simple folder structure**
   - Example:
     - `portfolio/`
       - `project1_name/`
       - `project2_name/`
       - `project3_name/` (optional)
   - Copy the main notebooks and data references into each folder (or at least set up the structure).

3. **Audit each chosen project**
   - For each notebook:
     - Open and run top‑to‑bottom.
     - Note issues:
       - Broken cells.
       - Inconsistent variable names.
       - Unused code.
       - Plots that are hard to read.
   - In your plan markdown, create a **to‑do checklist** per project:
     - Clean code?
     - Add clear headings?
     - Explain evaluation metrics?
     - Show baseline vs final model?
     - Interpret feature importance?

4. **Mini reflection**
   - Which project needs the most work?
   - Which is already close to “portfolio‑ready”?

**Outcome Day 1**
- Clear list of 2–3 portfolio projects and specific improvements needed for each.

---

## Day 2 – Clean & Refactor Project 1 Notebook

**Objectives**
- Turn your first project notebook into a clean, linear narrative:
  - No dead code.
  - Clear sections.
  - Readable outputs.

**Tasks**

1. **Pick Project 1** (e.g., Week 10 end‑to‑end project)
   - Open its main notebook (copy into `portfolio/project1_name/project1_notebook.ipynb`).

2. **Structure with clear headings**
   - Add/adjust markdown headings:
     1. Title & Problem Description
     2. Data Loading & Exploration
     3. Preprocessing & Feature Engineering
     4. Modeling & Evaluation
     5. Interpretation
     6. Conclusions & Next Steps

3. **Remove clutter**
   - Delete:
     - Debug prints.
     - Redundant cells (e.g., multiple versions of the same model; keep the best one and maybe one baseline).
   - Merge small, related cells into a single cohesive code block where it improves readability.

4. **Standardize style**
   - Use consistent:
     - Variable naming (`X_train`, `y_train`, `best_model`, etc.).
     - Plot styles (titles, axis labels, `plt.tight_layout()`).
   - Add short comments before tricky code segments.

5. **Tighten narrative**
   - After each major block (EDA, modeling, evaluation), add 2–4 bullets in markdown summarizing:
     - What you did.
     - What you found.
     - Why it matters for the next step.

6. **Re‑run notebook from top to bottom**
   - Fix any execution order issues.
   - Confirm all outputs appear where expected.

7. **Mini reflection**
   - Is it clear to a reader who knows ML what you did and why?
   - Is there any step that still feels confusing or under‑explained?

**Outcome Day 2**
- One polished project notebook with a clear, professional structure and narrative.

---

## Day 3 – Clean & Refactor Project 2 Notebook (Messy Data Project)

**Objectives**
- Do the same cleanup for your second project, focusing on **data cleaning and real‑world issues**.

**Tasks**

1. **Pick Project 2** (e.g., Week 11 messy data project)
   - Open its notebook under `portfolio/project2_name/`.

2. **Reorganize sections**
   - Suggested structure:
     1. Problem & Dataset
     2. Data Issues & Cleaning Strategy
     3. Preprocessing Pipeline
     4. Feature Engineering & Experiments
     5. Model Selection & Tuning
     6. Final Model & Interpretation
     7. Lessons Learned

3. **Highlight messiness clearly**
   - In “Data Issues & Cleaning Strategy”:
     - Show:
       - Missing values summary.
       - Weird encodings or outliers.
     - Describe in bullets:
       - What choices you made and why (imputation strategy, dropping some columns, etc.).

4. **Keep experiment history brief but informative**
   - If you tried many models:
     - Show a **single summary table** of experiments:
       - Model, key hyperparameters, main metric, notes.
     - Don’t keep every intermediate training cell; keep 1–2 representative examples.

5. **Emphasize robustness**
   - Make sure it’s obvious that:
     - All preprocessing is inside `Pipeline` / `ColumnTransformer`.
     - Leakage is avoided (no using test data in training/validation steps).
   - Include 3–5 bullets explicitly stating these good practices.

6. **Re‑run and fix**
   - Run the notebook end‑to‑end.
   - Ensure the narrative flows logically:
     - Cleaning → preprocessing → modeling → interpretation.

7. **Mini reflection**
   - Does the notebook show that you can handle messy data thoughtfully?
   - If someone skimmed only the headings and bullet summaries, would they “get” the project?

**Outcome Day 3**
- A second, well‑structured project notebook that emphasizes your ability to handle real‑world, messy data problems.

---

## Day 4 – Optional Project 3 & Short Written Summaries

**Objectives**
- Optionally clean a third project (e.g., NN/MNIST).
- Write short summaries (“project cards”) for each project that you can reuse (CV, README, LinkedIn, etc.).

**Tasks**

1. **If you have a third project** (e.g., MNIST NN or a different tabular project):
   - Repeat a **lighter version** of Day 2/3 cleanup:
     - Clear sections.
     - Remove clutter.
     - Make sure it runs top‑to‑bottom.
     - Add a brief reflection section at the end.

2. **Write project summaries** (in a markdown file `portfolio/project_summaries.md`):
   - For each project (2–3 of them), write:

   **a) 1–2 sentence overview**
   - Example:
     - “Built a gradient‑boosted model to predict customer churn from telco usage data, achieving AUC 0.89 on a held‑out test set.”

   **b) 3–6 bullet points on:**
   - Data:
     - Size, features, any notable messiness.
   - Techniques:
     - Pipelines, feature engineering, models (RF, GB, XGBoost, NN, etc.).
   - Evaluation:
     - Metrics used and key scores.
   - Interpretability:
     - Feature importance, partial dependence, error analysis.

   **c) 1–2 sentence “what I learned”**
   - Example:
     - “This project taught me how to handle high‑cardinality categorical variables and avoid data leakage in a complex pipeline.”

3. **Polish language**
   - Read each summary and:
     - Remove jargon where possible.
     - Make it understandable to a technical but non‑ML specialist (e.g., a software engineer, hiring manager).

4. **Mini reflection**
   - Which project feels strongest in terms of:
     - Technical depth?
     - Real‑world relevance?
     - Storytelling?

**Outcome Day 4**
- Optional third cleaned project and concise, reusable written summaries for each chosen project.

---

## Day 5 – Add a Minimal “Interface” or Script for One Project

**Objectives**
- Demonstrate that your model is usable outside a notebook.
- Build a minimal interface:
  - Either a simple CLI/script, or
  - A basic web UI with Streamlit (optional, but nice).

**Tasks**

1. **Choose 1 project for “interface” work**  
   - Preferably the Week 10 or Week 11 project with a saved pipeline.

2. **Ensure model saving/loading is in place**
   - In the project folder, have a small script `train_and_save.py` or a notebook cell that:
     - Trains the final model.
     - Saves it with joblib:
       ```python
       import joblib
       joblib.dump(best_model, "final_model_pipeline.joblib")
       ```

3. **Create a simple prediction script (`predict.py`)**
   - In `portfolio/projectX/`:

   ```python
   # predict.py
   import sys
   import pandas as pd
   import joblib

   def load_model(path="final_model_pipeline.joblib"):
       return joblib.load(path)

   def predict_from_csv(model, input_path, output_path="predictions.csv"):
       df = pd.read_csv(input_path)
       preds = model.predict(df)
       df["prediction"] = preds
       df.to_csv(output_path, index=False)
       print(f"Saved predictions to {output_path}")

   if __name__ == "__main__":
       if len(sys.argv) < 2:
           print("Usage: python predict.py input.csv [output.csv]")
           sys.exit(1)

       input_path = sys.argv[1]
       output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"

       model = load_model()
       predict_from_csv(model, input_path, output_path)
   ```

   - (Even if you don’t run it from terminal right now, having it structured is valuable.)

4. **Optional: Streamlit app**
   - Install Streamlit:
     ```bash
     pip install streamlit
     ```
   - Create a `app.py`:

   ```python
   import streamlit as st
   import pandas as pd
   import joblib

   @st.cache_resource
   def load_model():
       return joblib.load("final_model_pipeline.joblib")

   model = load_model()
   st.title("Your Project Name – Prediction App")

   st.write("Enter feature values to get a prediction.")

   # Build UI for a few key features
   # Example – adjust to your features:
   age = st.number_input("Age", min_value=0, max_value=100, value=30)
   sex = st.selectbox("Sex", ["male", "female"])
   pclass = st.selectbox("Pclass", [1, 2, 3])

   if st.button("Predict"):
       input_df = pd.DataFrame([{
           "Age": age,
           "Sex": sex,
           "Pclass": pclass
           # add other required features with defaults
       }])
       pred = model.predict(input_df)[0]
       st.write(f"Prediction: {pred}")
   ```

5. **Mini reflection**
   - Is it clear how someone else could use your model now?
   - What would you need to add for a “real” deployment (logging, error handling, schema checks, etc.)?

**Outcome Day 5**
- One project with a minimal but real usage interface (script or small app) that demonstrates end‑to‑end usability.

---

## Day 6 – Final Review: Consistency, Reproducibility, and README Files

**Objectives**
- Ensure your projects are **consistent and reproducible**.
- Add README files with clear instructions.

**Tasks**

1. **Create/Polish README for each project**
   - In each `portfolio/projectX/` folder, add a `README.md` containing:

   - **Title**
   - **Overview** (2–3 sentences)
   - **Data**
     - Source (e.g., Kaggle link).
     - Basic description (rows, features, target).
   - **Methods**
     - Feature preprocessing (ColumnTransformer, imputation, encoding).
     - Main models used (RF, GB, etc.).
   - **Results**
     - Key metric(s) on validation/test.
   - **How to Run**
     - Dependencies (Python version, key libraries).
     - How to run notebook.
     - How to run `train_and_save.py` or `predict.py`, if present.

2. **Check reproducibility**
   - For at least your 2 main projects:
     - Create a fresh environment (optional if time), or just:
       - Restart kernel.
       - Run notebook from top to bottom.
     - Note any:
       - Non‑deterministic results (slightly changing metrics).
       - Missing files/paths.

3. **Standardize random seeds**
   - Ensure that:
     - `random_state=42` (or similar) is set consistently for:
       - `train_test_split`.
       - Tree/boosting models.
       - K‑folds.
   - This helps make runs more reproducible.

4. **Check for hard‑coded paths**
   - Replace absolute paths (e.g., `/Users/you/...`) with relative paths (e.g., `../data/file.csv`).
   - Mention expected directory structure in README if necessary.

5. **Mini reflection**
   - Could another ML practitioner run your project with:
     - `git clone` (imagined) + `pip install -r requirements.txt` + “open notebook”?
   - If not, what’s missing?

**Outcome Day 6**
- Each project has a clear README and runs cleanly; your work is now much closer to something you could publish or share.

---

## Day 7 – Week 12 Wrap‑Up & Next Steps Plan

**Objectives**
- Reflect on your 12‑week journey.
- Make a concrete plan for what to do next (learning + projects).
- Optionally publish/share your work somewhere (GitHub, Kaggle, etc.).

**Tasks**

1. **Reflection document**
   - Create `week12_day7_reflection_and_next_steps.md` (or a notebook with markdown).
   - Answer:

   **a) What you can do now (skills inventory)**
   - Bullet list:
     - Data loading and cleaning (pandas, handling missing values).
     - EDA (plots, correlations).
     - Building pipelines with ColumnTransformer.
     - Training and tuning:
       - Linear/logistic models.
       - Trees, random forests, gradient boosting.
       - Basic neural networks with Keras.
     - Evaluation:
       - CV, train/val/test splits.
       - Metrics (accuracy, F1, AUC, RMSE, etc.).
     - Interpretation:
       - Feature importance, partial dependence.
     - Model saving & minimal deployment.

   **b) What you found hardest**
   - 5–10 bullets:
     - E.g., “Hyperparameter tuning search spaces”, “remembering all sklearn APIs”, “interpreting tree‑based models”.

   **c) What you want to improve next**
   - Example areas:
     - More advanced ML:
       - XGBoost/LightGBM in depth.
       - Time series forecasting.
       - NLP (text classification, embeddings).
       - Deep learning (CNNs for images, RNNs/Transformers for sequences).
     - Engineering:
       - Production ML, MLOps, deployment.
       - Better code organization, testing.

2. **Concrete 4–6 week next‑steps plan**
   - Example:
     - Weeks 13–14:
       - Do 1–2 small Kaggle competitions with gradient boosting.
     - Weeks 15–16:
       - Take a short course/tutorial on CNNs and do 1 image project.
     - Weeks 17–18:
       - Learn basics of deployment (FastAPI/Streamlit + Docker) and deploy one of your models.

3. **Optional: Share your work**
   - If you use Git/GitHub:
     - Initialize a repo (or imagine structure):
       - `/portfolio/project1_name/`
       - `/portfolio/project2_name/`
     - Push notebooks and README files.
   - If not yet ready to publish:
     - At least organize everything cleanly on your machine so you can share zipped folders later.

4. **Mini reflection (closing)**
   - Write a short paragraph:
     - What surprised you most about ML practice vs expectations?
     - Where do you feel most confident?
     - What’s one thing you would do differently if you restarted the 12 weeks?

**Outcome Day 7**
- A clear picture of what you’ve accomplished, well‑organized portfolio projects, and a realistic roadmap for continuing your ML journey beyond these 12 weeks.
