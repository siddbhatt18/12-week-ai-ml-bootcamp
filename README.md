# 12-Week Machine Learning Study Plan (80-20 Approach)

## Prerequisites
- Basic Python programming
- High school level math (algebra)
- 10-15 hours/week commitment

---

## **WEEKS 1-2: Python & Essential Libraries**

### Week 1: Python for ML
**Core Topics (20% that matters):**
- NumPy: arrays, indexing, broadcasting, basic operations
- Pandas: DataFrames, reading CSV/Excel, filtering, grouping
- Data manipulation: handling missing values, basic transformations

**Daily Practice:**
- Day 1-2: NumPy arrays and operations
- Day 3-4: Pandas DataFrames and data loading
- Day 5-6: Data cleaning exercises
- Day 7: Mini-project combining both

**Resources:**
- NumPy quickstart tutorial (official docs)
- Pandas 10 minutes to pandas guide

**Challenge Exercise:** Load a CSV dataset (e.g., Titanic), explore it, handle missing values, and create summary statistics.

---

### Week 2: Visualization & Math Foundations
**Core Topics:**
- Matplotlib/Seaborn: line plots, scatter plots, histograms, heatmaps
- Essential math: vectors, matrices, derivatives (conceptual)
- Statistics basics: mean, median, standard deviation, correlation

**Daily Practice:**
- Day 1-2: Basic plots with Matplotlib
- Day 3-4: Statistical visualizations with Seaborn
- Day 5-6: Math refresher (Khan Academy linear algebra basics)
- Day 7: Exploratory Data Analysis (EDA) mini-project

**Challenge Exercise:** Perform complete EDA on a dataset—find correlations, visualize distributions, identify patterns.

---

## **WEEKS 3-4: Supervised Learning Fundamentals**

### Week 3: Linear Models & Regression
**Core Topics:**
- Linear Regression (theory + implementation)
- Cost functions and gradient descent (conceptual understanding)
- Train/test split and evaluation metrics (MSE, RMSE, R²)
- Overfitting vs underfitting
- Scikit-learn basics

**Daily Practice:**
- Day 1-2: Linear regression theory and manual implementation
- Day 3-4: Scikit-learn Linear Regression
- Day 5-6: Model evaluation and interpretation
- Day 7: Build a price prediction model

**Challenge Exercise:** Predict house prices without following a tutorial step-by-step. Research what you need.

---

### Week 4: Classification Basics
**Core Topics:**
- Logistic Regression
- Classification metrics: accuracy, precision, recall, F1-score, confusion matrix
- K-Nearest Neighbors (KNN)
- Decision Trees (basic understanding)

**Daily Practice:**
- Day 1-2: Logistic regression theory and implementation
- Day 3-4: Classification metrics deep dive
- Day 5: KNN implementation
- Day 6: Decision Trees
- Day 7: Classification project

**Challenge Exercise:** Build a binary classifier (e.g., spam detection) and optimize for recall vs precision based on business requirements you define.

---

## **WEEKS 5-6: Advanced Algorithms & Ensembles**

### Week 5: Tree-Based Models
**Core Topics:**
- Random Forests (the workhorse of ML)
- Feature importance
- Hyperparameter tuning basics
- Cross-validation

**Daily Practice:**
- Day 1-2: Random Forest theory and implementation
- Day 3-4: Feature importance and selection
- Day 5-6: Cross-validation and hyperparameter tuning
- Day 7: Comparison project (Linear vs Tree models)

**Challenge Exercise:** Take a complex dataset and systematically improve model performance through feature engineering and hyperparameter tuning.

---

### Week 6: Gradient Boosting & Model Optimization
**Core Topics:**
- Gradient Boosting basics (XGBoost/LightGBM)
- Grid Search and Random Search
- Handling imbalanced data
- Model persistence (saving/loading models)

**Daily Practice:**
- Day 1-3: XGBoost implementation and tuning
- Day 4-5: Imbalanced data techniques (SMOTE, class weights)
- Day 6: Model optimization strategies
- Day 7: End-to-end ML pipeline

**Challenge Exercise:** Build a complete ML pipeline from raw data to saved model, including preprocessing, training, and evaluation.

---

## **WEEKS 7-8: Unsupervised Learning & Dimensionality Reduction**

### Week 7: Clustering
**Core Topics:**
- K-Means clustering
- Hierarchical clustering
- DBSCAN (density-based)
- Choosing the right number of clusters (elbow method, silhouette score)

**Daily Practice:**
- Day 1-2: K-Means theory and implementation
- Day 3-4: Other clustering algorithms
- Day 5-6: Cluster evaluation and interpretation
- Day 7: Customer segmentation project

**Challenge Exercise:** Segment customers/users without predefined labels and derive actionable insights.

---

### Week 8: Dimensionality Reduction
**Core Topics:**
- Principal Component Analysis (PCA)
- Feature scaling and normalization
- When and why to reduce dimensions
- Visualization of high-dimensional data

**Daily Practice:**
- Day 1-3: PCA theory and implementation
- Day 4-5: Feature scaling techniques
- Day 6: Applying PCA before classification/clustering
- Day 7: Comprehensive data preprocessing project

**Challenge Exercise:** Take a high-dimensional dataset, reduce dimensions, and show that your model improves or maintains performance.

---

## **WEEKS 9-10: Neural Networks & Deep Learning Basics**

### Week 9: Neural Network Fundamentals
**Core Topics:**
- Perceptrons and activation functions
- Feedforward neural networks
- Backpropagation (conceptual)
- TensorFlow/Keras basics
- Preventing overfitting (dropout, early stopping)

**Daily Practice:**
- Day 1-2: Neural network theory
- Day 3-4: Building simple networks with Keras
- Day 5-6: Training and regularization techniques
- Day 7: Classification with neural networks

**Challenge Exercise:** Build a neural network classifier and compare performance with traditional ML algorithms.

---

### Week 10: Convolutional Neural Networks (CNNs)
**Core Topics:**
- Image data preprocessing
- Convolutional layers, pooling
- CNN architectures (basic understanding)
- Transfer learning (using pre-trained models)

**Daily Practice:**
- Day 1-2: CNN theory and architecture
- Day 3-4: Building a simple CNN
- Day 5-6: Transfer learning with VGG16/ResNet
- Day 7: Image classification project

**Challenge Exercise:** Build an image classifier using transfer learning without following step-by-step tutorials.

---

## **WEEKS 11-12: Real-World ML & Deployment**

### Week 11: ML Pipeline & Best Practices
**Core Topics:**
- Feature engineering strategies
- Handling categorical variables (one-hot, label encoding, target encoding)
- Data leakage prevention
- Model interpretation (SHAP values basics)
- A/B testing concepts

**Daily Practice:**
- Day 1-2: Advanced feature engineering
- Day 3-4: Avoiding common pitfalls (data leakage, etc.)
- Day 5-6: Model interpretation techniques
- Day 7: Refactor a previous project with best practices

**Challenge Exercise:** Audit one of your earlier projects for potential issues and improve it.

---

### Week 12: Model Deployment & MLOps Basics
**Core Topics:**
- Creating a simple API with Flask/FastAPI
- Deploying a model to the cloud (Heroku/Streamlit)
- Monitoring and versioning basics
- Ethics in ML (bias, fairness)

**Daily Practice:**
- Day 1-3: Build a Flask API for your model
- Day 4-5: Deploy to Streamlit Cloud or Heroku
- Day 6: Model monitoring basics
- Day 7: Final project refinement

**Challenge Exercise:** Deploy one of your projects as a web application that others can use.

---

## **5 PROGRESSIVE PROJECTS**

### **Project 1: Customer Churn Prediction (Weeks 3-4)**
**Difficulty:** Beginner  
**Description:** Predict whether customers will leave a service based on their usage patterns and demographics.

**Dataset:** Telco Customer Churn (Kaggle)

**Key Concepts Reinforced:**
- Data preprocessing and EDA
- Binary classification
- Train/test split
- Logistic Regression and Decision Trees
- Evaluation metrics (precision, recall)
- Handling imbalanced data

**Challenge Elements:**
- Define business metrics (is false positive or false negative more costly?)
- Feature engineering without guidance
- Compare multiple algorithms

---

### **Project 2: House Price Prediction System (Weeks 5-6)**
**Difficulty:** Beginner-Intermediate  
**Description:** Build a comprehensive regression model to predict house prices with advanced feature engineering.

**Dataset:** House Prices - Advanced Regression Techniques (Kaggle)

**Key Concepts Reinforced:**
- Advanced feature engineering
- Handling missing data strategically
- Categorical encoding techniques
- Random Forest and Gradient Boosting
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis

**Challenge Elements:**
- Create new features from existing ones
- Handle outliers appropriately
- Build a complete preprocessing pipeline
- Achieve a specific performance threshold

---

### **Project 3: Customer Segmentation for Marketing (Weeks 7-8)**
**Difficulty:** Intermediate  
**Description:** Segment customers into distinct groups for targeted marketing campaigns using unsupervised learning.

**Dataset:** Online Retail Dataset (UCI) or create synthetic data

**Key Concepts Reinforced:**
- Unsupervised learning
- K-Means and hierarchical clustering
- PCA for visualization
- Feature scaling
- Cluster interpretation
- Business insights from ML

**Challenge Elements:**
- Determine optimal number of clusters
- Create customer personas from clusters
- Visualize high-dimensional data
- Provide actionable marketing recommendations
- Handle temporal aspects of customer behavior

---

### **Project 4: Sentiment Analysis System (Weeks 9-10)**
**Difficulty:** Intermediate-Advanced  
**Description:** Build a system that analyzes sentiment in product reviews or social media posts.

**Dataset:** IMDB Reviews, Twitter Sentiment, or Amazon Reviews

**Key Concepts Reinforced:**
- Text preprocessing (tokenization, stop words)
- Text vectorization (TF-IDF, word embeddings)
- Neural networks for NLP
- Recurrent Neural Networks basics (optional)
- Transfer learning with pre-trained embeddings
- Multi-class classification

**Challenge Elements:**
- Handle text data preprocessing independently
- Compare traditional ML (with TF-IDF) vs neural networks
- Deal with class imbalance
- Create a simple web interface for predictions
- Error analysis and model improvement

---

### **Project 5: Image Classification with Deployment (Weeks 11-12)**
**Difficulty:** Advanced  
**Description:** Build and deploy a complete image classification system (e.g., plant disease detection, product categorization) accessible via web interface.

**Dataset:** Plant Village, Fashion MNIST, or custom collected images

**Key Concepts Reinforced:**
- Computer vision with CNNs
- Transfer learning
- Data augmentation
- Model optimization for deployment
- API creation
- Cloud deployment
- Model monitoring basics
- Complete ML pipeline

**Challenge Elements:**
- Collect and label your own additional data
- Implement data augmentation
- Optimize model size vs accuracy tradeoff
- Create intuitive web interface
- Handle edge cases and errors gracefully
- Document the entire process
- Consider ethical implications

**Bonus Challenge:** Add continuous learning capability where the model can be retrained with new user-submitted images.

---

## **Learning Resources (80-20 Focused)**

### Essential Resources:
1. **Scikit-learn documentation** - Your primary reference
2. **Kaggle Learn** - Micro-courses on ML topics
3. **StatQuest with Josh Starmer** (YouTube) - Intuitive explanations
4. **Google Colab** - Free GPU for experiments

### Books (Reference Only):
- "Hands-On Machine Learning" by Aurélien Géron (practical focus)
- Python Data Science Handbook (free online)

### Practice Platforms:
- Kaggle (datasets + competitions)
- UCI Machine Learning Repository
- Google Dataset Search

---

## **Weekly Success Checklist**

Each week, ensure you:
- [ ] Understand the core concept (can you explain it simply?)
- [ ] Implemented code from scratch at least once
- [ ] Used a library implementation
- [ ] Applied it to a real dataset
- [ ] Documented your learning (blog post, notes, or GitHub)
- [ ] Struggled with at least one challenge (means you're learning!)

---

## **Study Tips for Maximum Efficiency**

1. **Active Learning:** Type every code example yourself; don't copy-paste
2. **Project-First Approach:** Start projects before you feel "ready"
3. **Debug Independently:** Spend 30 minutes debugging before searching for solutions
4. **Spaced Repetition:** Review previous weeks' concepts briefly each week
5. **Community:** Join ML Discord/Reddit, but limit time to 30 min/day
6. **80-20 Rule:** If stuck >2 hours, move on and return later
7. **Build in Public:** Share your projects on GitHub/LinkedIn

---

## **After 12 Weeks: Next Steps**

You'll have covered the core 20% that enables 80% of ML work. Continue with:

1. **Specialize:** Choose Computer Vision, NLP, or Time Series
2. **Compete:** Join Kaggle competitions
3. **Contribute:** Open source ML projects
4. **Build Portfolio:** 3-5 polished projects on GitHub
5. **Advanced Topics:** Deep RL, GANs, Transformers (based on interest)

---

## **Common Pitfalls to Avoid**

1. **Tutorial Hell:** Limit tutorials to 30% of your time; 70% should be building
2. **Math Paralysis:** Don't wait to master math; learn it as needed
3. **Tool Obsession:** Focus on concepts, not memorizing syntax
4. **Perfect Code:** Working > perfect; refactor later
5. **Isolation:** Share your work early and often

---

**Remember:** Machine Learning is learned by doing. Each week, prioritize hands-on coding over passive reading. The projects are where real learning happens. Struggle is a feature, not a bug—embrace it!

Good luck on your ML journey! 🚀
