# Machine Learning Fundamentals
### A Comprehensive Journey from Mathematical Foundations to Production-Ready Implementations

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)

> **"After 28 years building distributed systems, I'm bridging the gap between operational and analytical paradigms—learning ML deeply to architect Software 4.0: self-adapting operational systems."**

---

## 📚 Table of Contents

- [Overview](#overview)
- [Why This Project](#why-this-project)
- [What Makes This Different](#what-makes-this-different)
- [Repository Structure](#repository-structure)
- [Models Implemented](#models-implemented)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Learning Journey](#learning-journey)
- [What's Next: Software 4.0](#whats-next-software-40)
- [Who This Is For](#who-this-is-for)
- [Contributing](#contributing)
- [License](#license)
- [Connect](#connect)

---

## 🎯 Overview

This repository represents a systematic, mathematically-grounded exploration of machine learning fundamentals. Built over three years of deep study, it provides **21 professional Jupyter notebooks** covering supervised learning, unsupervised learning, and neural networks—with a focus on understanding *why* algorithms work, not just *how* to use them.

**Core Philosophy:** Learn the mathematics first, implement with rigor, compare systematically.

**Ultimate Goal:** Build the foundation for **Software 4.0** - operational systems that learn and adapt continuously from their own events.

---

## 💡 Why This Project

### The Software Evolution

**Software 1.0** (Traditional Programming)
- Explicit instructions in C++, Java, Python
- Deterministic logic, hard-coded rules
- Limited by human ability to anticipate scenarios

**Software 2.0** (Neural Networks - Andrej Karpathy)
- Learned from data via backpropagation
- Weights optimized, not manually coded
- Revolutionized vision, speech, NLP

**Software 3.0** (LLMs as Operating Systems - Andrej Karpathy, 2025)
- Programming in natural language
- LLMs democratize software creation
- "Vibe coding" - anyone can build apps

**Software 4.0** (This Vision - Analytics-Driven Microservices)
- **Operational systems that learn from their own events**
- Business logic adapts continuously, not hard-coded
- Deterministic ML models trained in real-time from streaming data
- **Where Karpathy's vision stops, this one starts**

---

### The Problem I Saw

After nearly three decades architecting distributed systems—from mainframe trading platforms at major French banks to cloud-native microservices at Confluent—I faced a fundamental limitation:

**Even in Software 3.0, business rules remain static.**

- Software 3.0 revolutionizes HOW we BUILD systems (LLMs help developers code)
- But OPERATIONAL SYSTEMS still run hard-coded logic deployed weeks ago
- Business rules don't learn from production events in real-time
- We've evolved data pipelines (batch → streaming) but not analytical intelligence

---

### The Vision: Software 4.0

**What if microservices could learn business logic dynamically from streaming events?**

Not LLM-powered chatbots. Not AI coding assistants. But **production systems where the business logic itself continuously learns and adapts**.

This is the paradigm shift I've been working toward for 28 years:
- **Software 1.0→2.0**: From explicit code to learned weights
- **Software 2.0→3.0**: From batch training to natural language interfaces  
- **Software 3.0→4.0**: From static deployments to **continuous runtime learning**

---

### The GenAI Detour (Why Not Software 3.0?)

Early 2025, I built an internal sales forecasting dashboard with Gemini (Software 3.0 approach). The results were impressive, but I faced critical issues:

**Problems with LLMs for Production Systems:**
1. **Probabilistic** - Can hallucinate, unsuitable for business-critical decisions
2. **High Latency** - 100ms-1s inference, too slow for real-time (<10ms required)
3. **Not Deterministic** - Can't audit "why" a decision was made
4. **Expensive** - LLM inference on every event is cost-prohibitive
5. **Static** - Don't learn from production data continuously

**I spent more time tuning prompts than solving problems.**

Software 3.0 is transformative for BUILDING software, but not for RUNNING mission-critical operational systems.

---

### The Philosophical Alignment

Then I heard Yann LeCun articulate his critique of autoregressive LLMs:

> *"Autoregressive LLMs don't really understand the physical world. They don't really have persistent memory. They can't really reason and they certainly can't plan. There's absolutely no way that autoregressive LLMs, the type that we know today, will reach human intelligence."*

**For the first time, one of the pioneers of modern AI was articulating exactly what I had concluded**—that we need fundamentally different approaches for production business-critical systems.

---

### The Solution: Deterministic ML + Streaming Architecture

**Software 4.0 Architecture:**
```
Kafka Event Stream → Real-Time Feature Engineering (Flink)
                              ↓
                    Online Learning Models
                    (Decision Trees, Random Forests)
                              ↓
                  Continuous Model Training
                  (Learn from every event)
                              ↓
            Deterministic Predictions (<10ms latency)
                              ↓
          Business Rules Adapt Automatically
```

**This is already proven in production**:
- **Netflix**: Continuous learning for recommendations
- **Uber**: Real-time fraud detection with incremental updates
- **LinkedIn**: Feed ranking with online gradient descent
- **Google Ads**: CTR prediction with FTRL algorithm

But to architect these systems effectively, I needed to understand ML deeply. This repository is that foundation.

---

## 🌟 What Makes This Different

### 1. **Built on Mathematical Understanding**
These implementations leverage scikit-learn and TensorFlow, but they're built from a foundation of deep mathematical study:
- Studied linear algebra (eigenvectors, covariance matrices) to understand PCA
- Learned calculus (gradient descent, backpropagation) to grasp neural networks
- Mastered statistics (entropy, information gain) to comprehend decision trees
- Understanding *why* algorithms work enables better architectural decisions

**Note**: Phase 2 (Q1 2026) will include from-scratch implementations showing these mathematical foundations directly in code.

### 2. **Professional Code Architecture**
This isn't "notebook soup." Every model has:
- **Wrapper classes** with consistent interfaces (`fit`, `predict`, `score`)
- **Reusable preprocessing pipelines** 
- **Built-in evaluation utilities** (cross-validation, grid search)
- **Comparison frameworks** for systematic analysis

### 3. **Self-Explanatory Structure**
Notebooks are organized to answer the questions I couldn't find answered elsewhere:
- **Which model** for which problem?
- **When to use** specific techniques (K-Means vs DBSCAN vs Hierarchical)?
- **How to tune** hyperparameters (grid search, cross-validation)?
- **Why visualize** with PCA vs LDA?

This is the structure I couldn't find when learning ML—clear guidance on model selection, appropriate metrics, and tuning techniques.

### 4. **Comprehensive Comparisons**
Side-by-side comparisons across all model families:
- Classification models (6 algorithms)
- Regression models (6 algorithms)  
- Clustering models (3 algorithms)
- Unified metrics for fair evaluation

Most tutorials show individual models. This shows how to *choose* between them.

### 5. **Foundation for Software 4.0**
Every algorithm studied here has a role in continuous learning systems:
- **Decision Trees** → Hoeffding Trees (streaming version)
- **Random Forests** → Adaptive Random Forests
- **Linear Models** → Online Gradient Descent
- **All models** → Incremental learning variants

---

## 📁 Repository Structure
```
ml-fundamentals/
│
├── notebooks/                          # 21 Professional Notebooks
│   ├── supervised/
│   │   ├── classification/             # 7 notebooks
│   │   │   ├── 01_logistic_regression.ipynb
│   │   │   ├── 02_decision_tree.ipynb
│   │   │   ├── 03_random_forest.ipynb
│   │   │   ├── 04_knn.ipynb
│   │   │   ├── 05_naive_bayes.ipynb
│   │   │   ├── 06_svm.ipynb
│   │   │   └── 07_classification_comparison.ipynb
│   │   │
│   │   └── regression/                 # 7 notebooks
│   │       ├── 01_linear_regression.ipynb
│   │       ├── 02_polynomial_regression.ipynb
│   │       ├── 03_ridge_regression.ipynb
│   │       ├── 04_lasso_regression.ipynb
│   │       ├── 05_decision_tree_regression.ipynb
│   │       ├── 06_random_forest_regression.ipynb
│   │       └── 07_regression_comparison.ipynb
│   │
│   ├── unsupervised/                   # 5 notebooks
│   │   ├── 01_kmeans.ipynb
│   │   ├── 02_dbscan.ipynb
│   │   ├── 03_hierarchical.ipynb
│   │   ├── 04_pca.ipynb
│   │   └── 05_clustering_comparison.ipynb
│   │
│   └── neural_networks/                # 2 notebooks
│       ├── 01_ann_classification.ipynb
│       └── 02_cnn_image_classification.ipynb
│
├── src/                                # Professional Code Architecture
│   ├── models/
│   │   ├── supervised/
│   │   │   ├── classification/
│   │   │   │   ├── sklearn_base.py
│   │   │   │   └── sklearn_classification.py
│   │   │   └── regression/
│   │   │       ├── sklearn_base.py
│   │   │       └── sklearn_regression.py
│   │   ├── unsupervised/
│   │   │   ├── sklearn_base.py
│   │   │   ├── sklearn_clustering.py
│   │   │   └── sklearn_pca.py
│   │   └── neural_networks/
│   │       └── keras_models.py
│   │
│   ├── data_loader.py                  # Dataset utilities
│   ├── data_preprocessing.py           # Preprocessing pipeline
│   └── data_utils.py                   # Helper functions
│
├── .gitignore
├── LICENSE
├── pyproject.toml                      # Dependencies & project config
└── README.md
```

---

## 🤖 Models Implemented

### Supervised Learning

#### Classification (6 Algorithms)
| Model | Use Case | Key Features |
|-------|----------|--------------|
| **Logistic Regression** | Binary/multi-class classification | Interpretable, probability estimates, fast |
| **Decision Tree** | Non-linear classification | Visual rules, no scaling needed |
| **Random Forest** | Ensemble classification | Robust, feature importance, handles overfitting |
| **K-Nearest Neighbors** | Instance-based learning | No training phase, good for small datasets |
| **Naive Bayes** | Probabilistic classification | Fast, works well with small data |
| **Support Vector Machine** | High-dimensional classification | Powerful, kernel trick for non-linear boundaries |

#### Regression (6 Algorithms)
| Model | Use Case | Key Features |
|-------|----------|--------------|
| **Linear Regression** | Simple linear relationships | Interpretable, fast, baseline model |
| **Polynomial Regression** | Non-linear patterns | Captures curves, feature engineering |
| **Ridge Regression** | Regularized linear model | L2 penalty, handles multicollinearity |
| **Lasso Regression** | Feature selection | L1 penalty, automatic feature selection |
| **Decision Tree Regressor** | Non-linear regression | Visual rules, handles non-linearity |
| **Random Forest Regressor** | Ensemble regression | Robust, feature importance |

### Unsupervised Learning

#### Clustering (3 Algorithms)
| Model | Use Case | Key Features |
|-------|----------|--------------|
| **K-Means** | Spherical clusters | Fast, scalable, centroid-based |
| **DBSCAN** | Arbitrary-shaped clusters | Density-based, outlier detection, no K needed |
| **Hierarchical** | Hierarchical structure | Dendrogram visualization, no K needed initially |

#### Dimensionality Reduction (2 Algorithms)
| Model | Use Case | Key Features |
|-------|----------|--------------|
| **PCA** | Unsupervised reduction | Maximum variance, visualization, decorrelation |
| **LDA** | Supervised reduction | Maximizes class separation, classification preprocessing |

### Neural Networks

| Model | Use Case | Key Features |
|-------|----------|--------------|
| **ANN (Multi-Layer Perceptron)** | Tabular data | Fully-connected, non-linear patterns |
| **CNN (Convolutional Neural Network)** | Image data | Spatial features, translation invariant |

---

## ✨ Key Features

### 1. Model Wrapper Classes
Every model has a consistent interface:
```python
from src.models.supervised.classification import SKLearnRandomForestClassifier

model = SKLearnRandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

### 2. Hyperparameter Optimization
Built-in methods for finding optimal parameters:
```python
# K-Means: Find optimal K
results = model.find_optimal_k(X, k_range=range(2, 11), method='both')

# DBSCAN: Find optimal eps
results = model.find_optimal_eps(X, eps_range=np.arange(0.1, 2.0, 0.1))
```

### 3. Advanced Evaluation
Comprehensive metrics and visualizations:
```python
# K-fold cross-validation
cv_results = k_fold_cross_validation(model, X, y, k=5)

# Grid search with cross-validation
best_params, best_model = grid_search_cv(model, param_grid, X, y, cv=5)
```

### 4. Intelligent Visualization
Built-in plotting for model insights:
```python
# Feature importance (tree-based models)
plot_feature_importance(model, feature_names, top_n=10)

# Cluster visualization with PCA
pca.plot_clusters_2d(X, labels, cluster_centers=centroids)

# Silhouette analysis
model.plot_silhouette_analysis(X)
```

### 5. Systematic Comparisons
Compare all models in a family:
- **Classification Comparison**: Accuracy, precision, recall, F1
- **Regression Comparison**: R², RMSE, MAE, overfitting analysis
- **Clustering Comparison**: Silhouette, cluster count, noise detection

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.11+
uv (recommended) or pip
```

### Installation

**Using uv (Recommended)**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/dcris19740101/ml-fundamentals.git
cd ml-fundamentals

# Create virtual environment and install dependencies
# uv will read pyproject.toml for exact versions
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Launch Jupyter
jupyter notebook
```

**Using pip**
```bash
# Clone the repository
git clone https://github.com/dcris19740101/ml-fundamentals.git
cd ml-fundamentals

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (versions specified in pyproject.toml)
pip install -e .

# Launch Jupyter
jupyter notebook
```

### Quick Start
```python
# Run any notebook
jupyter notebook notebooks/supervised/classification/01_logistic_regression.ipynb

# Or use the models directly
from src.models.supervised.classification import SKLearnLogisticRegression
from src.data_loader import load_titanic
from src.data_preprocessing import DataPreprocessor

# Load data
X, y = load_titanic()

# Preprocess
preprocessor = DataPreprocessor(scale=True, encode_categorical=True)
X_processed = preprocessor.fit_transform(X)

# Train model
model = SKLearnLogisticRegression()
model.fit(X_processed, y)

# Evaluate
accuracy = model.score(X_processed, y)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 📖 Learning Journey

### Timeline
- **Started**: Early 2022 (mathematics foundations)
- **Deep dive**: Early 2025 (ML implementation)
- **Completed**: December 31, 2025 (Phase 1)

### Resources That Shaped This Project

**Courses:**
- [Stanford Statistical Learning](https://www.statlearning.com/) (with Python)
- [deeplearning.ai: Mathematics for ML & Data Science](https://www.deeplearning.ai/courses/mathematics-for-machine-learning-and-data-science-specialization/)
  - ✅ Linear Algebra (completed)
  - ✅ Calculus (completed)
  - 🔄 Statistics & Probability (in progress)
- [MIT 18.06: Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (Gilbert Strang)
- [CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/) (Fei-Fei Li)
- [StatQuest ML Courses](https://www.youtube.com/c/joshstarmer) (Josh Starmer)

**Books:**
- *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman)
- *Why Machines Learn: The Elegant Math Behind Modern AI* (Anil Ananthaswamy)

**YouTube:**
- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) 🎯 **Essential**

### Breakthrough Moments

**The Mathematical Foundation**  
> "After watching 3Blue1Brown's linear algebra series, then diving deep into MIT's course with the amazing Professor Gilbert Strang and Stanford's CS231n with Professor Fei-Fei Li, I fell in love with the mathematics behind machine learning. The elegance of how linear algebra, calculus, and probability work together to enable learning was transformative."

**The Philosophical Validation**  
> "Hearing Yann LeCun explain his vision for JEPA and his critique of autoregressive LLMs was profound. He stated: *'Autoregressive LLMs don't really understand the physical world. They don't really have persistent memory. They can't really reason and they certainly can't plan.'* For the first time, one of the pioneers of modern AI was articulating exactly what I had been convinced of from the beginning—that we need fundamentally different approaches beyond probabilistic text generation for business-critical, deterministic systems."

This convergence of mathematical beauty and architectural conviction solidified my path: build deterministic, mathematically-grounded ML systems for production use cases.

---

## 🔮 What's Next: Production Grade Software 4.0

### Phase 2: From-Scratch Implementations (Q1 2026)
Rebuild every algorithm from mathematical first principles—**this is where the mathematical rigor becomes explicit in code**:
- Implement gradient descent from calculus fundamentals
- Build PCA using eigenvector decomposition of covariance matrices
- Code decision trees using entropy and information gain calculations
- Pure NumPy implementations—no scikit-learn, no TensorFlow
- Educational focus: *show* the mathematics, not just *use* it

---

### Phase 3: Deep Learning & Transformers (Q2 2026)
Master modern architectures that complement Software 4.0:
- **Neural Networks & CNNs from scratch** (backpropagation, convolution)
- **RNNs and LSTMs** (sequence modeling fundamentals)
- **Transformers**: Full implementation from "Attention Is All You Need"
  - Self-attention mechanism deep dive
  - Multi-head attention
  - Positional encoding
  - Train on machine translation task

**Why learn transformers?**
- They're the foundation of Software 3.0 (Karpathy's LLMs)
- Understand their strengths (natural language) and limitations (production systems)
- Use them strategically where appropriate (e.g., event understanding, feature generation)
- Know when NOT to use them (real-time deterministic decisions)

---

### Phase 4: Software 4.0 - Production Grade Analytics-Driven Microservices (Q3-Q4 2026) ⭐

**The Vision**: Where Karpathy's Software 3.0 stops, Software 4.0 begins.

#### **The Paradigm Shift**

**Software 3.0 (Karpathy)**:
- LLMs help humans BUILD software faster
- Natural language as programming interface
- Revolutionizes software DEVELOPMENT

**Software 4.0 (This Vision)**:
- Operational systems LEARN from their own events
- Business logic adapts continuously at RUNTIME
- Revolutionizes software OPERATIONS

#### **Architecture**
```
┌─────────────────────────────────────────────────────┐
│  SOFTWARE 3.0 LAYER (Development)                   │
│  ─────────────────────────────────────────          │
│  Use LLMs to:                                       │
│  • Design architecture                              │
│  • Generate initial code                            │
│  • Debug and optimize                               │
│  • Create documentation                             │
└─────────────────────────────────────────────────────┘
                        ↓
              Produces & deploys
                        ↓
┌─────────────────────────────────────────────────────┐
│  SOFTWARE 4.0 LAYER (Production Runtime)            │
│  ─────────────────────────────────────────          │
│  Kafka Event Stream                                 │
│         ↓                                           │
│  Real-Time Feature Engineering (Flink)              │
│         ↓                                           │
│  Online Learning Models (River, Flink ML)           │
│  • Hoeffding Trees                                  │
│  • Adaptive Random Forests                          │
│  • Online Gradient Descent                          │
│         ↓                                           │
│  Continuous Training (every event updates model)    │
│         ↓                                           │
│  Deterministic Predictions (<10ms latency)          │
│         ↓                                           │
│  Business Rules Evolve Dynamically                  │
└─────────────────────────────────────────────────────┘
```

#### **Key Technologies**

**Online Learning Framework**:
- **River**: Python library for streaming ML
- **Hoeffding Trees**: Decision trees for data streams
- **Adaptive Random Forests**: Ensemble methods with drift detection
- **Vowpal Wabbit**: Massive-scale online learning

**Streaming Infrastructure**:
- **Apache Kafka**: Event backbone
- **Apache Flink**: Stream processing & ML
- **Flink ML**: Native streaming machine learning
- **Feature Stores**: Feast for feature management

**Model Management**:
- **MLflow**: Model versioning and registry
- **Evidently**: Drift detection and monitoring
- **SHAP**: Explainability for online models

#### **The Innovation: Real-Time Training**

**Current Paradigm (Even in Software 3.0)**:
```
1. Train model offline (batch)
2. Deploy static model
3. API calls for predictions
4. Model stays frozen
5. Retrain weekly/monthly
```

**Software 4.0 Paradigm**:
```
1. Event arrives on Kafka
2. Model predicts (deterministic, <10ms)
3. Model LEARNS immediately from event
4. Next prediction uses updated knowledge
5. Business rules adapt continuously
6. Zero deployment lag
```

#### **Concrete Example: Fraud Detection**

**Traditional Approach**:
```python
# Static model deployed weeks ago
if transaction.amount > 1000:
    flag_as_suspicious()  # Hard-coded rule
```

**Software 4.0 Approach**:
```python
from river import tree
from kafka import KafkaConsumer

# Model learns continuously
model = tree.HoeffdingTreeClassifier()

for event in KafkaConsumer('transactions'):
    # Extract features in real-time
    features = {
        'amount': event.amount,
        'velocity': count_recent_transactions(event.user),
        'location_change': detect_location_anomaly(event),
        'device_fingerprint': hash_device(event.device)
    }
    
    # Predict (deterministic, <5ms)
    risk_score = model.predict_proba_one(features)[1]
    
    if risk_score > 0.8:
        flag_transaction(event)
    
    # Learn immediately when label arrives
    if event.has_fraud_label:
        model.learn_one(features, event.is_fraud)
    
    # Model is now smarter for next transaction
    # Business rule (0.8 threshold) could also adapt
```

**Benefits**:
- ✅ Adapts to new fraud patterns in real-time
- ✅ No retraining pipeline needed
- ✅ Deterministic (auditable decisions)
- ✅ Sub-10ms latency
- ✅ Interpretable (can extract rules)

#### **Production-Proven Companies**

Software 4.0 isn't theoretical - it's already running at scale:

**Netflix**:
- Continuous learning for recommendations
- Models update from user interactions in minutes
- Kafka + Flink architecture

**Uber**:
- Real-time fraud detection
- Dynamic pricing with online learning
- Michelangelo platform

**LinkedIn**:
- Feed ranking with continuous updates
- Online logistic regression
- Billions of events/day

**Google Ads**:
- CTR prediction with FTRL algorithm
- Models retrain continuously
- Massive scale (billions of impressions)

#### **Impact: The Convergence**

**What Software 4.0 Achieves**:
- Operational systems (event-driven, real-time)
- Analytical systems (ML-powered, adaptive)
- **Converge into a single paradigm**

**The business logic IS the learned model**
**The model IS the operational system**

This is the architecture I've been envisioning for 28 years.

---

### Phase 5: Advanced Research & Production Hardening (2028+)

**Online Learning at Scale**:
- Multi-armed bandits for A/B testing
- Contextual bandits for personalization
- Concept drift detection and adaptation
- Ensemble methods for streaming data

**Explainability & Governance**:
- SHAP values for online models
- Audit trails for regulatory compliance
- Model fairness monitoring
- Counterfactual explanations

**Advanced Architectures**:
- Hierarchical online learning
- Transfer learning in streaming contexts
- Meta-learning for fast adaptation
- Causal inference for business insights

**Production Infrastructure**:
- Multi-region deployment
- A/B testing frameworks
- Shadow mode evaluation
- Automated rollback on drift

**Research Exploration** (Observational):
- **JEPA**: Yann LeCun's vision (research phase)
- Emerging self-supervised methods
- Next-generation architectures

---

## 🎓 Who This Is For

- **ML Practitioners** seeking mathematically rigorous implementations
- **Software Engineers** transitioning to ML/AI
- **Solution Engineers** architecting ML systems
- **System Architects** designing Software 4.0 platforms
- **Anyone** interested in continuous learning systems
- **Visionaries** who believe operational and analytical systems should converge

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional algorithms (XGBoost, LightGBM, etc.)
- Online learning implementations
- Streaming ML examples
- More datasets and use cases
- Performance optimizations
- Documentation improvements

Please open an issue first to discuss proposed changes.

---

## 📫 Connect

**Christian Dubois**
- **GitHub**: [@dcris19740101](https://github.com/dcris19740101)
- **LinkedIn**: [Christian Dubois](https://www.linkedin.com/in/christian-dubois-confluent)
- **Role**: Senior Solution Engineer @ Confluent
- **Background**: 28 years in distributed systems, cloud-native architecture, event streaming
- **Certifications**: CKAD (Certified Kubernetes Application Developer)
- **Vision**: Architecting Software 4.0 - where operational systems learn continuously

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Stanford**, **deeplearning.ai**, and **MIT OpenCourseWare** for publicly available ML resources
- **Professor Gilbert Strang** for making linear algebra beautiful
- **Professor Fei-Fei Li** for exceptional computer vision course
- **Yann LeCun** for his vision and critique that validated my convictions
- **Andrej Karpathy** for defining Software 3.0 and inspiring Software 4.0
- **3Blue1Brown** for making mathematics visual and intuitive
- **The ML community** for open-source tools and knowledge sharing
- **Netflix, Uber, LinkedIn, Google** for proving continuous learning works at scale

---

## 📊 Project Stats

- **21 Professional Notebooks**
- **15+ Algorithms Implemented**
- **3 Years of Mathematical Study**
- **Production-Ready Code Architecture**
- **Comprehensive Documentation**
- **Foundation for Software 4.0**

---

<div align="center">

**Built with** ❤️ **and a lot of** 📐 **mathematics**

*"Software 3.0 revolutionizes how we BUILD systems.*  
*Software 4.0 revolutionizes how those systems BEHAVE in production."*

**— Christian Dubois, December 31, 2025**

[⭐ Star this repo](https://github.com/dcris19740101/ml-fundamentals) if you're ready for Software 4.0!

</div>
