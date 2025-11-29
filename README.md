# Solar Flare Prediction: Comparative Machine Learning Analysis

## 🌠 Project Overview

[cite_start]This project implements and compares the performance of four core Machine Learning (ML) algorithms—**Gradient Boosting (XGBoost)**, **Random Forest (RF)**, **Support Vector Machines (SVM)**, and a **Deep Artificial Neural Network (ANN)**—for forecasting **Solar Flares**[cite: 6]. [cite_start]The objective is to classify whether a major solar flare (C, M, or X class) will occur within a 24-hour window, based on observable features of active solar regions (sunspots)[cite: 7, 98].

[cite_start]A central challenge in this domain is the **severe class imbalance** inherent in space weather data (flares are rare events)[cite: 22, 31, 40]. [cite_start]This work addresses this by utilizing the **SMOTETomek hybrid resampling technique** [cite: 22] to ensure robust and non-biased model training.

### 🎯 Key Scientific Objectives

* [cite_start]**Binary Classification:** Convert the multi-class flare count problem into a binary classification: "Flare Will Occur (1)" vs. "Flare Will Not Occur (0)"[cite: 98].
* [cite_start]**Ensemble vs. Kernel vs. Deep Learning:** Empirically compare the predictive power of ensemble methods (RF, XGBoost) against kernel-based (SVM) and deep learning models (ANN/MLP)[cite: 6, 23].
* [cite_start]**Mitigating Bias:** Demonstrate the impact of **SMOTETomek** in improving the classification of the minority class (flares), a metric often prioritized over general accuracy in critical forecasting tasks[cite: 76].

### ⚙️ Methodology and Data

#### 1. Data Source and Characteristics
* [cite_start]**Dataset:** Solar Flare Data Set from the UCI Machine Learning Repository[cite: 7, 88].
* [cite_start]**Total Observations:** 1,065 rows[cite: 165].
* [cite_start]**Original Imbalance:** 81.1% "No Flare" (0) vs. 18.9% "Flare" (1)[cite: 161].
* [cite_start]**Features:** Includes 13 features describing active regions, such as Modified Zurich Class (`Zurich_Class`), Spot Size (`Spot_Size`), Spot Distribution (`Spot_Distribution`), and Activity level[cite: 91, 92].

#### 2. Data Preprocessing Pipeline

The following steps were applied sequentially to prepare the data for modeling:

1.  [cite_start]**Feature Conversion:** Categorical features (Zurich Class, Spot Size, Distribution) were converted using **One-Hot Encoding** to create a numerical matrix suitable for SVM and ANN[cite: 173].
2.  [cite_start]**Splitting:** Data was split into 70% for Training and 30% for Testing, ensuring the original class distribution was maintained (**stratified sampling**)[cite: 193].
3.  [cite_start]**Standard Scaling:** Numerical features were normalized using `StandardScaler` (Mean = 0, Standard Deviation = 1) to prevent features with larger scales from dominating the distance-based algorithms (SVM, ANN)[cite: 176]. [cite_start]The formula used is $z=\frac{x-\mu}{\sigma}$[cite: 178].
4.  **Hybrid Resampling (SMOTETomek):**
    * [cite_start]**SMOTE** was used to synthetically oversample the minority class (Flare)[cite: 182].
    * [cite_start]**Tomek Links** were applied to clean noisy data points near the decision boundary, enhancing the clarity of the classification margin[cite: 183].
    * [cite_start]**Result:** The training data was perfectly balanced (50% Flare, 50% No Flare)[cite: 184].

### 📊 Results and Comparative Performance

Models were trained on the balanced training set and evaluated on the untouched test set (30% of original data).

#### Final Accuracy Comparison

| Algorithm | Accuracy (%) | Rank |
| :--- | :--- | :--- |
| **Gradient Boosting (XGBoost)** | **76.88** | **1. (Best)[cite_start]** [cite: 8] |
| **Random Forest** | 74.69 | [cite_start]2. [cite: 9] |
| **SVM (Optimized)** | 72.81 | [cite_start]3. [cite: 9] |
| **Artificial Neural Networks (ANN)** | 71.25 | [cite_start]4. [cite: 9] |

#### Detailed Model Insights

* [cite_start]**Gradient Boosting (Best Performer):** Achieved the highest accuracy by sequentially training weak learners (shallow trees) to minimize the residuals (errors) of the previous models[cite: 59, 392]. [cite_start]This mechanism effectively modeled the complex, non-linear patterns within the solar data[cite: 256].
* [cite_start]**Random Forest:** Performed strongly, demonstrating the stability and reduced overfitting risk of ensemble methods through majority voting across multiple decision trees[cite: 37, 393].
* **SVM and ANN (Lower Performers):**
    * [cite_start]**SVM:** The dominance of categorical features (converted to a sparse matrix via One-Hot Encoding) challenged the hyperplane separation, causing it to fall behind the tree-based models[cite: 396, 397].
    * [cite_start]**ANN:** The model used a 3-layer deep architecture (100-50-25 neurons)[cite: 337], which typically requires a much larger dataset (millions of rows) to unleash its full potential. [cite_start]The relatively small size of this dataset (approx. 1,000 rows) constrained the deep network's ability to generalize effectively[cite: 347, 400].

### 📈 Confusion Matrix Visualization

The project generates individual confusion matrices and classification reports for each model. [cite_start]The reports emphasize **Recall** (Duyarlılık), which is vital in space weather, as a missed flare (False Negative) can result in catastrophic damage[cite: 76, 411].

[Example Confusion Matrix Image Placeholder]

---

## 🛠️ Setup and Execution

This project is implemented in Python and executed within a single Jupyter Notebook.

### Prerequisites

* Python 3.10+
* The project relies on the following key libraries: `pandas`, `numpy`, `sklearn`, `imblearn` (for SMOTETomek), and visualization tools (`matplotlib`, `seaborn`).

### Installation and Run Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/solar-flare-prediction-ml-comparison.git](https://github.com/YourUsername/solar-flare-prediction-ml-comparison.git)
    cd solar-flare-prediction-ml-comparison
    ```
2.  **Open the Notebook:** Upload the `SolarFlareForecast.ipynb` file to your preferred environment (Google Colab, Jupyter Lab, etc.).
3.  [cite_start]**Data Loading:** The notebook automatically loads the data directly from the UCI repository URL[cite: 7, 88].
4.  **Execute:** Run all cells sequentially (e.g., Run All in Google Colab). The notebook will:
    * Load and pre-process the raw data (Encoding, Scaling).
    * Apply the **SMOTETomek** technique to balance the training set.
    * Train and evaluate the four different ML models.
    * Generate the comparative table, confusion matrices, and classification reports.

### Future Work Recommendations

[cite_start]Based on this analysis, the following avenues are suggested for future research[cite: 414]:

* [cite_start]**Image Integration:** Incorporate actual solar image data (e.g., from NASA's SDO satellite) and process it using **Convolutional Neural Networks (CNN)** to create a powerful hybrid prediction model[cite: 416].
* [cite_start]**Time Series Analysis:** Utilize **LSTM** (Long Short-Term Memory) or other Recurrent Neural Networks to analyze sequential data from the last 24-48 hours, treating flares as the result of a process rather than an instantaneous event[cite: 417, 418].
* [cite_start]**Advanced Optimization:** Employ techniques like **Bayesian Optimization** or **Genetic Algorithms** to fine-tune the hyperparameters of the Gradient Boosting model for even greater precision[cite: 419].
