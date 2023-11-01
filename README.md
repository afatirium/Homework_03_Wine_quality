# Homework_03_Wine_quality
 Homework_03 for ML 2

# Exercises - Multi-Class Classification with Boosting Algorithms - White Wine Quality

In this exercise, I will ask you to use the white wine quality datasets (link: https://archive.ics.uci.edu/dataset/186/wine+quality) for the multi-class classification problem using boosting algorithms.

**Dataset description**: White Wine Quality dataset contains informations related to white and white variants of the Portuguese "Vinho Verde" wine.

Input variables (based on physicochemical tests):
1. fixed acidity
1. volatile acidity
1. citric acid
1. residual sugar
1. chlorides
1. free sulfur dioxide
1. total sulfur dioxide
1. density
1. pH
1. sulphates
1. alcohol

Output variable (based on sensory data):
12. quality (score between 0 and 10)

**Step 1: Data Loading**

1. Load the White Wine Quality (please omit Red Wine Quality) dataset.

**Step 2: Data Preprocessing and Splitting**

2. Create a target variable for multi-class classification. You can categorize wine quality scores into classes (e.g., low, medium, high) or keep them as is (but please use at least 3 classes).

3. Split dataset into features (X) and the target variable (y).

4. Split the data into training and testing sets using an 80-20 or similar ratio.

5. Perform any necessary SEDA, data preprocessing and feature engineering.

**Step 3: Feature Selection**

6. Apply initial feature selection techniques.

**Step 4: Model Training and Hyperparameter Tuning**

7. Train and fine-tune the following models separately using only your training data (is the data well balanced? if not maybe we can take this into account using some hyperparameters or we should rebalance our dataset using techniques from ML1 course?):

   * XGBoost
   * LightGBM
   * CatBoost
   * Classical Gradient Boosting Model
   * extra task: please use sklearn Histogram-Based Gradient Boosting model (alternative for LightGBM)
Please consider which evaluation metrics (https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel) and cost functions to choose!!! Your are solving multiclass classification problem!

8. Try to conduct the feature selection process based on the measures built into the models. Create a variable that is pure noise and does not add any value to the model (draw it from some distribution, e.g. a uniform distribution) and add it to the training set. If any variable is less important than the created noise, please remove it from the model. Of course, at the end, remove this noise from the model as well.


9. For each model, perform hyperparameter tuning using techniques like grid search or random search with cross-validation (e.g., GridSearchCV or RandomizedSearchCV). Tune hyperparameters such as learning rate, maximum depth, number of estimators, and any **model-specific hyperparameters - first of all, focus on HP that are related to overfitting**.

You can do task 8 and 9 in parallel, because testing variables should also be related to testing hyperparameters! Complete tasks 8 and 9 iteratively. Try different configurations.

**Step 5: Model Comparison**

10. Evaluate the tuned models on the  testing datasets using appropriate classification metrics (e.g., accuracy, precision, recall, F1-score) - please remember that you are solving multi-class classification problem so you have choose proper evaluation strategy - sometimes confusion matrix is the best!!!.

11. Compare the performance of XGBoost, LightGBM, CatBoost, and Classical Gradient Boosting. Discuss which model performed the best for our dataset and why.

# Exercises - Multi-Class Classification with Boosting Algorithms - White Wine Quality

In this exercise, I will ask you to use the white wine quality datasets (link: https://archive.ics.uci.edu/dataset/186/wine+quality) for the multi-class classification problem using boosting algorithms.

**Dataset description**: White Wine Quality dataset contains informations related to white and white variants of the Portuguese "Vinho Verde" wine.

Input variables (based on physicochemical tests):
1. fixed acidity
1. volatile acidity
1. citric acid
1. residual sugar
1. chlorides
1. free sulfur dioxide
1. total sulfur dioxide
1. density
1. pH
1. sulphates
1. alcohol

Output variable (based on sensory data):
12. quality (score between 0 and 10)

**Step 1: Data Loading**

1. Load the White Wine Quality (please omit Red Wine Quality) dataset.

**Step 2: Data Preprocessing and Splitting**

2. Create a target variable for multi-class classification. You can categorize wine quality scores into classes (e.g., low, medium, high) or keep them as is (but please use at least 3 classes).

3. Split dataset into features (X) and the target variable (y).

4. Split the data into training and testing sets using an 80-20 or similar ratio.

5. Perform any necessary SEDA, data preprocessing and feature engineering.

**Step 3: Feature Selection**

6. Apply initial feature selection techniques.

**Step 4: Model Training and Hyperparameter Tuning**

7. Train and fine-tune the following models separately using only your training data (is the data well balanced? if not maybe we can take this into account using some hyperparameters or we should rebalance our dataset using techniques from ML1 course?):

   * XGBoost
   * LightGBM
   * CatBoost
   * Classical Gradient Boosting Model
   * extra task: please use sklearn Histogram-Based Gradient Boosting model (alternative for LightGBM)
Please consider which evaluation metrics (https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel) and cost functions to choose!!! Your are solving multiclass classification problem!

8. Try to conduct the feature selection process based on the measures built into the models. Create a variable that is pure noise and does not add any value to the model (draw it from some distribution, e.g. a uniform distribution) and add it to the training set. If any variable is less important than the created noise, please remove it from the model. Of course, at the end, remove this noise from the model as well.


9. For each model, perform hyperparameter tuning using techniques like grid search or random search with cross-validation (e.g., GridSearchCV or RandomizedSearchCV). Tune hyperparameters such as learning rate, maximum depth, number of estimators, and any **model-specific hyperparameters - first of all, focus on HP that are related to overfitting**.

You can do task 8 and 9 in parallel, because testing variables should also be related to testing hyperparameters! Complete tasks 8 and 9 iteratively. Try different configurations.

**Step 5: Model Comparison**

10. Evaluate the tuned models on the  testing datasets using appropriate classification metrics (e.g., accuracy, precision, recall, F1-score) - please remember that you are solving multi-class classification problem so you have choose proper evaluation strategy - sometimes confusion matrix is the best!!!.

11. Compare the performance of XGBoost, LightGBM, CatBoost, and Classical Gradient Boosting. Discuss which model performed the best for our dataset and why.