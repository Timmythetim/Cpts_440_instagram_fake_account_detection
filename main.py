from Bayes_From_Scratch import NBC as Scratch_NBC
from Bayes_Scikit import NBC as Prebuilt_NBC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from SVM import CustomSVM
from Logistic_Regression_Scratch import CustomLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from DecisionForest_Scratch import SimpleDecisionForest
from Model_Hyper_Tuning import ClassifierTuning

# Data Processing
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

train_X = train_df.drop("fake", axis=1)
train_y = train_df['fake'].to_numpy()  # Convert to NumPy array

test_X = test_df.drop("fake", axis=1)
test_y = test_df['fake'].to_numpy()  # Convert to NumPy array

# Fix y to be between -1, 1
test_y = np.where(test_y == 0, -1, 1)
train_y = np.where(train_y == 0, -1, 1)

# Preprocess the data (scaling)
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# HyperTune Parameters
ClassifierTuning.decisionTreeHyperTuning(train_X, train_y, test_X, test_y)
ClassifierTuning.handmadeLogisticalRegressionHyperTuning(train_X, train_y, test_X, test_y)
ClassifierTuning.prebuiltLogisticalRegressionHyperTuning(train_X, train_y, test_X, test_y)
ClassifierTuning.handmadeSVMHyperTuning(train_X, train_y, test_X, test_y)
ClassifierTuning.prebuiltSVMHyperTuning(train_X, train_y, test_X, test_y)

# Custom SVM
svm = CustomSVM(C=1.0)
svm.fit(train_X, train_y)
y_pred_svm = svm.predict(test_X)
svm_accuracy = accuracy_score(test_y, y_pred_svm)
# Print accuracy
print(f"Custom SVM accuracy: {round(svm_accuracy * 100, 4)} %")

# Sci-Kit SVM
svm = SVC(C=1.0)
svm.fit(train_X, train_y)
y_pred_svm = svm.predict(test_X)
svm_accuracy = accuracy_score(test_y, y_pred_svm)
# Print accuracy
print(f"Sci-Kit SVM accuracy: {round(svm_accuracy * 100, 4)} %")

# NBC
Scratch_NBC = Scratch_NBC()
Prebuilt_NBC = Prebuilt_NBC()
Scratch_NBC.predict()
Prebuilt_NBC.predict()

# Logistic Regression
# Create and train the model
logistic_regression = CustomLogisticRegression(C = 0, learning_rate=0.001, max_train_iterations=100)
logistic_regression.fit(train_X, train_y)
y_pred_LR = logistic_regression.predict(test_X)
lr_accuracy = accuracy_score(test_y, y_pred_LR)
print(f"Custom Logistic Regression accuracy: {round(lr_accuracy * 100, 4)} %")

# Sci-Kit learn Logistic Regression
sci_log_regression = LogisticRegression()
sci_log_regression.fit(train_X, train_y)
y_pred_sci_lr = sci_log_regression.predict(test_X)
lr_sci_accuracy = accuracy_score(test_y, y_pred_sci_lr)
print(f"SciKit Logistic Regression accuracy: {round(lr_sci_accuracy * 100, 4)} %")

# Decision Forest
forest = SimpleDecisionForest(num_trees=10, max_depth=5)
forest.fit(train_X, train_y)
y_pred_forest = forest.predict(test_X)
forest_accuracy = accuracy_score(test_y, y_pred_forest)
print(f"Decision Forest accuracy: {round(forest_accuracy * 100, 4)} %")


