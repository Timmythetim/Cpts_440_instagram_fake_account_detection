from Bayes_From_Scratch import NBC as Scratch_NBC
from Bayes_Scikit import NBC as Prebuilt_NBC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from SVM import CustomSVM
from Logistic_Regression_Scratch import CustomLogisticRegression

#SVM
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

train_X = train_df.drop("fake", axis=1)
train_y = train_df['fake'].to_numpy()  # Convert to NumPy array

test_X = test_df.drop("fake", axis=1)
test_y = test_df['fake'].to_numpy()  # Convert to NumPy array

# Preprocess the data (scaling)
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# Create and train the models
svm = CustomSVM(C=1.0)
svm.fit(train_X, train_y)

# Make predictions on the test data
y_pred_svm = svm.predict(test_X)

# Calculate accuracy
svm_accuracy = accuracy_score(test_y, y_pred_svm)

# Print accuracy
print(f"Custom SVM accuracy: {svm_accuracy * 100} %")

# NBC
Scratch_NBC = Scratch_NBC()
Prebuilt_NBC = Prebuilt_NBC()
Scratch_NBC.predict()
Prebuilt_NBC.predict()

# Logistic Regression
# Create and train the model
logistic_regression = CustomLogisticRegression(C=1.0, max_train_iterations=100)
logistic_regression.fit(train_X, train_y)

# Predict based on model
y_pred_LR = logistic_regression.predict(test_X)

# Find accuracy
lr_accuracy = accuracy_score(test_y, y_pred_LR)
print(f"Custom Logistic Regression accuracy: {lr_accuracy * 100} %")