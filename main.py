from Bayes_From_Scratch import NBC as Scratch_NBC
from Bayes_Scikit import NBC as Prebuilt_NBC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from SVM import CustomSVM

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

# Create and train the SVM model
svm = CustomSVM(C=1.0)
svm.fit(train_X, train_y)

# Make predictions on the test data
y_pred = svm.predict(test_X)

# Calculate accuracy
accuracy = accuracy_score(test_y, y_pred)
print(f"Custom SVM accuracy: {accuracy * 100} %")



# NBC
Scratch_NBC = Scratch_NBC()
Prebuilt_NBC = Prebuilt_NBC()
Scratch_NBC.predict()
Prebuilt_NBC.predict()