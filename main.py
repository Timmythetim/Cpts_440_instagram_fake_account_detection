from Bayes_From_Scratch import NBC as Scratch_NBC
from Bayes_Scikit import NBC as Prebuilt_NBC
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from SVM import CustomSVM
from Logistic_Regression_Scratch import CustomLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from DecisionForest_Scratch import SimpleDecisionForest
from Model_Hyper_Tuning import ClassifierTuning
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
# ClassifierTuning.decisionTreeHyperTuning(train_X, train_y, test_X, test_y)
# ClassifierTuning.handmadeLogisticalRegressionHyperTuning(train_X, train_y, test_X, test_y)
# ClassifierTuning.prebuiltLogisticalRegressionHyperTuning(train_X, train_y, test_X, test_y)
# ClassifierTuning.handmadeSVMHyperTuning(train_X, train_y, test_X, test_y)
# ClassifierTuning.prebuiltSVMHyperTuning(train_X, train_y, test_X, test_y)

# Model Training
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="fake")

model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="fake")

predictions = model.predict(test_ds)
threshold = 0.5
predictions = (predictions > threshold).astype(int)
test_y_val = test_df['fake'].to_numpy()
tf_forest_metrics = {
    'Accuracy': accuracy_score(test_y_val, predictions),
    'F1 Score': f1_score(test_y_val, predictions),
    'Confusion Matrix': confusion_matrix(test_y_val, predictions),
    'ROC AUC': roc_auc_score(test_y_val, predictions)
}


# Custom SVM
svm = CustomSVM(C=0.1)
svm.fit(train_X, train_y)
y_pred_svm = svm.predict(test_X)
custom_svm_metrics = {
    'Accuracy': accuracy_score(test_y, y_pred_svm),
    'F1 Score': f1_score(test_y, y_pred_svm),
    'Confusion Matrix': confusion_matrix(test_y, y_pred_svm),
    'ROC AUC': roc_auc_score(test_y, y_pred_svm)
}

# Sci-Kit SVM
svm = SVC(C=10)
svm.fit(train_X, train_y)
y_pred_svm = svm.predict(test_X)
svm_metrics = {
    'Accuracy': accuracy_score(test_y, y_pred_svm),
    'F1 Score': f1_score(test_y, y_pred_svm),
    'Confusion Matrix': confusion_matrix(test_y, y_pred_svm),
    'ROC AUC': roc_auc_score(test_y, y_pred_svm)
}

# NBC
# Print accuracy
scratch_NBC = Scratch_NBC()
custom_NBC_y = scratch_NBC.predict()
custom_NBC_y = np.array(custom_NBC_y)
custom_NBC_y = np.where(custom_NBC_y == 0, -1, 1)
custom_nbc_metrics = {
    'Accuracy': accuracy_score(test_y, custom_NBC_y),
    'F1 Score': f1_score(test_y, custom_NBC_y),
    'Confusion Matrix': confusion_matrix(test_y, custom_NBC_y),
    'ROC AUC': roc_auc_score(test_y, custom_NBC_y)
}
prebuilt_NBC = Prebuilt_NBC()
prebuild_NBC_y = prebuilt_NBC.predict()
prebuild_NBC_y = np.array(prebuild_NBC_y)
prebuild_NBC_y = np.where(prebuild_NBC_y == 0, -1, 1)
prebuild_nbc_metrics = {
    'Accuracy': accuracy_score(test_y, prebuild_NBC_y),
    'F1 Score': f1_score(test_y, prebuild_NBC_y),
    'Confusion Matrix': confusion_matrix(test_y, prebuild_NBC_y),
    'ROC AUC': roc_auc_score(test_y, prebuild_NBC_y)
}

# Logistic Regression
logistic_regression = CustomLogisticRegression(C = 0, learning_rate=0.001, max_train_iterations=50)
logistic_regression.fit(train_X, train_y)
y_pred_lr = logistic_regression.predict(test_X)
lr_metrics = {
    'Accuracy': accuracy_score(test_y, y_pred_lr),
    'F1 Score': f1_score(test_y, y_pred_lr),
    'Confusion Matrix': confusion_matrix(test_y, y_pred_lr),
    'ROC AUC': roc_auc_score(test_y, y_pred_lr)
}

# Sci-Kit learn Logistic Regression
sci_log_regression = LogisticRegression(C=0.1, max_iter=50)
sci_log_regression.fit(train_X, train_y)
y_pred_sci_lr = sci_log_regression.predict(test_X)
lr_sci_metrics = {
    'Accuracy': accuracy_score(test_y, y_pred_sci_lr),
    'F1 Score': f1_score(test_y, y_pred_sci_lr),
    'Confusion Matrix': confusion_matrix(test_y, y_pred_sci_lr),
    'ROC AUC': roc_auc_score(test_y, y_pred_sci_lr)
}

# Decision Forest
forest = SimpleDecisionForest(num_trees=30, max_depth=4)
forest.fit(train_X, train_y)
y_pred_forest = forest.predict(test_X)
forest_metrics = {
    'Accuracy': accuracy_score(test_y, y_pred_forest),
    'F1 Score': f1_score(test_y, y_pred_forest),
    'Confusion Matrix': confusion_matrix(test_y, y_pred_forest),
    'ROC AUC': roc_auc_score(test_y, y_pred_forest)
}

# BarGraphs of metrics 
# Accuracy Scores
models = ['Decision Forest', 'SVM', 'Naive Bayes', 'Logistic Regression']
custom_accuracies = [
    forest_metrics['Accuracy'] * 100,  # Custom Decision Forest
    custom_svm_metrics['Accuracy'] * 100,  # Custom SVM
    custom_nbc_metrics['Accuracy'] * 100,  # Custom Naive Bayes
    lr_metrics['Accuracy'] * 100  # Custom Logistic Regression
]
prebuilt_accuracies = [
    tf_forest_metrics['Accuracy'] * 100,  # TensorFlow Decision Forest
    svm_metrics['Accuracy'] * 100,  # SciKit SVM
    prebuild_nbc_metrics['Accuracy'] * 100,  # Prebuilt Naive Bayes
    lr_sci_metrics['Accuracy'] * 100  # SciKit Logistic Regression
]

x = range(len(models))  # positions for the groups
width = 0.35  # bar width

### Step 2: Plot the Data
fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted for the additional model
rects1 = ax.bar(x, custom_accuracies, width, label='Custom Models', color='blue')
rects2 = ax.bar([p + width for p in x], prebuilt_accuracies, width, label='Prebuilt/SciKit Models', color='green')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Model Type')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Comparison by Model Type')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(models)
ax.legend()

### Step 3: Add Labels to Bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='black')

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.show()


# F1 Scores
# Define the model names and their F1 scores
models = ['Decision Forest', 'SVM', 'Naive Bayes', 'Logistic Regression']
custom_f1_scores = [
    forest_metrics['F1 Score'],  # Custom Decision Forest
    custom_svm_metrics['F1 Score'],  # Custom SVM
    custom_nbc_metrics['F1 Score'],  # Custom Naive Bayes
    lr_metrics['F1 Score'],  # Custom Logistic Regression
]
prebuilt_f1_scores = [
    tf_forest_metrics['F1 Score'],  # TensorFlow Decision Forest
    svm_metrics['F1 Score'],  # SciKit SVM
    prebuild_nbc_metrics['F1 Score'],  # Prebuilt Naive Bayes
    lr_sci_metrics['F1 Score'],  # SciKit Logistic Regression
]

x = range(len(models))  # positions for the groups
width = 0.35  # bar width

### Step 2: Plot the Data
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, custom_f1_scores, width, label='Custom Models', color='blue')
rects2 = ax.bar([p + width for p in x], prebuilt_f1_scores, width, label='Prebuilt/SciKit Models', color='green')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Model Type')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison by Model Type')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(models)
ax.legend()

### Step 3: Add Labels to Bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='black')

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.show()


# ROC AUC Score
# Define the model names and their ROC AUC scores
models = ['Decision Forest', 'SVM', 'Naive Bayes', 'Logistic Regression']
custom_roc_auc_scores = [
    forest_metrics['ROC AUC'],  # Custom Decision Forest
    custom_svm_metrics['ROC AUC'],  # Custom SVM
    custom_nbc_metrics['ROC AUC'], # Custom Naive Bayes
    lr_metrics['ROC AUC']  # Custom Logistic Regression
]
prebuilt_roc_auc_scores = [
    tf_forest_metrics['ROC AUC'],  # TensorFlow Decision Forest
    svm_metrics['ROC AUC'],  # SciKit SVM
    prebuild_nbc_metrics['ROC AUC'], # Prebuilt Naive Bayes
    lr_sci_metrics['ROC AUC']  # SciKit Logistic Regression
]

x = range(len(models))  # positions for the groups
width = 0.35  # bar width

### Step 2: Plot the Data
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, custom_roc_auc_scores, width, label='Custom Models', color='blue')
rects2 = ax.bar([p + width for p in x], prebuilt_roc_auc_scores, width, label='Prebuilt/SciKit Models', color='green')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Model Type')
ax.set_ylabel('ROC AUC Score')
ax.set_title('ROC AUC Score Comparison by Model Type')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(models)
ax.legend()

### Step 3: Add Labels to Bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='black')

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.show()

# Confusion Matrix
# Custom settings for seaborn for better aesthetics
sns.set(style="whitegrid")

def plot_confusion_matrix(cm, model_name, ax, title='Confusion Matrix'):
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'{model_name} - {title}')

# Define the model names and include both custom and prebuilt models
models = ['Custom Decision Forest', 'Prebuilt Decision Forest', 'Custom SVM', 'Prebuilt SVM', 'Custom NBC', 'Prebuilt NBC', 'Custom Logistic Regression', 'Prebuilt Logistic Regression']

# Assuming confusion matrices are stored like this
conf_matrices = [
    forest_metrics['Confusion Matrix'],  # Custom Decision Forest
    custom_svm_metrics['Confusion Matrix'],  # Custom SVM
    custom_nbc_metrics['Confusion Matrix'], # Custom Naive Bayes
    lr_metrics['Confusion Matrix'],  # Custom Logistic Regression
    tf_forest_metrics['Confusion Matrix'],  # Prebuilt Decision Forest
    svm_metrics['Confusion Matrix'],  # Prebuilt SVM
    prebuild_nbc_metrics['Confusion Matrix'], # Prebuilt Naive Bayes
    lr_sci_metrics['Confusion Matrix']  # Prebuilt Logistic Regression
]

# Create a figure to hold the subplots, adjust the size accordingly
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 12))

# Flatten the axes array to make indexing easier
axes = axes.flatten()

# Loop through the list of confusion matrices and create a subplot for each
for i, cm in enumerate(conf_matrices):
    plot_confusion_matrix(cm, models[i], axes[i])

# Adjust layout to prevent overlap
fig.tight_layout()

# Display the plot
plt.show()

