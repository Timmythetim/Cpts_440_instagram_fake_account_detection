# Cpts_440_instagram_fake_account_detection

In the current digital age where social media platforms like Instagram play a pivotal role in our
lives, the large amount of fake accounts threatens the user experience and the platform's integrity.
These fake accounts, engaging in activities ranging from spam to scamming, present a significant
challenge. Manually distinguishing between real and fake accounts is impractical due to
Instagram's extremely large user base, necessitating an automated solution to accurately identify
the presence of these fake accounts. The data we use is collected from kaggle data collection,
more specifically will be instagram fake spammer genuine accounts
datasets [[1]](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-account).
Some preparation for the data using python and pandas libraries was needed for all model types. Initially, 
we planned to use just a decision tree, however this proved too trivial. We pivoted and each team member 
chose a model to implement and compare against a prebuilt version. We ended up implementing 
a Naive Bayes Classifier, SVM, Decision Tree, and Logistic Regression. After implementation, we measured each 
model’s accuracy, the overall correctness of the model's predictions, and Precision which measures the
proportion of correctly identified fake accounts among all the instances predicted as fake. Then,
we also measured Sensitivity or True Positive Rate, which is the proportion of actual fake
accounts that are correctly identified by the model. And lastly, specificity, which measures the
proportion of actual genuine accounts that are correctly identified by the model.

# Demo Links
[Colab Demo](https://youtu.be/ONreByLg5G8) \
[Presentation Demo](https://youtu.be/ONreByLg5G8)

## Prerequisites

This project need to install package inside requirement.txt first (highly recommended to make a virtual environment first):
  ```sh
  pip install -r requirements.txt
  ```
Then, run the main.py
  ```sh
  python main.py
  ```
Or with python3:
  ```sh
  python3 main.py
  ```

## Description Of Files

The file names tend to describe them accurately, but here we will provide a brief explaination of each.

### Bayes_From_Scratch.py
This file is a Naive Bayes Classifier from scratch by Tim, using our training data. Some concessions had to be made during the training,
as the distribution of the data made it difficult to work with. First we modified the data, instead of raw values, we transformed
the data into categorical data to calculate the probabilities. Due to these classifications, some data points did not have probabilities
for every "bin". We skipped these values when calculating if any were missing, only including values where all probabilities were available.
This led to a small hit in accuracy, but with a larger data set it would have been even less of a hit.

### Bayes_Scikit
This file is a Naive Bayes Classifier as implemented by the Scikit-learn library. Coded by Tim, There are some utilities to set up the data for
the model.

### DecisionForest_Scratch.py
This file is a Decision Forest from Scratch. Coded by Nazar, he implemented the decision forest from scratch with comparable accuracy
to the prebuilt model.

### Decision_Forest_Tensorflow.py
This file is a Decision Forest from Tensorflow. Coded by Tim, with some modifications by Nazar, it is a simple implementation.

### Logistic_Regression_Scratch.py
This file is the Logistic Regression model from scratch. Coded by William, he implemented this from scratch with some utilities and 
achieved comparable accuracy to the prebuilt model.

### SVM / SVM_test.py
These files are the implementation of a Support Vector Machine by hand. Coded by Kwan Tou, this model achieved comparable accuracy
to the prebuilt SVM.

### test.py
This file is not very significant, and could be removed. It was included for initial testing with the Decision Forest.

Some of the implementations of the prebuilt models were so simple that they were only included in the demo notebook of our project, 
found here. https://colab.research.google.com/drive/1UVLXh2PSKmKS9cCmW4YkWjfEbtxWLhBj#scrollTo=kZI7K8uv7xTR
Please note the top section about importing data, as it will not work without the data imported.
