# Cpts_440_instagram_fake_account_detection

In the current digital age where social media platforms like Instagram play a pivotal role in our
lives, the large amount of fake accounts threatens the user experience and the platform's integrity.
These fake accounts, engaging in activities ranging from spam to scamming, present a significant
challenge. Manually distinguishing between real and fake accounts is impractical due to
Instagram's extremely large user base, necessitating an automated solution to accurately identify
the presence of these fake accounts. The data we use is collected from kaggle data collection,
more specifically will be instagram fake spammer genuine accounts
datasets.(https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-account
) Some preparation for the data using python and pandas libraries was needed for all model types. Initially, 
we planned to use just a decision tree, however this proved too trivial. We pivoted and each team member 
chose a model to implement and compare against a prebuilt version. We ended up implementing 
a Naive Bayes Classifier, SVM, Decision Tree, and Logistic Regression. After implementation, we measured each 
model’s accuracy, the overall correctness of the model's predictions, and Precision which measures the
proportion of correctly identified fake accounts among all the instances predicted as fake. Then,
we also measured Sensitivity or True Positive Rate, which is the proportion of actual fake
accounts that are correctly identified by the model. And lastly, specificity, which measures the
proportion of actual genuine accounts that are correctly identified by the model.


### Prerequisites

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
