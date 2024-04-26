# Cpts_440_instagram_fake_account_detection

In the current digital age where social media platforms like Instagram play a pivotal role in our
lives, the large amount of fake accounts threatens the user experience and the platform's integrity.
These fake accounts, engaging in activities ranging from spam to scamming, present a significant
challenge. Manually distinguishing between real and fake accounts is impractical due to
Instagram's extremely large user base, necessitating an automated solution to accurately identify
the presence of these fake accounts. The data we use is collected from kaggle data collection,
more specifically will be instagram fake spammer genuine accounts
datasets.(https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-account
) Some preparation for the data using python and pandas libraries might be needed. We will be
using Decision Trees as our algorithm due to it being simple to implement, the way it can handle
both numerical and categorical data. This algorithm is also quite intuitive, therefore it will be
easier to understand the reasoning behind a decision. Then, we will measure the model’s
accuracy, the overall correctness of the model's predictions, and Precision which measures the
proportion of correctly identified fake accounts among all the instances predicted as fake. Then,
we also measure Sensitivity or True Positive Rate, which is the proportion of actual fake
accounts that are correctly identified by the model. And lastly, specificity, which measures the
proportion of actual genuine accounts that are correctly identified by the model.
As a back-up, we have the Logistic Regression algorithm in mind, if the original plan using the
Decision Trees algorithm fails or proves too challenging.


### Prerequisites

This project need to install package inside requirement.txt first:
  ```sh
  pip install -r requirements.txt
  ```

