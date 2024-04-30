# Load data
import numpy as np 
import pandas as pd 	
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB

class NBC:
    def __init__(self):
        train_df = pd.read_csv("./data/train.csv")
        test_df = pd.read_csv("./data/test.csv")
        self.train_X = train_df.drop("fake", axis=1)
        self.train_y = train_df['fake']

        self.test_X = test_df.drop("fake", axis=1)
        self.test_y = test_df['fake']

        self.defaultBinsNum = 4
        self.length_username_bins = []
        self.fullname_words_bins = []
        self.length_fullname_bins = []
        self.description_length_bins = []
        self.num_posts_bins = []
        self.num_followers_bins = []
        self.num_follows_bins = []

        self.priorProbs = {}
        self.conditionalProbs = {}
        self.fake_prob = 0
        self.real_prob = 0

        # self.recatagorize_data_into_sections()
    
    def recatagorize_data_into_sections(self):
        self.set_length_username_bins()
        self.set_fullname_words_bins()
        self.set_length_fullname_bins()
        self.set_description_length_bins()
        self.set_num_posts_bins()
        self.set_num_followers_bins()
        self.set_num_follows_bins()
    
    def set_length_username_bins(self):
        data = pd.cut(self.train_X['nums/length username'], bins=self.defaultBinsNum, retbins=True)
        self.length_username_bins = data[1]
        self.train_X['nums/length username'] = np.digitize(self.train_X['nums/length username'], data[1])
        data = pd.cut(self.test_X['nums/length username'], bins=self.defaultBinsNum, retbins=True)
        self.test_X['nums/length username'] = np.digitize(self.test_X['nums/length username'], data[1])
    
    def set_fullname_words_bins(self):
        data = pd.cut(self.train_X['fullname words'], bins=self.defaultBinsNum, retbins=True)
        self.fullname_words_bins = data[1]
        self.train_X['fullname words'] = np.digitize(self.train_X['fullname words'], data[1])
        data = pd.cut(self.test_X['fullname words'], bins=self.defaultBinsNum, retbins=True)
        self.test_X['fullname words'] = np.digitize(self.test_X['fullname words'], data[1])

    def set_length_fullname_bins(self):
        data = pd.cut(self.train_X['nums/length fullname'], bins=self.defaultBinsNum, retbins=True)
        self.length_fullname_bins = data[1]
        self.train_X['nums/length fullname'] = np.digitize(self.train_X['nums/length fullname'], data[1])
        data = pd.cut(self.test_X['nums/length fullname'], bins=self.defaultBinsNum, retbins=True)
        self.test_X['nums/length fullname'] = np.digitize(self.test_X['nums/length fullname'], data[1])

    def set_description_length_bins(self):
        data = pd.cut(self.train_X['description length'], bins=self.defaultBinsNum, retbins=True)
        self.description_length_bins = data[1]
        self.train_X['description length'] = np.digitize(self.train_X['description length'], data[1])
        data = pd.cut(self.test_X['description length'], bins=self.defaultBinsNum, retbins=True)
        self.test_X['description length'] = np.digitize(self.test_X['description length'], data[1])

    def set_num_posts_bins(self):
        data = pd.cut(self.train_X['#posts'], bins=self.defaultBinsNum, retbins=True)
        self.num_posts_bins = data[1]
        self.train_X['#posts'] = np.digitize(self.train_X['#posts'], data[1])
        data = pd.cut(self.test_X['#posts'], bins=self.defaultBinsNum, retbins=True)
        self.test_X['#posts'] = np.digitize(self.test_X['#posts'], data[1])

    def set_num_followers_bins(self):
        data = pd.cut(self.train_X['#followers'], bins=self.defaultBinsNum, retbins=True)
        self.num_followers_bins = data[1]
        self.train_X['#followers'] = np.digitize(self.train_X['#followers'], data[1])
        data = pd.cut(self.test_X['#followers'], bins=self.defaultBinsNum, retbins=True)
        self.test_X['#followers'] = np.digitize(self.test_X['#followers'], data[1])
        

    def set_num_follows_bins(self):
        data = pd.cut(self.train_X['#follows'], bins=self.defaultBinsNum, retbins=True)
        self.num_follows_bins = data[1]
        self.train_X['#follows'] = np.digitize(self.train_X['#follows'], data[1])
        data = pd.cut(self.test_X['#follows'], bins=self.defaultBinsNum, retbins=True)
        self.test_X['#follows'] = np.digitize(self.test_X['#follows'], data[1])
    
    def predict(self):
        # Due to the data type, you must keep the random_state the same, as some categories are undefined for some data points, causing a failure for some states.
        gnb = CategoricalNB()
        y_pred = gnb.fit(self.train_X, self.train_y).predict(self.test_X)
        return y_pred
