import numpy as np 
import pandas as pd 	
import matplotlib.pyplot as plt 

class NBC:
    def __init__(self):
        train_df = pd.read_csv("./data/train.csv")
        test_df = pd.read_csv("./data/test.csv")

        self.defaultBinsNum = 4
        self.length_username_bins = []
        self.fullname_words_bins = []
        self.length_fullname_bins = []
        self.description_length_bins = []
        self.num_posts_bins = []
        self.num_followers_bins = []
        self.num_follows_bins = []

        self.train_X = train_df.drop("fake", axis=1)
        self.train_y = train_df['fake']

        self.test_X = test_df.drop("fake", axis=1)
        self.test_y = test_df['fake']

        self.priorProbs = {}
        self.conditionalProbs = {}
        self.fake_prob = 0
        self.real_prob = 0

        self.recatagorize_data_into_sections()
    
    def recatagorize_data_into_sections(self):
        self.set_length_username_bins()
        self.set_fullname_words_bins()
        self.set_length_fullname_bins()
        self.set_description_length_bins()
        self.set_num_posts_bins()
        self.set_num_followers_bins()
        self.set_num_follows_bins()
        self.calculate_prior_probability()
        self.calculate_conditional_probability()
    
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
        
    def calculate_frequencies_of_bins(self):
        frequency = {}
        counter = 0
        for x in self.train_X.keys():
            counter = 0
            frequency[x] = {"fake":{}, "real":{}}
            for y in self.train_X[x]:
                if self.train_y[counter] == 1:
                    if y in frequency[x]['fake']:
                        frequency[x]['fake'][y] += 1
                    else:
                        frequency[x]['fake'][y] = 1
                else:
                    if y in frequency[x]['real']:
                        frequency[x]['real'][y] += 1
                    else:
                        frequency[x]['real'][y] = 1
                counter += 1
        return frequency

    def combine_dicts(self, dict1, dict2):
        for x in dict1:
            if x in dict2:
                dict2[x] += dict1[x]
            else:
                dict2[x] = dict1[x]
        return dict2
    
    def calculate_conditional_probability(self):
        frequencies = self.calculate_frequencies_of_bins()
        num_fake = list(self.train_y).count(0)
        num_real = list(self.train_y).count(1)
        self.fake_prob = num_fake / (num_fake + num_real)
        self.real_prob = num_real / (num_fake + num_real)
        for key in frequencies:
            self.conditionalProbs[key] = {"fake":{}, "real":{}}
            for fake_key in frequencies[key]['fake']:
                self.conditionalProbs[key]['fake'][fake_key] = frequencies[key]['fake'][fake_key] / num_fake
            for real_key in frequencies[key]['real']:
                self.conditionalProbs[key]['real'][real_key] = frequencies[key]['real'][real_key] / num_real
    
    def calculate_prior_probability(self):
        frequencies = self.calculate_frequencies_of_bins()
        for key in frequencies:
            self.priorProbs[key] = {}
            sum_dict = self.combine_dicts(frequencies[key]['fake'], frequencies[key]['real'])
            for inner_key in sum_dict:
                self.priorProbs[key][inner_key] = sum_dict[inner_key] / len(self.train_y)

    def predict(self):
        self.test_X = self.test_X.transpose()
        output = []
        real_denominator = 1
        fake_denominator = 1
        for key in self.test_X:
            real_numerator = self.real_prob
            fake_numerator = self.fake_prob
            for inner_key in self.conditionalProbs.keys():
                if self.test_X.loc[:,key][inner_key] in self.conditionalProbs[inner_key]['real'] and self.test_X.loc[:,key][inner_key] in self.priorProbs[inner_key]:
                    real_numerator *= self.conditionalProbs[inner_key]['real'][self.test_X.loc[:,key][inner_key]]
                    real_denominator *= self.priorProbs[inner_key][self.test_X.loc[:,key][inner_key]]
                if self.test_X.loc[:,key][inner_key] in self.conditionalProbs[inner_key]['fake'] and self.test_X.loc[:,key][inner_key] in self.priorProbs[inner_key]:
                    fake_numerator *= self.conditionalProbs[inner_key]['fake'][self.test_X.loc[:,key][inner_key]]
                    fake_denominator *= self.priorProbs[inner_key][self.test_X.loc[:,key][inner_key]]
            if (real_numerator) < (fake_numerator):
                output.append(1)
            else:
                output.append(0)
        total = len(output)
        correct = 0
        for x in range(0, len(self.test_y)):
            if output[x] == self.test_y[x]:
                correct += 1
        percentage = 100*(correct / total)
        print(f"Custom Naive Bayes accuracy: {round(percentage, 4)} %")

