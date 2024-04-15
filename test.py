import tensorflow_decision_forests as tfdf

import pandas

test_df = pandas.read_csv("./data/testcsv")

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="fake")

