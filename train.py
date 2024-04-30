import tensorflow_decision_forests as tfdf

import pandas

train_df = pandas.read_csv("./data/train.csv")

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="fake")

model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

test_df = pandas.read_csv("./data/test.csv")

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="fake")

# model.compile(metrics=['accuracy'])
# print(model.evaluate(test_ds))

# test_df = pandas.read_csv("./data/testcsv")

# test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="fake")

# html = tfdf.model_plotter.plot_model(model, tree_idx=0)
# print(html)
# with open("output.html", "w") as f:
#     f.write(html)