import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

def load_directory_data(directory):
    data = {}
    data['senetence'] = []
    data['sentiment'] = []
    for file_path in os.listdir(directory):
        with tf.io.gfile.GFile(os.path.join(directory,file_path),'r') as f:
            data['senetence'].append(f.read())
            data['sentiment'].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory,'pos'))
    neg_df = load_directory_data(os.path.join(directory, 'neg'))
    pos_df['polarity'] = 1
    neg_df['polarity'] = 0
    return pd.concat([pos_df,neg_df]).sample(frac=1).reset_index(drop=True)

def download_and_load_dataset(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname ="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)
    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                       "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))
    return train_df, test_df

# tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
train_df, test_df = download_and_load_dataset()
print(train_df.head())
# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.compat.v1.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.003))

# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset
# contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=1000)

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))

def get_predictions(estimator, input_fn):
  return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

LABELS = [
    "negative", "positive"
]

# Create a confusion matrix on training data.
with tf.Graph().as_default():
  cm = tf.confusion_matrix(train_df["polarity"],
                           get_predictions(estimator, predict_train_input_fn))
  with tf.Session() as session:
    cm_out = session.run(cm)

# Normalize the confusion matrix so that each row sums to 1.
cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted");
plt.ylabel("True")

plt.show()