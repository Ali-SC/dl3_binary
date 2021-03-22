#%%-*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:11:54 2021

@author: ali.kadhim
"""

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


#%%  Generate Training Data
train_labels = []
train_samples = []



for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

# for i in train_samples:
#     print(i)
#
# for i in train_labels:
#     print(i)


#%% shuffle order and normalize data

# put labels and samples into an array so you can shuffle/randomize the order of data for training
# shuffling/randomizing is important this early due to the shuffling when training being applied after
# pulling the validation dataset (0.1)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

# make all sample ages between 13 and 100 to between 0 and 1; scaling for algorithm
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# for i in scaled_train_samples:
#     print(i)

#%% Simple tf.keras Sequential Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy



#%% Set up model

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
    ])

model.summary()


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

import datetime
import os
# log_dir = os.getcwd() + "\\logs\\fit\\"
# # log_dir = "logs\fit"
# os.makedirs(log_dir)

# tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1)

# logdir=mylogs:C:\path\to\output\folder
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    x=scaled_train_samples,
    y=train_labels,
    validation_split=0.1,
    batch_size=10,
    epochs=30,
    shuffle=True,
    verbose=2,
    callbacks=[tensorboard_callback])

assert model.history.history.get('accuracy')[-1] > 0.90
assert model.history.history.get('val_accuracy')[-1] > 0.90



#%% Generate Test Data

test_labels = []
test_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

#%% Predict

predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

rounded_predictions = np.argmax(predictions, axis=-1)

#%% Confusion Matrix & Plotting

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")

    print(cm)

    thresh = cm.max()/2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title = 'Confusion Matrix')


#%% Save and Load a Model

import os.path
if os.path.isfile('models/ali_trial_model.h5') is False:
    model.save('models/ali_trial_model.h5')


from tensorflow.keras.models import load_model
new_model = load_model('models/ali_trial_model.h5')


#%% Model to JSON

json_string = model.to_json()

# with open('json_model.json', 'w') as outfile:
#     json.dump(json_string[0], outfile)


#%% JSON to Model
# model reconstruction from JSON:
from tensorflow.keras.models import model_from_json

# with open('json_model.json','r') as json_file:
#     json_string_new = json.load(json_file)

json_model = model_from_json(json_string)
json_model.summary()


#%% Model.save_weights()

import os.path
if os.path.isfile('models/ali_model_weights.h5') is False:
    model.save_weights('models/ali_model_weights.h5')

model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])


model2.load_weights('models/ali_model_weights.h5')
model2.get_weights()

model2.get_config()
