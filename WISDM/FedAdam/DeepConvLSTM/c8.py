import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection, preprocessing, metrics
import tensorflow as tf
import flwr as fl
import statistics
import random
from scipy import stats
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Number of independent experiments
num_experiments = 30
accuracy_scores = []

n_time_steps = 25
n_features = 3
step = 10

# Function to set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

columns = ['user','activity', 'x-axis', 'y-axis', 'z-axis']
train_path = r"/home/users/sardara/Revised_Approach/Datasets/WISDM/train_all.csv"
test_path = r"/home/users/sardara/Revised_Approach/Datasets/WISDM/test_all.csv"
df_train = pd.read_csv(train_path, on_bad_lines='skip', header=None, names=columns, comment=';', skiprows=1)
df_test = pd.read_csv(test_path, on_bad_lines='skip', header=None, names=columns, comment=';', skiprows=1)
df_train = df_train.dropna()
df_test = df_test.dropna()

df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

print(df_train.shape, df_test.shape)
segment = []
label = []

for i in range(0,  df_test.shape[0]- n_time_steps, step):

    xt = df_test['x-axis'].values[i: i + 25]

    yt = df_test['y-axis'].values[i: i + 25]

    zt = df_test['z-axis'].values[i: i + 25]

    labet = stats.mode(df_test['activity'][i: i + 25])[0]

    segment.append([xt, yt, zt])

    label.append(labet)

#reshape the segments which is (list of arrays) to a list
reshaped_segment = np.asarray(segment, dtype= np.float32).reshape(-1, n_time_steps, n_features)

label = np.asarray(pd.get_dummies(label), dtype = np.float32)

x_test = reshaped_segment
y_test = label

df_train = df_train[df_train['user'] == 19]

segments = []
labels = []

for i in range(0,  df_train.shape[0]- n_time_steps, step):

    xs = df_train['x-axis'].values[i: i + 25]

    ys = df_train['y-axis'].values[i: i + 25]

    zs = df_train['z-axis'].values[i: i + 25]

    label = stats.mode(df_train['activity'][i: i + 25])[0]

    segments.append([xs, ys, zs])

    labels.append(label)

#reshape the segments which is (list of arrays) to a list
reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

x_train = reshaped_segments

y_train = labels

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        return loss, len(x_test), {"accuracy": accuracy}

for i in range(num_experiments):
    print(f"\n🔥 Running Independent Experiment {i+1}/{num_experiments} with Random Seed {i}")
    set_random_seed(i)  # Set a new seed for each run

    # Define a fresh model for each independent run
    model = Sequential([
    # Convolutional layer
    Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    
    # LSTM layer
    LSTM(units=128),
    Dropout(0.5),

    # Dense layer with ReLU
    Dense(units=64, activation='relu'),

    # Softmax output layer
    Dense(units=y_train.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Start FL training for 200 rounds
    fl.client.start_numpy_client(server_address="127.0.0.1:8092", client=FLClient())

    # Evaluate final accuracy after 200 rounds
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    accuracy_scores.append(accuracy)

# Compute mean and standard deviation across 30 independent FL runs
mean_acc = statistics.mean(accuracy_scores)
std_acc = statistics.stdev(accuracy_scores)

print(f"\n✅ Final Results after {num_experiments} Independent Runs:")
print(f"Mean Accuracy: {mean_acc:.4f}, Standard Deviation: {std_acc:.4f}")

