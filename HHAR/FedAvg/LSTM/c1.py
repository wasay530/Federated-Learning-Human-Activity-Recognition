import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection, preprocessing, metrics
import tensorflow as tf
import flwr as fl
import statistics
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Number of independent experiments
num_experiments = 30
round = 0
experiment = 0
round_1 = 0
accuracy_scores = []

# Function to set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

columns = ['x-axis', 'y-axis', 'z-axis' , 'subject', 'Activity' ]
train_path = r"/home/users/sardara/Revised_Approach/Datasets/HHAR/train_all.csv"
test_path = r"/home/users/sardara/Revised_Approach/Datasets/HHAR/test_all.csv"
df_train = pd.read_csv(train_path, on_bad_lines='skip', header=None, names=columns, comment=';')
df_test = pd.read_csv(test_path, on_bad_lines='skip', header=None, names=columns, comment=';')
df_train = df_train[1:]
print(df_train.shape, df_test.shape)

df_train.head()

df_train.info()

# Count of NA values in each column

print(df_train.isna().sum())
print("\n\nTotal NA values:", df_train.isna().sum().sum())

print(df_train["Activity"].value_counts(), "\n\n")
print(df_test["Activity"].value_counts())

activites = df_train["Activity"].unique()

class_map = {i : val for i, val in enumerate(activites)}
class_map_reverse = {key : val for val, key in class_map.items()}

print(class_map)

df_train = df_train[df_train['subject'] == 'a']

df_train["Activity"] = df_train["Activity"].apply(lambda x : class_map_reverse[x])
df_test["Activity"] = df_test["Activity"].apply(lambda x : class_map_reverse[x])

sensor_columns = ['x-axis', 'y-axis', 'z-axis']
df_train[sensor_columns] = df_train[sensor_columns].apply(pd.to_numeric, errors='coerce')
df_test[sensor_columns] = df_test[sensor_columns].apply(pd.to_numeric, errors='coerce')

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.drop("subject", axis = 1, inplace = True)
df_test.drop("subject", axis = 1, inplace = True)

time_steps = 30

def create_sequences(X, y, time_steps = 5):
    X_, y_ = [], []
    n = X.shape[0]
    for i in np.arange(n - time_steps):
        X_.append(X[i:(i + time_steps)])
        y_.append(y[i + time_steps])
    return np.array(X_), np.array(y_)

X_train_ = df_train.drop("Activity", axis = 1).values
X_test_ = df_test.drop("Activity", axis = 1).values
y_train_ = df_train["Activity"].values
y_test_ = df_test["Activity"].values

X_train, y_train = create_sequences(X_train_, y_train_, time_steps)
X_test, y_test = create_sequences(X_test_, y_test_, time_steps)

print(X_train.shape, X_test.shape)

y_train.shape
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        global round_1
        round_1 = round_1 + 1
        model.save(f"/home/users/sardara/Revised_Approach/Results/FL/HHAR/FedAvg/LSTM/experiment__{experiment}/round_{round_1}.h5")
        return loss, len(X_test), {"accuracy": accuracy}

def NumberExperiment():
    global experiment
    experiment = experiment + 1  # Increment experiment counter
    global round
    round += 1  # Increment round counter

for i in range(num_experiments):
    print(f"\n🔥 Running Independent Experiment {i+1}/{num_experiments} with Random Seed {i}")
    set_random_seed(i)  # Set a new seed for each run
    NumberExperiment()

    # Define a fresh model for each independent run
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(units=256, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(units=128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(units=64),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=len(class_map), activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=["accuracy"])

    # Start FL training
    fl.client.start_numpy_client(server_address="127.0.0.1:8091", client=FLClient())

    # Save model after each experiment
    model.save(f"/home/users/sardara/Revised_Approach/Results/FL/HHAR/FedAvg/LSTM/num_experiment_{round}.h5")

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracy_scores.append(accuracy)


# Compute mean and standard deviation across 30 independent FL runs
mean_acc = statistics.mean(accuracy_scores)
std_acc = statistics.stdev(accuracy_scores)

loss, accuracy = model.evaluate(X_test, y_test, batch_size = 64, verbose = 1)
print("LSTM")
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)

print(f"\n✅ Final Results after {num_experiments} Independent Runs:")
print(f"Mean Accuracy: {mean_acc:.4f}, Standard Deviation: {std_acc:.4f}")



