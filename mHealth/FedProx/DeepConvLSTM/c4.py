import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection, preprocessing, metrics
import tensorflow as tf
import flwr as fl
import statistics
import random
from sklearn.utils import resample
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Number of independent experiments
num_experiments = 30
accuracy_scores = []

# Function to set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Loading the dataset
path = r"/home/users/sardara/Revised_Approach/Datasets/Mhealth/mhealth_resampled_data.csv"
df = pd.read_csv(path)

# Remove duplicates
df = df.drop(df[df.duplicated(keep='first')].index, axis=0)
print(df.shape)

# Splitting majority and minority classes
df_majority = df[df.Activity == 0]
df_minorities = df[df.Activity != 0]

# Downsampling majority class
df_majority_downsampled = resample(df_majority, n_samples=30000, random_state=42)
df = pd.concat([df_majority_downsampled, df_minorities])
print(df.Activity.value_counts())

# Dropping features with data outside the 98% confidence interval
df1 = df.copy()

class_map = {
    0: 'Nothing',
    1: 'Standing still',  
    2: 'Sitting and relaxing', 
    3: 'Lying down',  
    4: 'Walking',  
    5: 'Climbing stairs',  
    6: 'Waist bends forward',
    7: 'Frontal elevation of arms', 
    8: 'Knees bending (crouching)', 
    9: 'Cycling', 
    10: 'Jogging', 
    11: 'Running', 
    12: 'Jump front & back' 
}

for feature in df1.columns[:-2]:  # Assuming 'subject' and 'Activity' are the last two columns
    lower_range = np.quantile(df[feature], 0.01)
    upper_range = np.quantile(df[feature], 0.99)
    print(feature, 'range:', lower_range, 'to', upper_range)

    # Dropping values outside the range
    df1 = df1.drop(df1[(df1[feature] > upper_range) | (df1[feature] < lower_range)].index, axis=0)
    print('shape', df1.shape)

# Splitting data into train and test set
train = df1[(df1['subject'] != 'subject10') & (df1['subject'] != 'subject9') & (df1['subject'] != 'subject8')]
test = df1.drop(train.index, axis=0)
print(train.shape, test.shape)

train = train[train['subject'] == "subject4"]

# Preparing features and labels for train/test
X_train_ = train.drop(['Activity', 'subject'], axis=1)
y_train_ = train['Activity']
X_test_ = test.drop(['Activity', 'subject'], axis=1)
y_test_ = test['Activity']

# Define time_steps
time_steps = 25

# Function to create sequences
def create_sequences(X, y, time_steps=5):
    X_, y_ = [], []
    n = X.shape[0]
    
    # Ensure that the sequence creation does not go out of bounds
    for i in range(n - time_steps):
        X_.append(X.iloc[i:(i + time_steps)].values)  # Correctly slicing rows and columns
        y_.append(y.iloc[i + time_steps])  # Append the corresponding label for that sequence
        
    return np.array(X_), np.array(y_)

# Creating sequences for train and test
X_train, y_train = create_sequences(X_train_, y_train_, time_steps)
X_test, y_test = create_sequences(X_test_, y_test_, time_steps)

# Print shapes of the resulting data
print(X_train.shape, X_test.shape)

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
        return loss, len(X_test), {"accuracy": accuracy}

for i in range(num_experiments):
    print(f"\n🔥 Running Independent Experiment {i+1}/{num_experiments} with Random Seed {i}")
    set_random_seed(i)  # Set a new seed for each run

    # Define a fresh model for each independent run
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
        tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(units=256, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(units=128, return_sequences=False),  # Only last LSTM should have return_sequences=False
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=len(class_map), activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )


    # Start FL training for 200 rounds
    fl.client.start_numpy_client(server_address="127.0.0.1:8091", client=FLClient())

    # Evaluate final accuracy after 200 rounds
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracy_scores.append(accuracy)

# Compute mean and standard deviation across 30 independent FL runs
mean_acc = statistics.mean(accuracy_scores)
std_acc = statistics.stdev(accuracy_scores)

print(f"\n✅ Final Results after {num_experiments} Independent Runs:")
print(f"Mean Accuracy: {mean_acc:.4f}, Standard Deviation: {std_acc:.4f}")

