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

df_train = df_train[df_train['subject'] == 'b']

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

# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.positional_encoding = self.get_positional_encoding(maxlen, embed_dim)

    def get_positional_encoding(self, maxlen, embed_dim):
        pos = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / embed_dim)
        angles = pos * angle_rates
        pos_enc = tf.where(tf.cast(i, tf.int32) % 2 == 0, tf.math.sin(angles), tf.math.cos(angles))
        return tf.cast(pos_enc, tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[: tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()  # Correct super() call
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config


# Transformer Block Layer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Create the Transformer Model outside of the CifarClient class
def create_transformer_model(input_shape, embed_dim=128, num_heads=4, ff_dim=256, maxlen=200):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    x = PositionalEncoding(maxlen, embed_dim)(x)
    
    # Adding transformer blocks
    x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(x)
    x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(x)

    # Global Average Pooling and Dense Layers for classification
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)  # Example output for 10 classes
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model

# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def get_parameters(self, config):
        # Return model weights
        return model.get_weights()

    def fit(self, parameters, config):
        # Set model weights
        model.set_weights(parameters)
        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=64)
        # Return updated weights
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        # Set model weights
        model.set_weights(parameters)
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        # Increment the round counter

        # Return evaluation results
        return loss, len(X_test), {"accuracy": accuracy}


for i in range(num_experiments):
    print(f"\n🔥 Running Independent Experiment {i+1}/{num_experiments} with Random Seed {i}")
    set_random_seed(i)  # Set a new seed for each run

    # Define a fresh model for each independent run
    model = create_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Start FL training for 200 rounds
    fl.client.start_numpy_client(server_address="127.0.0.1:8091", client=CifarClient(model, X_train, y_train, X_test, y_test))

    # Evaluate final accuracy after 200 rounds
    _, accuracy = model.evaluate(X_test, y_test, verbose=1)
    accuracy_scores.append(accuracy)

# Compute mean and standard deviation across 30 independent FL runs
mean_acc = statistics.mean(accuracy_scores)
std_acc = statistics.stdev(accuracy_scores)

print(f"\n✅ Final Results after {num_experiments} Independent Runs:")
print(f"Mean Accuracy: {mean_acc:.4f}, Standard Deviation: {std_acc:.4f}")

