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
    
def create_simple_transformer_model(input_shape, embed_dim=128, num_heads=4, ff_dim=256, maxlen=200):
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Embedding Layer
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    x = PositionalEncoding(maxlen, embed_dim)(x)

    # Single Transformer Block
    x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(x)

    # Global Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense Layers
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output Layer
    outputs = tf.keras.layers.Dense(units=y_train.shape[1], activation="softmax")(x)  # Adjust output units based on your task

    # Compile Model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def get_parameters(self, config):
        # Return model weights
        return model.get_weights()

    def fit(self, parameters, config):
        # Set model weights
        model.set_weights(parameters)
        # Train the model
        model.fit(x_train, y_train, epochs=5, batch_size=64)
        # Return updated weights
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # Set model weights
        model.set_weights(parameters)
        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        # Return evaluation results
        return loss, len(x_test), {"accuracy": accuracy}

for i in range(num_experiments):
    print(f"\n🔥 Running Independent Experiment {i+1}/{num_experiments} with Random Seed {i}")
    set_random_seed(i)  # Set a new seed for each run

    # Define a fresh model for each independent run
    model = create_simple_transformer_model(input_shape=(x_train.shape[1], x_train.shape[2]))

    # Start FL training for 200 rounds
    fl.client.start_numpy_client(server_address="127.0.0.1:8092", client=CifarClient(model, x_train, y_train, x_test, y_test))

    # Evaluate final accuracy after 200 rounds
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    accuracy_scores.append(accuracy)

# Compute mean and standard deviation across 30 independent FL runs
mean_acc = statistics.mean(accuracy_scores)
std_acc = statistics.stdev(accuracy_scores)

print(f"\n✅ Final Results after {num_experiments} Independent Runs:")
print(f"Mean Accuracy: {mean_acc:.4f}, Standard Deviation: {std_acc:.4f}")

