# Federated vs. Centralized Learning Benchmark for Human Activity Recognition (HAR)

## Project Overview
This repository contains a comprehensive benchmark study comparing **Federated Learning (FL)** and **Centralized Learning (CL)** methods applied to **Human Activity Recognition (HAR)** tasks. The study evaluates performance, communication efficiency, and robustness across **4 diverse datasets, 3 deep learning architectures,** and **3 federated aggregation strategies**.

The goal is to analyze the trade-offs between data privacy (FL) and model performance (CL) in the context of wearable sensor data.

<p align="center">
<img src = "https://github.com/wasay530/Federated-Learning-Human-Activity-Recognition/blob/82d67bf1ef33b3cfbb8a663171e09c591a1ffc54/Block_Diagram.png" title= "The Architecture for Benchmarking Federated Learning as a Privacy-Preserving Alternative to Centralized Learning for Human Activity Recognition" width="900" height="600" alt>
</p>
<p align="center">
  <em>Figure 1: The Architecture for Benchmarking FL as a Privacy-Preserving Alternative to CL for Human Activity Recognition</em>  
</p>

## Datasets
This benchmark utilizes four standard HAR datasets covering various sensor types, placement locations, and activities.

			
Dataset   | Full Name	 | Description | Sensors | Subjects
--------- | ----------- | ----------- | ----------- | ----------- 
MHealth |	Mobile Health |	Multi-modal body motion and vital signs recordings. |Chest, Wrist, Ankle (Acc, Gyro, Mag, ECG)	| 10
HHAR	| Heterogeneity HAR |	Focuses on device heterogeneity (smartphones/watches) and sensor biases.| Accelerometer, Gyroscope	| 9
WISDM	| Wireless Sensor Data Mining |	Large-scale dataset with diverse activities of daily living. | Smartphone/Smartwatch (Acc, Gyro)	| 36
UCI-HAR	| UCI Human Activity Recognition | Standard benchmark collected via waist-mounted smartphone. |	Accelerometer, Gyroscope	| 30

## Model Architectures
We evaluate three deep learning models tailored for time-series classification:
### 1. LSTM (Long Short-Term Memory):
* Standard RNN architecture capable of learning long-term dependencies in time-series sensor data.
* Best for: General temporal sequence modeling.

### 2. DeepConvLSTM:
* A hybrid architecture combining Convolutional Neural Networks (CNN) for automated feature extraction and LSTM for temporal modeling.
* Best for: Raw sensor data where spatial feature extraction is crucial.

### 3. Transformer:
* Attention-based mechanism that captures global dependencies across the entire time window.
* Best for: Handling complex, long-range dependencies and parallelizing training.

## Aggregation Techniques (Federated Learning)
The study implements three distinct aggregation strategies to handle client updates:
### 1. FedAvg (Federated Averaging):
* The baseline FL algorithm. Clients train locally, and the server averages the weights.
* Equation: $w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k$

### 2. FedProx (Federated Proximal):
* Adds a proximal term to the local objective function to limit the impact of non-IID data (statistical heterogeneity) and system heterogeneity.
* Key Feature: Prevents local models from drifting too far from the global model.

### 3. FedAdam (Federated Adam):
* Applies adaptive optimization (Adam) on the server side using the pseudo-gradients computed from client updates.
* Key Feature: Improves convergence stability in complex optimization landscapes.

## References
1. DeepConvLSTM: Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and lstm recurrent neural networks for multimodal wearable activity recognition. Sensors.
2. FedAvg: McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
3. FedProx: Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks. MLSys.
4. FedAdam: Reddi, S., et al. (2020). Adaptive Federated Optimization. ICLR.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
