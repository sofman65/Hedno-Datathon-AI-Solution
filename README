# HEDNO Datathon Project: Power Theft Detection

## Overview

This project was developed as part of the HEDNO (Hellenic Electricity Distribution Network Operator) Datathon. The primary objective was to address the critical issue of power theft in electricity distribution networks. The solution involves data preprocessing, imputation, clustering, and machine learning models to detect anomalies and potential power theft.

## Project Structure

The project is organized into several components, each responsible for different aspects of the solution:

1. **Data Preprocessing and Imputation**
2. **Clustering**
3. **Machine Learning Models**
4. **Evaluation and Metrics**

## Notebooks

### 1. `DQN.ipynb`

This notebook implements a Double Deep Q-Network (DQN) for binary classification. It includes the following steps:

- Importing necessary libraries
- Initializing the environment and agent
- Training the agent
- Evaluating the agent on new data

Relevant code snippet:


### 2. `Imputation.ipynb`

This notebook handles the imputation of missing data using the `ImbDataProcessor` class. It includes:

- Loading the dataset
- Defining the `ImbDataProcessor` class
- Processing the data
- Saving the processed data and the processor instance

Relevant code snippet:


### 3. `Sensor_Predictions.ipynb`

This notebook focuses on data encoding, splitting, and evaluating different machine learning models. It includes:

- Installing necessary libraries
- Loading and preprocessing the dataset
- Evaluating models like Random Forest, XGBoost, CatBoost, and LightGBM

Relevant code snippet:


### 4. `Main.ipynb`

This notebook integrates various components of the project, including geoclustering and imputation. It includes:

- Installing necessary libraries
- Defining functions for geoclustering and imputation
- Running the model on the dataset




## Python Scripts

### `Agent/Utils/Imputation.py`

This script defines the `ImbDataProcessor` class, which is responsible for imputing missing data in the dataset. It includes methods for calculating averages, processing data, imputing new data, and saving/loading the processor.




### `Agent/Utils/ModelRunner.py`

This script defines the `ModelRunner` class, which is responsible for running the machine learning models on the dataset. It includes methods for splitting the data and executing the provided function on the train and test sets.




### `Agent/Utils/GeoClustering.py`

This script defines the `GeoClustering` class, which is responsible for clustering geographical data using the HDBSCAN algorithm. It includes methods for converting coordinates, clustering data, predicting new points, and saving/loading the model.




### `Agent/Sensors/Stacking.py`

This script defines the `StackingAnomalyDetector` class, which uses a stacking ensemble method for anomaly detection. It includes methods for fitting the model, predicting probabilities, making predictions, and evaluating the model.




### `Agent/Sensors/lightgbm.py.amltmp`

This script defines the `LightGBMAnomalyDetector` class, which uses the LightGBM algorithm for anomaly detection. It includes methods for fitting the model, predicting probabilities, making predictions, and evaluating the model.




### `Agent/Sensors/xgboost.py.amltmp`

This script defines additional methods for the `XGBoostAnomalyDetector` class, including getting and setting parameters.




## How to Run

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the notebooks:**
    Open each notebook in Jupyter and run the cells sequentially.

4. **Run the scripts:**
    Execute the Python scripts as needed to preprocess data, run models, and evaluate results.

## Conclusion

This project provides a comprehensive solution to detect power theft in electricity distribution networks. By combining data preprocessing, imputation, clustering, and machine learning models, the solution aims to identify anomalies and potential theft effectively.
