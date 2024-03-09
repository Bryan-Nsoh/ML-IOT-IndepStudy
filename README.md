## LSTM for Predicting Volumetric Water Content (VWC)

### Introduction

This project focuses on developing and training a Long Short-Term Memory (LSTM) model to predict volumetric water content (VWC) in soil. VWC is a crucial indicator of soil moisture and plant stress, and accurate predictions can be used to optimize irrigation scheduling and improve water management. 

This project is part of a larger effort to build an automated irrigation system using LoRaWAN IoT technology. 

### System Components

The main components of this project include:

* **Data Preprocessing and Feature Engineering:** Functions for cleaning, transforming, and scaling the data to prepare it for the LSTM model. This includes techniques like:
    * **Mean subtraction:** Removing the mean from target columns to center the data.
    * **Derivative calculation:** Calculating difference-based derivatives to highlight changes in VWC and emphasize irrigation/precipitation events.
    * **Log transformation:** Addressing data distribution issues and improving model performance.
    * **Savitzky-Golay filter:** Smoothing data in target columns to reduce noise.
    * **Cyclical encoding of timestamps:** Encoding timestamps as sine and cosine values to capture daily, hourly, and day-of-week patterns.
* **Model Development and Training:** Functions for building, training, and evaluating the LSTM model. This includes:
    * **Sequence creation:** Using a sliding window approach to create sequences of input and target data for the LSTM model.
    * **Model architecture:** Building an LSTM model with multiple layers, BatchNormalization, and Dropout for regularization.
    * **Loss function:** Using Mean Squared Error (MSE) as the loss function to penalize prediction errors.
    * **Validation strategy:** Implementing time series cross-validation with `TimeSeriesSplit` to ensure chronological validation and prevent data leakage.
    * **Early stopping:** Stopping the training process if validation loss doesn't improve for a specified number of epochs.
* **Inference and Plotting:** Functions for performing inference on new data using a sliding window approach and visualizing the predictions alongside actual values. This includes:
    * **Reverse transformation:** Transforming predictions back to the original scale and format for interpretation.
    * **Visualization:** Plotting predictions and actuals for visual comparison and evaluation.

**Note:** This project is currently in progress and includes explorations with other models like XGBoost and ARIMA. However, the focus is on refining the LSTM model due to its promising performance. 

### Technical Details

* **Expanding window cross-validation:** This project implements expanding window cross-validation, where the training window size increases with each fold, allowing the model to learn from longer historical sequences.
* **Hyperparameter tuning:** Various hyperparameters, such as the number of LSTM layers, units per layer, batch size, and learning rate, are tuned to optimize model performance.
* **Regularization techniques:** BatchNormalization and Dropout are used to prevent overfitting and improve the model's generalizability.

### Benefits and Advantages

Using an LSTM model to predict VWC offers several advantages over traditional methods:

* **Improved accuracy:** LSTMs can capture complex temporal dependencies in the data, leading to more accurate predictions.
* **Automated irrigation scheduling:** Accurate VWC predictions can be used to trigger irrigation only when necessary, saving water and reducing costs.
* **Reduced plant stress:** By ensuring optimal soil moisture levels, plant stress can be minimized, leading to healthier crops and higher yields.

### Future Directions

This project has several potential areas for future improvement and expansion:

* **Introducing categorical features:** Incorporating additional features like soil type, crop type, and weather data to enhance the model's predictive power.
* **Integration with LoRaWAN IoT platform:** Connecting the model to real-time sensor data and irrigation controllers for automated irrigation scheduling.

### Repository Contents

This repository contains the following:

* Jupyter notebook with data processing, model building, training, and inference functions.
* Documentation on project usage and contribution guidelines.

Please refer to the README file and documentation within the notebook for specific requirements and instructions. 


This project is actively being developed and forks, comments and questions are welcome!
