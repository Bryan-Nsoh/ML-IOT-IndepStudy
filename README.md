## LSTM for Predicting Volumetric Water Content (VWC)

### Introduction

This project focuses on developing and training a Long Short-Term Memory (LSTM) model to predict volumetric water content (VWC) in soil. VWC is a crucial indicator of soil moisture and plant stress, and accurate predictions can be used to optimize irrigation scheduling and improve water management. 

This project is part of a larger effort to build an automated irrigation system using LoRaWAN IoT technology. 

### System Components

The main components of this project include:

* **Data Preprocessing and Feature Engineering:** Functions for cleaning, transforming, and scaling the data to prepare it for the LSTM model. This includes techniques like mean subtraction, derivative calculation, and log transformation.
* **Model Development and Training:** Functions for building, training, and evaluating the LSTM model. This includes creating sequences of data, configuring the model architecture, and implementing training and validation steps.
* **Performance Evaluation and Challenges:** Time series cross-validation is used to assess the model's generalizability and identify challenges such as overfitting.
* **Inference and Plotting:** Functions for performing inference on new data using a sliding window approach and visualizing the predictions alongside actual values.

**Note:** This project is currently in progress and includes explorations with other models like XGBoost and ARIMA. However, the focus is on refining the LSTM model due to its promising performance. 

### Benefits and Advantages

Using an LSTM model to predict VWC offers several advantages over traditional methods:

* **Improved accuracy:** LSTMs can capture complex temporal dependencies in the data, leading to more accurate predictions.
* **Automated irrigation scheduling:** Accurate VWC predictions can be used to trigger irrigation only when necessary, saving water and reducing costs.
* **Reduced plant stress:** By ensuring optimal soil moisture levels, plant stress can be minimized, leading to healthier crops and higher yields.

### Future Directions

This project has several potential areas for future improvement and expansion:

* **Expanding window cross-validation:** Exploring different window sizes and stride lengths to improve model performance.
* **Introducing categorical features:** Incorporating additional features like soil type, crop type, and weather data to enhance the model's predictive power.
* **Integration with LoRaWAN IoT platform:** Connecting the model to real-time sensor data and irrigation controllers for automated irrigation scheduling.

### Repository Contents

This repository contains the following:

* Jupyter notebook with data processing, model building, training, and inference functions.
* Documentation on project usage and contribution guidelines.

Please refer to the README file and documentation within the notebook for specific requirements and instructions. 


This project is actively being developed and forks, comments and questions are welcome!
