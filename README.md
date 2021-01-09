# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

### Problem Statement
The dataset used for this project is the Bank Marketing Dataset which contains data related to marketing campaigns and phone calls of a banking institution. The aim of this project is to train models that can use this data to predict whether or not a person taking part in the campaign will subscribe to the term deposit.

### Solution
The project compares the performance of the Scikit-learn model developed using the Python SDK and model found using AutoML. It was found that the performance of both the models were more or less the same with not much difference in Accuracies. With a difference of around 0.2-0.4% of accuracy, the AutoML model was found to perform better.


## Scikit-learn Pipeline

The code for the scikit-learn pipeline is separated into 2 files, *train.py* and *udacity-project.ipynb*. The pipeline has 3 stages:
 * Data Preparation
 * Training Configuration
 * Taining Validation
Out of these the code for the first 2 stages are covered in the training script *train.py* while the Training Validation, i.e. Hyperparameter Tuning using Hyperdrive is covered in the Jupyter Notebook *udacity-project.ipynb*. The stages are explained in more detail below.
 
**Data preparation.** <br>
First step in the pipeline is the Data Preparation. Using the AzureML SDK, the data was first retrieved from an online CSV file source as a TabularDataset object using the TabularDatasetFactory class. The data is then cleaned by performing encoding on the categorical features. Label Encoding was performed on binary class features like *marital, housing, loan, etc.* whereas as One-Hot Encoding was performed on multi-class features like *job, education*. This cleaned data was then split into train and test sets.

**Training Configuration** <br>
For this task, the algorithm that was chosen to train and predict is the Logistic Regeression algorithm. We first start by defining the hyperparameters to the algorithm that will be tuned by the HyperDrive, which in our case is *max_iter (maximum iterations)* and *C (Inverse Regularisation Strength)*. These hyperparameters are designed to be taken as arguments while running the training script. The hyperparameters and the data is used to train the Logitic Regression model after which the Accuracy of the model is logged in the run, followed by saving the model using the *joblib* library.

**Training Validation** <br>
This is the stage of the pipeline that uses the HyperDrive power of AzureML to find the optimal hyperparameters for the Logistic Regression model. We start by defining the Parameter search space and the Stopping Policy required by the HyperDiveConfig after which an SKLearn estimator is created to connect with the trining script. The set number of runs with different parameter combinations are executed to find the model and hyperparameters that gives the best Accuracy. This best model is then registered so that it can be used later in deployment.


### Benefits of the chosen Parameter Sampler
The Parameter Sampler finally chosen for this project is the Random Parameter sampler. The other 2 Parameter Samplers available are the Grid Parameter sampler and Bayesian Parameter sampler, which were not chosen for the following reasons. 
 * For the chosen Logistic Regression algorithm, the hypermarameters to be fine tuned included both a discrete hyperparameter like the *max_iter (maximum iterations)* and a continuous hyperparameter like *C (Inverse Regularization Strength)*. Hence, the Grid Sampler could not be used to tune the continuous hyperparameter for the task. 
 * To chose between the Bayesian and Random Parameter sampler, the pipeline was run twice to compare the performance of the 2. It was found that there was not much of a difference in Best Accuracies using both the samplers, but the Bayesian Parameter Sampler took a slightly longer time to run than the Random Parameter Sampler. In addition, the official [Azure ML Docs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#bayesian-sampling) also mentions that Bayesian Samplers provide better results only when the number of runs is greater than at least 20 times the number of hyperparameters tuned, which would be very costly if there are more hyperparameters to be tuned.
Hence, considering the time, costs and versatility to use for different types of parameters both now and in future, the Random Parameter Sampler was chosen as the ideal sampler for this task.

### Benefits of chosen Early Stopping Policy
The Early Stopping policy chosen is the Bandit Policy which stops a particular run if its performance drops below a certain threshold than the known best performing model at the time. For this project we aimed at finding the model with the best aacuracy and hence the Bandit Policy is a useful policy to stop by considering only the best known accuracy at that point rather than the MedianStoppingPolicy which considers the median performance over all the runs so far. This policy is hence useful and saves time by stopping runs without the required performance earlier.

## AutoML
The best model generated by AutoML is the VotingEnsemble model with an accuracy of 91.84%. Using the Model Explanation feature in AzureML, it was seen that the *duration (campaign call duration to each person)* was the most important feature in predicting the outcome.
The VotingEnsemble  as the name suggests uses soft voting from multiple algorithms, i.e takes the sum of the probabilities of multiple algorithms and chooses the largest sum [1] to come up with the prediction. The algorithms used by the AutoML VotingEnsemble with some of the hyperparameters used are as follows:
 * XGBoostClassifier with SparseNormalizer (thrice with different hyperparameters each time as follows)
   * colsample_bytree = 0.9, 1, 0.9
   * max_depth = 9, 4, 6
   * n_estimators = 25, 100, 100
 * XGBoostClassifier with MaxAbsScaler
   * colsample_bytree = 1
   * max_depth = 3
   * n_estimators = 100
 * LightGBMClassifier with MaxAbsScaler
   * boosting_type = 'gbdt'
   * colsample_by_tree = 1.0
   * learning_rate = 0.1
   * min_child_weight = 0.001
 * LightGBMClassifier with SparseNormalizer
   * boosting_type = 'gbdt'
   * colsample_by_tree = 0.693
   * learning_rate = 0.579
   * min_child_weight = 6
 * RandomForestClassifier with MinMaxScaler
   * criterion = 'gini'
   * min_samples_split = 0.01
   * n_estimators = 25

## Pipeline comparison
When comparing the performance of the 2 models, it was found that both the models performed equally well on the give dataset. The best accuracy of the Logistic Regression Scikit-learn model over multiple runs and experiments was found to be 91.77%, while the best accuracy of the AutoML model was found to be 91.82%. 
This difference between the best accuracies is not too much, but as the Scikit-learn model uses RandomParmeterSampling for the HyperDrive, the parameters chosen for the model varies each time the Pipeline was executed and hence the best accuracies too varied each time and was found to be in the range of 91.2-91.6%. Whereas the AutoML model consistemtly has the same accuracy of 91.8% Hence in this respect, it can be considered that the VotingEnsemble model generated by AutoML model performs slightly better than the Scikit-learn model.

## Future work
From the confusion matrix generated by AutoML, it was found that, the Bank Marketing dataset has imbalanced classes with 21k+ records (66.7% of the data) for the class 0 ('no') while only around 2.7k+ (33.3% of the data) records for class 1 ('yes'), due to which the model performed badly in predicting the class 1, with a lot of False Negatives, and hence a low Recall rate. Hence, some measures must be taken to reduce this imbalance and train better for class 1. Some suggestions includes [2]:
 * collecting more data for class 1
 * using alternate algorithms like Clustering which are better suited for training on imbalanced classes
 * using ensemble techniques like Boosting

Another suggestion for future work includes using more hyperparameters of the LogisticRegression class to be tuned by the HyperDrive

## References
1. [How to Develop Voting Ensembles With Python](https://machinelearningmastery.com/voting-ensembles-with-python/)
2. [Imbalanced Data : How to handle Imbalanced Classification Problems](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/)
