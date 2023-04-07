# horse_colic_data_with-_ML

The raw data were obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Horse+Colic)

Aim: 
- To classify the condition of the horse whether it lived, died or was euthanized
- To perform classification using ANN and supervsied classifiers such as SVM and Random Forest 
- To perform clustering using Self-rganizing Map (SOM)
- To identify the best approach by comparing between the methods used 
- To compute the confusion matrix, ROC curve and AUROC for the best prediction model of
each classifier

The preprocessing steps carried out were:
- Remove columns with more than 40% of missing values 
- Replace missing values with median value of that particular column for numerical data
- Replace missing values with nearest non-missing value for categorical data

Conclusion: 
- ANN is the best classifier among all as it shows 81.52% classifcation accuracy while RF and SVM shows 76.36% and 56.16 classification accuracy respectively
- SOM was able to cluster the data into two distinct group

