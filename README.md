# Horse Colic Data Classification with Machine Learning

This project aims to classify the condition of horses (whether they lived, died, or were euthanized) using various machine learning techniques. The raw data were obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Horse+Colic).

## Objectives

**Classification**: Use Artificial Neural Networks (ANN), Support Vector Machines (SVM), and Random Forest (RF) to classify horse conditions.<b/>
**Clustering**: Apply Self-Organizing Map (SOM) for clustering.<b/>
**Evaluation**: Identify the best approach by comparing methods used and compute the confusion matrix, ROC curve, and AUROC for the best prediction model of each classifier.<b/><b/>

## Preprocessing Steps
- Remove columns with more than 40% missing values.<b/>
- Replace missing values with the median value for numerical data.<b/>
- Replace missing values with the nearest non-missing value for categorical data.<b/><b/>

## Conclusion
ANN: Best classifier with 81.52% accuracy.<b/>
Random Forest: 76.36% accuracy.<b/>
SVM: 56.16% accuracy.<b/>
SOM: Successfully clustered data into two distinct groups.<b/>

