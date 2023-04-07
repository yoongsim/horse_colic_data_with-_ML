% Pseudocode

% 1. Load the horse colic dataset  
% 2. Split the data to training:testing (70%:30%)
% 3. Define the Random Forest module and fit the training data into it
% 4. Predict the testing set 
% 5. Calculate accuracy, Plot confusion matrix, ROC curve and compute AUC
% 6. Repeat Step 1 to 5 for 10 times and take the best result

%%
% Load the data from horseColic.mat
load horseColic.mat

% Transpose the matrix
X = inputs.' ;
y = targets.' ;

%%
% Split the data to training(70%) and testing(30%)
% use cvpartition to random partition on dataset
% Hold out for testing dataset
% Split the data in a stratified way so that each class has 70% for
% training and 30% for testing.
splited_data = cvpartition(y,'Holdout',0.3,'Stratify',true);

% Extract the test indices
indices = splited_data.test;
% Retrieve the training data 
% ~ means 'not', so it will only take train set (not test set)
X_train = X(~indices,:);
y_train = y(~indices,:);
% Retrieve the testing data
X_test = X(indices,:);
y_test = y(indices,:);

%%
% fit training data into Random Forest model
% 'Bag' is bagging (boostrap aggregation)
% In bagging, a random sample of data in a training set is selected with replacement
% —meaning that the individual data points can be chosen more than once.
model = fitensemble(X_train,y_train,'Bag',100,'Tree','Type','classification');

% Predict the model on testing data
[~,Score] = predict(model,X_test);

% Plot ROC
rocObj = rocmetrics(y_test,Score,model.ClassNames);
rocObj.AUC
idx = strcmp(rocObj.Metrics.ClassName,model.ClassNames(1));
head(rocObj.Metrics(idx,:))
plot(rocObj)

%% confusion matrix
% Calculate prediction accuracy in percentage
% Predict the model on testing data
[prediction] = predict(model,X_test);

% Calculate prediction accuracy in percentage
accuracy = sum(prediction == y_test)/length(y_test)*100

%% 
% Display confusion matrix
cm1 = confusionmat(y_test,prediction)
cm = confusionchart(cm1)
cm_plot = confusionchart(y_test,prediction)
cm_plot.Title = 'Confusion Matrix for Classification by RF';