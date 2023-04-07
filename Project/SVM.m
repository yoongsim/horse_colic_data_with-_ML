%% Pseudocode
%1. Load horseColic.mat
%2. Assign the input and target matrix to variables
%3. Split data for training and testing
%4. Store the standardized support vectors of each SVM. Unstandardize the support vectors.
%5. Plot the data, and identify the support vectors.
%6. Compute the confusion matrix

%% Load data set
load horseColic.mat

% assign inputs and targets into variables
X = inputs';
Y = targets';

% convert array into table
tbl = array2table(X);
tbl.Y = Y;

%% Split data for training and testing

% For reproducibility
rng('default') 
n = length(tbl.Y);

% Nonstratified partition
hpartition = cvpartition(n,'Holdout',0.3);
idxTrain = hpartition.test;

% Retrieve the training data
X_train = X(~idxTrain,:);
y_train = Y(~idxTrain,:);

% Retrieve the testing data
X_test = X(idxTrain,:);
y_test = Y(idxTrain,:);

%% Train model

% Store the standardized support vectors of each SVM.
t = templateSVM('Standardize',true,'SaveSupportVectors',true);
responseName = 'Outcomes';
classNames = {'1','2','3'}; % Specify class order

% Train the model using fitececoc function
Md1 = fitcecoc(X_train,y_train,'Learners',t,'ResponseName',responseName,'ClassNames',classNames);

% Obtain the classname
Md1.ClassNames

% Obtain the coding matrix
Md1.CodingMatrix

%Store the standardized support vectors of each SVM. 
% Unstandardize the support vectors.
L = size(Md1.CodingMatrix,1); % Number of SVMs
sv = cell(L,1); % Preallocate for support vector indices
for j = 1:L
    SVM = Md1.BinaryLearners{j};
    sv{j} = SVM.SupportVectors;
    sv{j} = sv{j}.*SVM.Sigma + SVM.Mu;
end

% Plot the data, and identify the support vectors.
figure
gscatter(X_test,y_test);
xlabel('X Test')
ylabel('Y Test')
hold on
markers = {'ko','ro','bo'}; % Should be of length L
for j = 1:L
    svs = sv{j};
    plot(svs(:,1),markers{j},'MarkerSize',10 + (j - 1)*3);
end

title('Horse COlic -- ECOC Support Vectors')

legend([classNames,{'Support vectors - SVM 1',...
    'Support vectors - SVM 2','Support vectors - SVM 3'}],'Location','Best')
hold off

%% Confusion matrix

% Train SVM classification model 
Md1 = fitcecoc(X,Y);

% Predict the labels of the training data.
predictedY = resubPredict(Md1);

%Create a confusion matrix chart from the true labels Y and the predicted 
% labels predictedY.
cm = confusionchart(Y,predictedY);

%The NormalizedValues property contains the values of the confusion matrix.
cm.NormalizedValues

% Add a title.
cm.Title = 'Horse Colic Classification Using fitcecoc';

% Add column and row summaries.
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ROC AND AUROC

% Train SVM classification model 
Md1 = fitcecoc(X_train,y_train,'Learners',t,'ResponseName',responseName);

%Compute the classification scores for the test set.
[~,Score] = predict(Md1,X_test);

% Create a rocmetrics object by using the true labels in YTest and the classification scores in scores.
% Specify the column order of Scores using Mdl.ClassNames.
rocObj = rocmetrics(y_test,Score,Md1.ClassNames);

% Display the AUC property.
rocObj.AUC

% Plot the ROC curve for each class by using the plot function.
plot(rocObj)

% The table in Metrics contains the performance metric values for all three
% classes.
idx = strcmp(rocObj.Metrics.ClassName,Mdl.ClassNames(1));

% Find and display the rows for the second class in the table.
rocObj.Metrics(idx,:)

% Specify AverageROCType="micro" to compute the performance metrics for the 
% average ROC curve using the micro-averaging method. 
plot(rocObj,AverageROCType="micro")
