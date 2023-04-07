% Classification on horse colic dataset using Artificial Neural Network

% Pseudocode:
% 1. Load the data from horseColic.mat
% 2. Change target variable (outcome- what eventually happened to the
%    horse?) to categorical and one-hot encode it
% 3. Setting up the pattern recognition network
% 4. Split data for training and testing (70:30)
% 5. Train and test the network
% 6. Calculate the performance accuracy by computing the AUROC curve for
%    multiclass
% 7. Plot network performnace, confusion matrix and ROC curve

%Load the data from horseColic.mat
load horseColic.mat 

% check the dimensions of both inputs and targets variable
size(inputs)
size(targets)

%% change targets to categorical
targets = categorical(targets);

% View the order of the categories
categories(targets)

% one-hot encode the targets
targetsOH  = onehotencode(targets,1);
size(targetsOH)

%% Setting up a pattern recognition network
trainFcn = 'trainlm';
hiddenLayerSize = 25;
net = patternnet(hiddenLayerSize,trainFcn);

% Split data for training and testing
net.divideParam.trainRatio = 70/100;
net.divideParam.testRatio = 30/100;

%% train and test the network
%train network
[net,tr] = train(net,inputs,targetsOH);

% Test network
outputs = net(inputs);
errors  = gsubtract(targetsOH, outputs);


%% Calculate performance accuracy 
performance = perform(net, targetsOH, outputs)

%% Plot network performnace, confusion matrix and AUROC curve
figure, plotperform(tr)
figure, plotconfusion(targetsOH,outputs)

%overall percentages of correct and incorrect classification
[c,cm] = confusion(targetsOH,outputs)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%% AUROC for multiclass
% create a rocmetrics object 
rocObj = rocmetrics(targets,transpose(outputs), categories(targets));

% display AUC property
rocObj.AUC

% average ROC & AUC curve using the micro-averaging method
plot(rocObj,AverageROCType="micro")

