%% Set up import options and import data
% delimitedTextImportOptions control the data import process, including the handling of errors and missing data.
opts = delimitedTextImportOptions("NumVariables", 28); % number of variables specified, 28

% Declare range and delimiter
opts.DataLines = [1, Inf]; %sets the property to read lines starting from row 1 to the end of file
opts.Delimiter = " ";

% Declare variable properties, column names and types
opts.VariableNames = ["Surgery", "Age", "Hospital Number", ...
    "Rectal Temperature", "Pulse", "Respiratory Rate", ...
    "Temperature of Extremities", "Peripheral Pulse", "Mucous Membranes", ...
    "Capillary Refill Time", "Pain", "Peristalsis", "Abdominal Distension",...
    "Nasogastric Tube", "Nasogastric Reflux", "Nasogastric Reflux pH", ...
    "Rectal Examination", "Abdomen", "Packed Cell Volume", "Total Protein", ...
    "Abdominocentesis Appearance", "Abdomcentesis Total Protein", "Outcome", ...
    "Surgical Lesion", "Type of Lesion", "Type of Lesion", "Type of Lesion", "Cp_data"];
opts.VariableTypes = ["categorical", "categorical", "string", "double", ...
    "double", "double", "categorical", "categorical", "categorical", ...
    "categorical", "categorical", "categorical", "categorical", ...
    "categorical", "categorical", "double", "categorical", "categorical",...
    "double", "double", "categorical", "double", "categorical", ...
    "categorical", "string", "string", "string", "categorical"];

% Declare file level properties
opts.EmptyLineRule = "read"; 	%Import the empty lines
opts.ConsecutiveDelimitersRule = "join"; %Join the delimiters into one delimiter
opts.LeadingDelimitersRule = "ignore"; %Ignore the delimiter
opts.ExtraColumnsRule = "ignore"; %Ignore the extra columns of data

% Declare variable properties (updates all the variables in the opts object based on the specifications listed)
opts = setvaropts(opts, ["Hospital Number", "Type of Lesion", ... 
    "Type of Lesion", "Type of Lesion"], "WhitespaceRule", "preserve"); %preserve leading and trailing white spaces

opts = setvaropts(opts, ["Surgery", "Age", "Hospital Number", ...
    "Temperature of Extremities", "Peripheral Pulse", "Mucous Membranes", ...
    "Capillary Refill Time", "Pain", "Peristalsis", "Abdominal Distension", ...
    "Nasogastric Tube", "Nasogastric Reflux", "Rectal Examination", ...
    "Abdomen", "Abdominocentesis Appearance", "Outcome", "Surgical Lesion", ...
    "Type of Lesion", "Type of Lesion", "Type of Lesion", "Cp_data"], ...
    "EmptyFieldRule", "auto"); %imports the empty fields based on data type of the variable

opts = setvaropts(opts, ["Nasogastric Reflux pH", "Abdomcentesis Total Protein"], ...
    "TrimNonNumeric", true); %Remove nonnumeric characters

opts = setvaropts(opts, ["Nasogastric Reflux pH", "Abdomcentesis Total Protein"], ...
    "ThousandsSeparator", ","); %',' is used to indicate thousands grouping

% preview the data
preview('horse-colic.data',opts)
preview('horse-colic.test',opts)


% Import the predefined train and test data
trainData = readtable("horse-colic.data", opts);
testData = readtable("horse-colic.test", opts); 

%% Data examination and cleaning
% Combine both predefined dataset into one
data = [trainData; testData];

% Standardize missing values according to type
dataClean = standardizeMissing(data,"?");

% Count number of missing values for each variable
sum(ismissing(dataClean))

% Delete columns with more than 40% missing values (threshold value = 40%)
dataClean = rmmissing(dataClean,2,'MinNumMissing',147);

% Fill missing values for columns less than 40% missing values
% Extract numerical columns only
dataNum = dataClean(:,[4:6,18:19]);
dataNum = table2array(dataNum);

% Replace missing values with median value of that particular column for numerical data
NumFill = fillmissing(dataClean,'constant',median(dataNum,1,'omitnan'), 'DataVariables',@isnumeric);

% Replace missing values with nearest non-missing value for categorical data
AllFill = fillmissing(NumFill,"nearest",'DataVariables', @iscategorical);

%% Prepare input & target matrix
%Convert categorical data to numerical data
input = convertvars(AllFill, (1:3), 'double');
input = convertvars(input, (7:17), 'double');
input = convertvars(input, (20:25), 'double');
input = table2array(input(:, 1:25));

%Input matrix for clustering
input = input';

%Input and target matrix for classification
inputs = input([1:2,4:19,21:24],:);
targets = input(20, :); %take outcome as target
