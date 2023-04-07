%Clustering on horse colic dataset with Self Organising Map (SOM) Neural
%Network

%%Pseudocode:

% 1. Load horse colic dataset
% 2. Assign the input matrix to variables
% 3. Set dimensions size
% 4. Create a Self-Organizing Map
% 5. Train and test the network
% 6. Plot som topology, neigbour connection, neighbour distance,
% planes,hits and positions


%% Load horse colic dataset
load horseColic.mat

%% horsecolic - input data.
x = input;

 %% Create a Self-Organizing Map
dimension1 = 15;
dimension2 = 15;
net = selforgmap([dimension1 dimension2]);

%% Train the Network

[net,tr] = train(net,x);
%% Test the Network
outputs = net(x);

%% View the Network
view(net)

%% Plots
figure, plotsomtop(net)
figure, plotsomnc(net)
figure, plotsomnd(net)
figure, plotsomplanes(net)
figure, plotsomhits(net,x)
figure, plotsompos(net,x)