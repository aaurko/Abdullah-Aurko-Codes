% clear all

% domain length
L = 6;
% number of spatial points
NumSpatialPts = 128;
% spatial grid
x = linspace(-L/2,L/2,NumSpatialPts)';
% number of initial conditions
% initial conditions, periodic
% u1 = 1*ones(size(x));
% u2 = 1*ones(size(x)) + 0.1*cos(2*pi*x/L);
% u3 = 1*ones(size(x)) + 0.1*cos(2*pi*2*x/L);
% stability output training data
% OutputTrain = [0 1];
% OutputTrain = [0 1 1];
% store data
% ds = arrayDatastore([u1 u2]);
% ds = arrayDatastore([u1 u2 u3]);
ds = arrayDatastore(InputTrain);

% number of layers
numLayers = 4;
numNeurons = 4;

% 1 input layer with NumStates inputs: u1, u2, ..., u_NumStates
% numLayers = number of hidden layers with numNeurons neurons each
% 1 output layer with NumStates output: 0 = stable, 1 = unstable

% pre-allocate parameters field
parameters = struct;
% size of of the weight matrix between input layer and first hidden layer
sz = [numNeurons NumSpatialPts];
% randomly assign weights, structure is size of sz
parameters.fc1.Weights = initializeHe(sz,NumSpatialPts);
% inititally assign zeros for the biases
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

% parameters.fc1 contains the weights and biases between the input layer 
% and the first hidden layer

% construct the weights and biases between all the hidden layers
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end

% allocate weights and biases between last hidden layer and output layer
sz = [1 numNeurons];
numOut = 1;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numOut);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);

parameters


numEpochs = 100000;
miniBatchSize = NumSpatialPts;

executionEnvironment = "auto";

initialLearnRate = 0.01;
 decayRate = 0.001;

% mbq stands for mini batch queue ...
mbq = minibatchqueue(ds, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFormat="CB");%, ...
%     OutputEnvironment=executionEnvironment);

% enable GPU, if one is available
% if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
%     X0 = gpuArray(X0);
%     T0 = gpuArray(T0);
%     U0 = gpuArray(U0);
% end

averageGrad = [];
averageSqGrad = [];

% supposed to speed up the computations
accfun = dlaccelerate(@modelLoss);

figure
C = colororder;
lineLoss = animatedline(Color=C(2,:));
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

start = tic;

iteration = 0;
loss = 10;

tol=1e-6;
% tol=1e-8;
epoch=0;

while loss>tol && epoch<=numEpochs
    reset(mbq);
    
    while hasdata(mbq)
        iteration = iteration + 1;

        U = next(mbq);

        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        % this is applying automatic differentiation
        [loss,gradients] = dlfeval(accfun,parameters,U,OutputTrain);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration,loss);
    epoch=epoch+1;

    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
    
     
end


accfun

% minimal loss values
loss
epoch

stop

% compare output
u = 1*ones(size(x)) + 0.*cos(2*pi*x/L) + 0.*cos(2*pi*2*x/L)  ...
    + 0.*cos(2*pi*3*x/L);

u_input = dlarray(u,"CB");
NNOutput = model(parameters,u_input);
stability = double(extractdata(NNOutput))

if round(stability)==1 || round(stability)>=1
    disp('Initial Condition Appears to be Unstable')
elseif round(stability)<=0
    disp('Initial Condition Appears to be Stable')
end
    
