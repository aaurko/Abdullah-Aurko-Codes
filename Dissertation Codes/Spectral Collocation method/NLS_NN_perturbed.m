clear all

% equally spaced time points to enforce BCs
% first component is number of points at left endpoint
% second component is number of points at right endpoint
numBoundaryConditionPoints = [50 50];

% domain length
%L = 20;
L = 6;

% time interval length
 Ti = 1.5; 
%  Ti = 2; 
% left endpoint, constant
xBC1 = -L/2*ones(1,numBoundaryConditionPoints(1));
% right endpoint, constant
xBC2 = L/2*ones(1,numBoundaryConditionPoints(2));

% equally spaced time points at left endpoint
tBC1 = linspace(0,Ti,numBoundaryConditionPoints(1));
% equally spaced time points at right endpoint
tBC2 = linspace(0,Ti,numBoundaryConditionPoints(2));

% equally spaced spatial points to enforce ICs
numInitialConditionPoints = 50;

x0 = linspace(-L/2,L/2,numInitialConditionPoints);
t0 = zeros(1,numInitialConditionPoints);


% % initial condition, periodic
% A=1;
% u0 = A*ones(size(x0));
% v0 = zeros(size(x0));


 % initial condition, modulational instability
A=1;
% u0 = A*ones(size(x0))+ 0.1*cos(pi*x0/3);
u0 = 1+ 0.5*cos(pi*x0/3);
v0 = zeros(size(x0));



% points used to satisfy equation residual
numInternalCollocationPoints = 20000;

% seed a two-dimensional quasirandom Sobol sequence for (x,t) grid
pointSet = sobolset(2);
% generate the grid points between 0 and 1
% first column of points indicates the eventual x values
% second column of points indicates the eventual t values
points = net(pointSet,numInternalCollocationPoints);

% transform interior x points from [0,1] to [-L/2,L/2]
dataX = L*points(:,1)-L/2;
% transform interior x points from [0,1] to [0,Ti]
dataT = Ti*points(:,2);

% store interior x and t training data
ds = arrayDatastore([dataX dataT]);

% number of layers
numLayers = 7;
numNeurons = 15;

% 1 input layer with 2 inputs: x and t
% numLayers-2 hidden layers with numNeurons neurons each
% 1 output layer with 2 output: real part u(x,t), imag part v(x,t)

% pre-allocate parameters field
parameters = struct;
% size of of the weight matrix between input layer and first hidden layer
sz = [numNeurons 2];
% randomly assign weights, structure is size of sz
parameters.fc1.Weights = initializeHe(sz,2);
% inititally assign zeros for the biases
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

% parameters.fc1 contains the weights and biases between the input layer 
% and the first hidden layer

% construct the weights and biases between all 7 hidden layers
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end


% allocate weights and biases between last hidden layer and output layer
sz = [2 numNeurons];
numOut = 2;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numOut);
parameters.("fc" + numLayers).Bias = initializeZeros([2 1]);


parameters

numEpochs = 100000;
miniBatchSize = 1000;

executionEnvironment = "auto";

initialLearnRate = 0.05;
decayRate = 0.001;
% decayRate = 0.0001;

% mbq stands for mini batch queue ...
mbq = minibatchqueue(ds, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFormat="BC", ...
    OutputEnvironment=executionEnvironment);

% convert X0, T0, U0, V0 into data format ...
% C stands for channel
% B stands for batch
x0 = dlarray(x0,"CB");
t0 = dlarray(t0,"CB");
u0 = dlarray(u0);
v0 = dlarray(v0);

xBC1 = dlarray(xBC1,"CB");
tBC1 = dlarray(tBC1,"CB");
xBC2 = dlarray(xBC2,"CB");
tBC2 = dlarray(tBC2,"CB");

% 
% % enable GPU, if one is available
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
tol=1e-3;
epoch=0;


%  for epoch = 1:numEpochs
while loss>tol && epoch<=numEpochs

    reset(mbq);
    
    while hasdata(mbq)
        iteration = iteration + 1;

        XT = next(mbq);
        X = XT(1,:);
        T = XT(2,:);

        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        % this is applying automatic differentiation
        [loss,gradients] = dlfeval(accfun,parameters,X,T,x0,t0,u0,v0,xBC1,tBC1,xBC2,tBC2);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end

   

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);
    epoch=epoch+1;

    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow

   

end

accfun





%  tTest = [.5 1 1.5 2];
 tTest = [.25 .75 1 1.5];

numPredictions = 257;



XTest = linspace(-L/2,L/2,numPredictions);
XTest(end)=[];
XTest = dlarray(XTest,"CB");

figure(2)

for i=1:numel(tTest)
    t = tTest(i);


    TTest = t*ones(1,numPredictions-1);

    % Make predictions.

    TTest = dlarray(TTest,"CB");
    HPred = model(parameters,XTest,TTest);
    HNN = extractdata(HPred(1,:)) + 1i*extractdata(HPred(2,:));
    
    % Calculate true values.
   [UTest,VTest] = solveNLS(extractdata(XTest),t,A);
   HTest = UTest + 1i*VTest;

    % Calculate error.

err = norm(HNN - HTest, inf);

    % Plot predictions.
    subplot(1,4,i)

    plot(extractdata(XTest),extractdata(sqrt((HPred(1,:)-UTest).^2 + (HPred(2,:)-VTest).^2)),"-",LineWidth=2);
    ylim([0, 1.5])



    title("t = " + t + ", Error = " + gather(err));
end


subplot(1,4,i)
legend("Predicted","True")



% tTest = linspace(0,Ti,201);
tTest = linspace(0,Ti,151);

H_store = [];



for i=1:numel(tTest)
    t = tTest(i);

TTest = t*ones(1,numPredictions-1);

    % Make predictions.

    TTest = dlarray(TTest,"CB");
    HPred = model(parameters,XTest,TTest);
    H = extractdata(HPred).';
    H_store = [H_store H(:,1)+1i*H(:,2)];
    
end

load NLS_MI_soln_real.data
load NLS_MI_soln_imag.data

q_save=NLS_MI_soln_real +1i*NLS_MI_soln_imag;

x_plot = extractdata(XTest)';
t_plot = tTest';
[T,X]=meshgrid(t_plot, x_plot);
figure(3)

% surface(T,X,abs(H_store-q_save)); shading interp

%  surface(T,X,abs(H_store)); shading interp
 surface(T,X,abs(q_save)); shading interp

xlim([0 Ti]); ylim([-5 5])
axis tight
set(gca,'FontSize',18)
xlabel('t','FontSize',20)
ylabel('x','FontSize',20)
title('|h(x,t)|','FontSize',20)
colorbar



max_err=max(max(abs(H_store-q_save)))


beep
