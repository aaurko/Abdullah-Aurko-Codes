function [loss,gradients] = modelLoss(parameters,U,OutputTrain)

% make predictions with the initial conditions

% forward propagation
NNOutput = model(parameters,U);
% loss function
loss = mse(NNOutput,OutputTrain);

% Calculate gradients of the loss function with respect to the 
% learnable parameters
gradients = dlgradient(loss,parameters);

end
