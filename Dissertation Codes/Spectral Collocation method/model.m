function U = model(parameters,X,T)

% 2 input data quantities
% X and T are row vectors, XT is a 2xN data vector in size
XT = [X;T];
% this counts the number of total layers, including input and output
numLayers = numel(fieldnames(parameters));

% call first fully connect operation
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
% compute argurment of the activation function U = weights*XT + bias
U = fullyconnect(XT,weights,bias);

% tanh and fully connect operations for remaining layers
for i=2:numLayers
    name = "fc" + i;
    % evlauate the argument in the activation function
%     U = tanh(U);
     U = 1./(1 + exp(-U));
    % compute the next argument
    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    U = fullyconnect(U, weights, bias);
end

end
