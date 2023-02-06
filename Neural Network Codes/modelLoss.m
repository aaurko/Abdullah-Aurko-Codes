function [loss,gradients] = modelLoss(parameters,X,T,x0,t0,u0,v0,xBC1,tBC1,xBC2,tBC2)

% Make predictions with the initial conditions
% this is the forward propagation, it evaluates the neural network in the 
% interior using the current parameters with inputs X and T and outputs
% the function value U(X,T)
H = model(parameters,X,T);
% real part of H
U = H(1,:);
% imaginary part of H
V = H(2,:);

% Calculate derivatives of U with respect to X and T
gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Ut = gradientsU{2};
Uxx = dlgradient(sum(Ux,"all"),X);

% Calculate derivatives of V with respect to X and T
gradientsV = dlgradient(sum(V,"all"),{X,T},EnableHigherDerivatives=true);
Vx = gradientsV{1};
Vt = gradientsV{2};
Vxx = dlgradient(sum(Vx,"all"),X);

% Calculate lossF_Re. Enforce real part of NLS equation
f_Re = -Vt + Uxx + 2*(U.^2 + V.^2).*U;
% Calculate lossF_Im. Enforce imag part of NLS equation
f_Im = Ut + Vxx + 2*(U.^2 + V.^2).*V;
zeroTarget = zeros(size(f_Re), "like", f_Re);
% calculate the mean square error: 1/n* SUM_i F_i^2 
lossF = mse(f_Re, zeroTarget) + mse(f_Im, zeroTarget);

% Calculate lossIC. Enforce initial conditions
H0Pred = model(parameters,x0,t0);
% Real part of IC
U0Pred = H0Pred(1,:);
% Imag part of IC
V0Pred = H0Pred(2,:);
% % calculate the mean square error of the IC
lossIC = mse(U0Pred, u0) + mse(V0Pred, v0);

% Calculate lossBC. Enforce periodic boundary conditions
HBC1 = model(parameters,xBC1,tBC1);
HBC2 = model(parameters,xBC2,tBC2);
% extract real part
UBC1 = HBC1(1,:); UBC2 = HBC2(1,:);
% extract imag part
VBC1 = HBC1(2,:); VBC2 = HBC2(2,:);

% calculate X derivative along boundary for real part
gradientsUBC1 = dlgradient(sum(UBC1,"all"),{xBC1,tBC1});
UBC1x = gradientsUBC1{1};
gradientsUBC2 = dlgradient(sum(UBC2,"all"),{xBC2,tBC2});
UBC2x = gradientsUBC2{1};
% calculate X derivative along boundary for imag part
gradientsVBC1 = dlgradient(sum(VBC1,"all"),{xBC1,tBC1});
VBC1x = gradientsVBC1{1};
gradientsVBC2 = dlgradient(sum(VBC2,"all"),{xBC2,tBC2});
VBC2x = gradientsVBC2{1};
% calculate the mean square error of the IC
lossBC = mse(UBC1,UBC2) + mse(UBC1x,UBC2x) + ...
         mse(VBC1,VBC2) + mse(VBC1x,VBC2x);  

% Combine losses
loss = lossF + lossIC + lossBC;

% Calculate gradients of the loss function with respect to the 
% learnable parameters
gradients = dlgradient(loss,parameters);

end