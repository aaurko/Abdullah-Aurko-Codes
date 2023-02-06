% function [U,V] = solveNLS(X,t,mu)
function [U,V] = solveNLS(X,t,A)


% Soliton solution
% U = sqrt(mu)*sech(sqrt(mu)*X)*cos(mu*t);
% V = sqrt(mu)*sech(sqrt(mu)*X)*sin(mu*t);

%Periodic solution
U = A*cos(2*A^2*t);
V = A*sin(2*A^2*t);
% U = (1 + 0.1*cos(pi*X/3))*cos(2*A^2*t);
% V = (1 + 0.1*cos(pi*X/3))*sin(2*A^2*t);
end