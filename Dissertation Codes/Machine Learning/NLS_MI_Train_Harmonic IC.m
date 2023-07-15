% Time evolution of 1D nonlinear Schrodinger equation
% i*psi_t + psi_xx + g*|psi|^2*psi = 0
% 4th-order accurate split-step (SS) method is used to integrate the PDE
% 
 clear all; InputTrain = []; OutputTrain = [];
format long

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% spatial discretization in x
Lx = 6;       % computational domain [-Lx/2,Lx/2]
Nx = 128;      % number of spatial grid points
dx = Lx/Nx;     % grid spacing
x = dx*(-Nx/2:Nx/2-1)';         % spatial computational domain
% Note: remove last point due to periodic BCs
kx = (2*pi/Lx)*[0:Nx/2-1 -Nx/2:-1]';       % Fourier computational domain

% nonlinearity coefficient: g = 2 is focusing, g = -2 is defocusing
g = 2;

% time discretization
T = 5;
dt = .001;                  % time step
S = round(T/dt);                   % number of time steps

% split-step scheme coefficients
c = 1/(2-2^(1/3));                     
a1 = c/2; a2 = (1-c)/2; a3 = a2; a4 = c/2;
b1 = c; b2 = 1-2*c; b3 = c;

% exact solution of linear part of PDE
E1 = exp(-a1*dt*1i*kx.^2);
E2 = exp(-a2*dt*1i*kx.^2);
E3 = exp(-a3*dt*1i*kx.^2);
E4 = exp(-a4*dt*1i*kx.^2);

% initial condition
bckgrd = 1;
q_IC = bckgrd + 0.1*cos(2*pi*2*x/Lx);
%  q_IC = bckgrd + 0.*cos(2*pi*9*x/Lx);
t0 = 0;

% initial conditions
t = t0; q = q_IC;

% storage information
t_save = t;
q_save = q;
count = 0;
savestep = 100;

rogue_crit = 2;
rogue_event = 0;

% start split-step method
for ii = 1:S

  t = t + dt;   
  v = ifft(fft(q).*E1);
  v = v.*exp(b1*dt*1i*(g*abs(v).^2));
  v = ifft(fft(v).*E2);
  v = v.*exp(b2*dt*1i*(g*abs(v).^2));
  v = ifft(fft(v).*E3);
  v = v.*exp(b3*dt*1i*(g*abs(v).^2));
  q = ifft(fft(v).*E4);
  
  
  count = count + 1;
  if count == savestep
  % save for plotting
  t_save = [t_save; t];
  q_save = [q_save q];
  count = 0;
  end
  
  % check for instability
  if max(abs(q)) >= rogue_crit*bckgrd
    rogue_event = 1;
  end

end

% output the result of the evolution
if round(rogue_event) == 1
    disp('An instability was detected')
else
    disp('No instability detected')
end

% plot evolution of solution
figure(1)
surface(x,t_save,abs(transpose(q_save)))
shading interp; axis tight
set(gca,'FontSize',20)
xlabel('x','FontSize',25)
ylabel('t','FontSize',25,'rot',00)
title('|q(x,t)|','FontSize',20)
colorbar

% save IC
InputTrain = [InputTrain q_IC];
% save stability result
OutputTrain = [OutputTrain round(rogue_event)];
   
