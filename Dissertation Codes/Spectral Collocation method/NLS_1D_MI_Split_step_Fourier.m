% Time evolution of 1D nonlinear Schrodinger equation
% i*psi_t + psi_xx + g*|psi|^2*psi = 0
% 4th-order accurate split-step (SS) method is used to integrate the PDE

clear all
format long

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% spatial discretization in x
Lx = 6;       % computational domain [-Lx/2,Lx/2]
Nx = 2*128;      % number of spatial grid points
dx = Lx/Nx;     % grid spacing
x = dx*(-Nx/2:Nx/2-1)';         % spatial computational domain
% Note: remove last point due to periodic BCs
kx = (2*pi/Lx)*[0:Nx/2-1 -Nx/2:-1]';       % Fourier computational domain

% nonlinearity coefficient: g = 2 is focusing, g = -2 is defocusing
g = 2;

% initial condition
% q_IC = 1 + 0.1*cos(2*pi*x/Lx);
q_IC = 1 + 0*cos(2*pi*x/Lx);
t0 = 0;

% initial conditions
t = t0; q = q_IC;

% time discretization
% T = 1.5;
T = 1;
% dt = .001;
dt = .0001;                  % time step
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

% storage information
t_save = t;
q_save = q;
count = 0;
savestep = 100;

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

end

% plot evolution of solution
figure(1)
subplot(1,2,1)
surface(x,t_save,abs(transpose(q_save)))
shading interp; axis tight
set(gca,'FontSize',20)
xlabel('x','FontSize',25)
ylabel('t','FontSize',25,'rot',00)
title('|q(x,t)|','FontSize',20)
colorbar

surface(t_save,x,abs(q_save))
shading interp; axis tight
set(gca,'FontSize',20)
xlabel('t','FontSize',25)
ylabel('x','FontSize',25,'rot',00)
title('|q(x,t)|','FontSize',20)
colorbar

subplot(1,2,2)
surface(x,t_save,abs(transpose(q_save)))
shading interp; axis tight
set(gca,'FontSize',20)
xlabel('x','FontSize',25)
ylabel('t','FontSize',25,'rot',00)
zlabel('|q(x,t)|','FontSize',20)

% solution at final time t = t0 + T
exact_q = exp(1i*g*t)

final_error = norm(q-exact_q,inf)   
realq=real(q_save);
imagq=imag(q_save);
   
file_title = ['NLS_MI_soln_real.data'];
save(file_title,'realq','-ascii','-double')

file_title = ['NLS_MI_soln_imag.data'];
save(file_title,'imagq','-ascii','-double')

   
