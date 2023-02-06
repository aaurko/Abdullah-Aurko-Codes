% the 4th-order integrating-factor method for solving the QR
% system:  iq_t - q_{xx} - q_{yy} + 2*r*q^2 = 0
%         -ir_t + r_{xx} + r_{yy} + 2*q*r^2 = 0

clear all

Lx = 40; Ly = 20;
Nx=4*128; Ny=2*128;
% Nx = 8*128; Ny = 4*128; 

dx=Lx/Nx; x=[-Lx/2:dx:Lx/2-dx]'; kx=[0:Nx/2-1 -Nx/2:-1]'*2*pi/Lx;
dy=Ly/Ny; y=[-Ly/2:dy:Ly/2-dy]'; ky=[0:Ny/2-1 -Ny/2:-1]'*2*pi/Ly;

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2;

dt = 0.001; tmax=1; nmax=round(tmax/dt);

Eq = exp(-1i*K2*dt/2); E2q = Eq.*Eq;
Er = exp(1i*K2*dt/2); E2r = Er.*Er;

mu=1; sigma=-1; theta=0;

%mu = 1, epsilon = 0.1 fixed

%classical NLS (and RT)
%  q_0 = sqrt(mu)*sech(sqrt(mu)*x);
%  r_0 = sigma*conj(q_0);

%PT Symm NLS

q_0 = sqrt(mu)*sech(sqrt(mu)*x-1i*theta);
r_0 = sigma*conj(sqrt(mu)*sech(sqrt(mu)*(-x)-1i*theta));

[xx,Dxx] = fourdif(Nx,2);
Dxx = Dxx*(2*pi/Lx)^2;
 
l = (2*pi*5)/Ly;
% l = 3*(2*pi*5)/Ly;

L1 = -Dxx + diag(mu+6*sigma*q_0.^2+l^2);
L2 = -Dxx + diag(mu+2*sigma*q_0.^2+l^2);

Z=zeros(Nx,Nx);   
  MAT = [Z L1; L2 Z];     
  [Vec,E] = eig(MAT); % find eig
[eig_value, index]= sort(imag(diag(E)),'ascend');

F = Vec(1:Nx,index(1));
G = Vec(Nx+1:2*Nx,index(1));


% PSI=sqrt(2*mu/gamma)*sech(sqrt(2*mu)*X); 
F_2d=repmat(transpose(F),Ny,1);
G_2d=repmat(transpose(G),Ny,1);

f = (F_2d + G_2d)/2;
g = (G_2d - F_2d)/2;

%eta=f_2d.*exp(i*q*Y)+conj(g_2d).*exp(-i*q*Y);
Q = f.*exp(1i*l*Y) + conj(g).*exp(-1i*l*Y);
R = sigma*g.*exp(1i*l*Y) + sigma*conj(f).*exp(-1i*l*Y);

% PHI=PSI+epsilon*eta;
q_0 = sqrt(mu)*sech(sqrt(mu)*X-1i*theta);
r_0 = sigma*conj(sqrt(mu)*sech(sqrt(mu)*(-X)-1i*theta));

epsilon = 0.1;
q = q_0 + epsilon*Q;
r = r_0 + epsilon*R;


%  plot(x,real(Q(Ny/2+1,:)),x,real(R(Ny/2+1,:)),'--','LineWidth',2)
 plot(x,imag(Q(Ny/2+1,:)),x,imag(R(Ny/2+1,:)),'--','LineWidth',2)


set(gca,'FontSize',18)
xlabel('x','FontSize',20)
ylabel('y','FontSize',20)
% ylabel('Im(Q)','FontSize',20)
stop

qx=ifft2(1i*KX.*fft2(q)); rx=ifft2(1i*KX.*fft2(r));
qy=ifft2(1i*KY.*fft2(q)); ry=ifft2(1i*KY.*fft2(r));

gamma3=sum(sum(qx.*rx +qy.*ry+ (q.*r).^2))*dx*dy
real(gamma3)

V=sum(sum((X.^2+Y.^2).*q.*r))*dx*dy

gamma1=sum(sum(q.*r))*dx*dy;
Vt=-4i*sum(sum(X.*q.*rx+Y.*q.*ry))*dx*dy-4i*gamma1
t_star=(-Vt+sqrt(Vt^2-16*gamma3.*V))/(8*gamma3)


% stop

qdata=q; rdata=r; tdata=0;
%our equations are: 1. iq_t-Lap(q)+2q^2r=0 and 2. ir_t+Lap(r)-2r^2q=0
%previous equation was: iphi_t+0.5*Lap(phi)+gamma*|phi^2|*phi=0 

  for nn = 1:nmax                       % integration begins
   %v=fft(u);                  dv1=gamma*i*fft(u.*u.*conj(u));
    vq=fft2(q);                  dv1q=2i*fft2(q.*q.*r);
    vr=fft2(r);                  dv1r=-2i*fft2(r.*r.*q);
    
    wq=ifft2((vq+dv1q*dt/2)./Eq);  wr=ifft2((vr+dv1r*dt/2)./Er);
    dv2q=2i*Eq.*fft2(wq.*wq.*wr); dv2r=-2i*Er.*fft2(wr.*wr.*wq);
    
    wq=ifft2((vq+dv2q*dt/2)./Eq); wr=ifft2((vr+dv2r*dt/2)./Er);  
    dv3q=2i*Eq.*fft2(wq.*wq.*wr);dv3r=-2i*Er.*fft2(wr.*wr.*wq);
    
    wq=ifft2((vq+dv3q*dt)./E2q);  wr=ifft2((vr+dv3r*dt)./E2r);
    dv4q=2i*E2q.*fft2(wq.*wq.*wr); dv4r=-2i*E2r.*fft2(wr.*wr.*wq);
    
    vq=vq+(dv1q+2*dv2q+2*dv3q+dv4q)*dt/6;  q=ifft2(vq./E2q);
    vr=vr+(dv1r+2*dv2r+2*dv3r+dv4r)*dt/6;  r=ifft2(vr./E2r);
    
        
%     if mod(nn,round(nmax/100)) == 0
% %      udata=[udata u]; tdata=[tdata nn*dt];
%        qdata=[qdata q]; rdata=[rdata r];tdata=[tdata nn*dt];
%     end
  end                                   % integration ends
  
%   u_final=sqrt(2*mu/gamma)*sech(sqrt(2/mu)*x)*exp(i*mu*tmax);
  %q_final=sqrt(mu)*sech(sqrt(mu)*X)*exp(-i*mu*tmax);
  
  %Classical and RT
%   q_final=sqrt(mu)*sech(sqrt(mu)*X)*exp(-i*mu*tmax);
%   r_final=sigma*conj(q_final);
  
  %PT Sym
  q_final=sqrt(mu)*sech(sqrt(mu)*X-i*theta)*exp(-i*mu*tmax);
  r_final=sigma*conj(sqrt(mu)*sech(sqrt(mu)*(-X)-i*theta)*exp(-i*mu*tmax));
%  r_final=sigma*conj(q_final);
  
  
  
diffq=abs(q_final-q);
diffr=abs(r_final-r);

errq=max(max(diffq))
errr=max(max(diffr))

surface(x,y,abs(q))
shading interp
  
%   waterfall(x, tdata, abs(udata'));     % solution plotting
%   colormap([0 0 0]); view(5, 60)
  xlabel('x', 'fontsize', 10)
  ylabel('y', 'fontsize', 10)
  colorbar
%   zlabel('|u|', 'fontsize', 15)
%   axis([-L/2 L/2 0 tmax 0 2]); grid off
  
  