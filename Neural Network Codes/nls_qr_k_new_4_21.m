
clear all
Lx=32;

%  N=256;
N=512;

dx=Lx/N; x=[-Lx/2:dx:Lx/2-dx]'; %kx=[0:N/2-1 -N/2:-1]'*2*pi/Lx;


mu=3; sigma=-1; theta=0; s=1;
%psi=sqrt(2*mu/gamma)*sech(sqrt(2*mu)*x);

%classical NLS (and RT)

% q_0=sqrt(mu)*sech(sqrt(mu)*x);
% r_0=sigma*conj(q_0);

%PT Symm NLS

q_0=sqrt(mu)*sech(sqrt(mu)*x-i*theta);
r_0=sigma*conj(sqrt(mu)*sech(sqrt(mu)*(-x)-i*theta));

%Spectral differentiation matrix
[xx,Dxx] = fourdif(N,2);
Dxx = Dxx*(2*pi/Lx)^2;


k_save = (0:0.04:4)';
P = length(k_save);
imag_save = [];
%Z = zeros(N,N);

for pp = 1:P
    k = k_save(pp);
    
    
%     L1= -Dxx + diag(mu+6*sigma*q_0.*r_0+s*k^2);
%     L2= -Dxx + diag(mu+2*sigma*q_0.*r_0+s*k^2);


L1= -Dxx + diag(mu+6*sigma*q_0.^2+k^2);
L2= -Dxx + diag(mu+2*sigma*q_0.^2+k^2);

%     M_11=L1;
%     M_22=L2;

    M_12=L1;
    M_21=L2;
    
    Z = zeros(N,N);  
%     MAT = [M_11 Z; Z M_22]; 
    MAT = [Z M_12; M_21 Z]; 
   
    [Vec,E] = eig(MAT); % find eig
    imag_save = [imag_save; norm(imag(diag(E)),inf) ];
end

plot(k_save,imag_save, 'r')
%omega=E(index(1),index(1));
hold on
% xline(0.94248, '--k')
% plot(k_save,2*sqrt(mu)*k_save, 'DisplayName','Asymptotic fit', 'LineStyle','--b')
plot(k_save,2*sqrt(mu)*k_save, '--r')
xlabel('$l$','Interpreter','LaTeX','Fontsize',18)
%ylabel('Im(\omega)')
ylabel('$Im(\omega)$','Interpreter','LaTeX','Fontsize',18)
axis square

