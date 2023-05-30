%%
%This is a demo for the proposed SAM algorithm in the paper 
%    ``Sample-Efficient Sparse Phase Retrieval via Stochastic Alternating Minimization''
%By YOU Juntao & Jiao Yuling 
%%
clc
close all
addpath(genpath(fileparts(mfilename('fullpath'))));

%% ===============================================
%generate data:dimension,signal,matrix,measurements and so on
n = 2000;                        %signal dimension
m = 800;                           %sample size
s = 20;                           %sparsity
maxit = 20;                      %max no. of iteration
samratio = 0.6;
size = [m,n];
xtrue = zeros(n,1);
order = randperm(n);
for i=1:s
  xtrue(order(i))=randn; %real x
end  
xtrue=xtrue/norm(xtrue,'fro');         
A = randn(m,n);%
sigma = 0;
b =abs(A*xtrue) + sigma*randn(m,1);
tol = 1e-10;

tic;
xini = Compute_x_init(b,A,s);
[x_rec,error,Err,iter] = sparse_stochasticADM(xini,xtrue,s,A,b,samratio,maxit,tol);
toc;
%%
figure(1)
semilogy(1:iter,error,'+-r','linewidth',2) ;
ylabel('Relative error in recovery','Interpreter','Latex','fontsize',20)
xlabel('Iteration $$k$$','Interpreter','Latex','fontsize',20);
str=sprintf( 'SAM: m=%d, n=%d, s=%g',m,n,s);
title(str,'Interpreter','Latex','fontsize',20)