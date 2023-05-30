function [x0] = Compute_x_init(y_abs,A,s)
%updated 5/31/2017

%% initialize parameters
[m, n] = size(A);
%If ground truth is unknown
if nargin < 7
    z = zeros(n,1);
end
Marg = zeros(1,n); %marginals
MShat = zeros(s); %truncated correlation matrix
AShat = zeros(m,s); %truncated sensing matrix
y_abs2 = y_abs.^2;
phi_sq = sum(y_abs2)/m;
phi = sqrt(phi_sq); %signal power

%% s-Truncated sensing vectors
%signal marginals
Marg = ((y_abs2)'*(A.^2))'/m; % n x 1
[~, MgS] = sort(Marg,'descend');
S0 = MgS(1:s); %pick top s-marginals
Shat = sort(S0); %store indices in sorted order
%supp(Shat) = 1; figure; plot(supp); %support indicator
AShat = A(:,Shat); % m x s %sensing sub-matrix

%% Initialize x
%compute top singular vector according to thresholded sensing vectors and large measurements
for i = 1:m
   MShat = MShat + (y_abs2(i))*AShat(i,:)'*AShat(i,:); % (s x s)
end

%svd_opt = 'svd'; %more accurate, but slower for larger dimensions
svd_opt = 'power'; %approximate, faster

switch svd_opt
    case 'svd'
        [u,~,~] = svd(MShat);
        v1 = u(:,1); %top singular vector of MShat, normalized - s x 1
    case 'power'
        v1 = svd_power(MShat);
end
v = zeros(n,1);
v(Shat,1) = v1;
x_init = phi*v; %ensures that the energy/norm of the initial estimate is close to actual
x0 = x_init;
end
%% svd_power
function max_sv = svd_power(M)
[m, n] = size(M);
x = randn(n,1);
for it = 1:20
   y = M*x;
   y = y/norm(y);
   if norm(x-y)/norm(x) < 1e-6
        break;
   end
   x = y;
end
   max_sv = x; 
end