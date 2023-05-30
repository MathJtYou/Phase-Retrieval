function [x_rec,error,Err,iter] = sparse_stochasticADM(xini,xtrue,K,A,b,samratio,maxit,tol)
k=0;
[n,p]=size(A);
change = 1;
error=zeros;
Err=zeros;
y2=(A*xtrue).^2;
%[A,d] = normalize(A); 
while k <=maxit && change >= tol
   k = k+1;
   xini_old=xini;
   id = randperm(n);
   idx = id(1:floor(samratio*n));
   Aidx = A(idx,:);
   bidx = b(idx);
   sgnidx = sign(Aidx*xini);
   bidx = sgnidx.*bidx;
   opts.T = K;
   opts.MaxIter = 5;
   opts.xini = xini_old;
   [xini,~,~]=HTPmu(Aidx,bidx,xtrue,xini_old,0.95,K,5);
    change = min(norm(xini-xtrue),norm(xini+xtrue))/norm(xtrue);
    x0 = real(xini);
    
    errhis=log(norm(b-abs(A*x0)));
    Err(k)=errhis;
    errorhis=min(norm(x0-xtrue),norm(xtrue+x0))/norm(x0);
    error(k)=errorhis;
   
end
x_rec=real(x0);
iter=k;
end

function [x_rd,errhis,iter]=HTPmu(A,y,x_true,x0,mu,s,maxit)

[m,n]=size(A);
errhis=zeros(maxit+1,1);
it=0;
stop=0;
tol=0;
 
while ~stop
 it=it+1;
   err = min(norm(x0-x_true),norm(x0+x_true))/norm(x_true,'fro');
   Ax = A*x0;
    grad_i=A'*(y-Ax)/m;
    x1=x0+mu*grad_i;
    [~,num]=sort(abs(x1),'descend');
    S=num(1:s);
    As=A(:,S);
    Y=As'*As;
    z1=zeros(n,1);
    z1(S)=Y\(As'*y);
   x0=z1;
 %fprintf( 'Intermediate Soln:Error of recovered signal is %3.3e \n', err);
    if it>=maxit ||err<tol
         stop =1;
    end
   if it>=maxit
        stop =1;
   end
end
x_rd=x0;
iter=it;
end