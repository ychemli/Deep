function net = backward(net, out, Y, eta)

dl_by=zeros(size(Yhat(1,:),2));
dl_bh=zeros(size(h));
dl_Wy=zeros(size(Yhat(1,:),2),size(h));
dl_Wh=zeros(size(h),size(X(1,:),2));


Y_c= Yhat(:,i);
% NbBatch = 10;

for i=1:N
   
    
        dl_by(i)=(out.Yhat(i,:)-Y(i,:))';
        dl_bh(i)=((net.Wy(j,i))'*(out.Yhat(:,j)-Y(:,j))).*(1-tanh(out.htild(i,:)^2));
        dl_Wy(i,j)=dl_by*out.h(:,j)';
        dl_Wh(i,j)=dl_bh*X(j,:)';
  
    
end

