function [ Yhat, out ] = forward( net, X )
%Calcule de la sortie du réseau 

k = size(X(1,:),2);

X = zeros(k,nx); %k = nombre d'entrées x
Yhat = zeros(k,ny);

out = struct;
out.htild = X*Wh.'+bh;
out.h = tanh(htild);
out.ytild = h*Wy.'+by;
out.Yhat = softmax(ytild);



end

