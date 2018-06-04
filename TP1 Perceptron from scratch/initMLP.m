function [ net ] = initMLP( nx, nh, ny )
% Cette fonction initialise un réseau perceptron à partir des tailles nx nh
% ny

Wh = zeros(nh,nx);
Wy = zeros(ny,nh);

bh = zeros(nh);
by = zeros(ny);

sigma = 0.3;

net = struct;
net.Wh = randn(nh,nx)*sigma;
net.Wy = randn(ny,nh)*sigma;
net.bh = randn(nh,1)*sigma;
net.by = randn(ny,1)*sigma;

end

