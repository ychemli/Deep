clc;
clear all;
close all;

%Init matconvnet
run /usr/local/matconvnet-1.0-beta23/matlab/vl_setupnn;

load('15SceneSplit.mat');

nTrain = size(X_train_path,1);
nTest = size(X_test_path,1);
nImages = nTrain + nTest;
imdb.images.data = zeros(64, 64, 1, nImages , 'single');
imdb.images.labels = zeros(1, nTrain , 'single');
imdb.images.set = zeros(1, nImages ,'uint8');  % train:1 / val:2

for i=1:nImages
    if i <= nTrain
        im_path = X_train_path{i,1};
        imdb.images.set(i) = 1;
        imdb.images.labels(i) = y_train(i,1);
    else
        im_path = X_test_path{i-nTrain,1};
        imdb.images.set(i) = 2;
        imdb.images.labels(i) = y_test(i-nTrain,1);
    end;
    im = imread(im_path);
    im = single(im);
    im = imresize(im, [64 64]);
    imdb.images.data(:,:,1,i) = im;
end

net.layers = {};
% conv1 layer
net.layers{end+1} = struct ( 'name', 'conv1', ...
                             'type', 'conv', ...
                             'weights', {{0.01*randn(9, 9, 1 ,10 ,'single'), zeros(1,10,'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);
% pool1 layer                      
net.layers{end+1} = struct ( 'name', 'pool1', ...
                             'type', 'pool', ...
                             'method', 'max', ...
                             'pool', [7 7], ...
                             'stride', 7, ...
                             'pad', 0); 
% relu1 layer                      
net.layers{end+1} = struct ( 'name', 'relu1', ...
                             'type', 'relu');    
% fc2 layer
net.layers{end+1} = struct ( 'name', 'fc2', ...
                             'type', 'conv', ...
                             'weights', {{0.01*randn(8, 8, 10 ,15 ,'single'), zeros(1,15,'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);

% softmaxloss layer                         
net.layers{end+1} = struct('type','softmaxloss');

net = vl_simplenn_tidy(net);

vl_simplenn_display(net,'inputSize', [64  64 1 1]);

opts.batchSize = 50;
opts.learningRate = 0.0001;
opts.numEpochs = 500;
opts.expDir = 'exp/';

[net, info] = cnn_train(net, imdb , @getBatch , ...
    opts , ...
    'val', find(imdb.images.set == 2));
                         

'ok'
