clc;
clear  all;
close  all;
% Initialisation  de  MatConvNet  sous  MATLAB (à faire à chaque nouvelle  session)
run /home/yanis/Bureau/matconvnet-1.0-beta23/matlab/vl_setupnn;
% Chargement  du modèle et  conversion  vers la  version  actuelle de  MatConvNet
net = load('imagenet-vgg-f.mat');
net = vl_simplenn_tidy(net);
% Visualisation  des  informations  sur les  couches  du réseau
vl_simplenn_display(net ,'inputSize', [224  224 3 1])

% Charger  et préparer l'image
 load('15SceneSplit.mat');
 I = imread(X_train_path(1));
 imshow(I);
im = imread('cat2.jpg');
im_ = single(im); % Note : valeurs  entre 0 et 255
im_ = imresize(im_ , net.meta.normalization.imageSize (1:2));
im_ = im_ - net.meta.normalization.averageImage;
% Calculer  la prédiction  du modèle
res = vl_simplenn(net , im_);

% Afficher  la  classe  prédite
scores = squeeze(gather(res(end).x));
[bestScore , best] = max(scores);
figure (1); clf; imagesc(im);
title(sprintf('%s (%d), score  %.3f',...
net.meta.classes.description{best}, best , bestScore));

for i = 1:64
    I = res(8).x(:,:,i);
    imagesc(I);
    waitforbuttonpress
    imwrite(I,'image','jpg');
end