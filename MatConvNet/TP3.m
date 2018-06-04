clc;
clear  all;
close  all;
% Initialisation  de  MatConvNet  sous  MATLAB (� faire � chaque nouvelle  session)
run /home/yanis/Bureau/matconvnet-1.0-beta23/matlab/vl_setupnn;
% Chargement  du mod�le et  conversion  vers la  version  actuelle de  MatConvNet
net = load('imagenet-vgg-f.mat');
net = vl_simplenn_tidy(net);
% Visualisation  des  informations  sur les  couches  du r�seau
vl_simplenn_display(net ,'inputSize', [224  224 3 1])
load('15SceneSplit.mat');
Xtrain = zeros(size(X_train_path,1),4096);
Xtest = zeros(size(X_test_path,1),4096);
% Charger et pr�parer les images de train
 for i = 1:size(X_train_path,1)
     
     
 imDir = strjoin(X_train_path(i,1));
 I = imread(imDir);
 im = repmat(I, 1, 1, 3);
 im_ = single(im); % Note : valeurs  entre 0 et 255
 im_ = imresize(im_ , net.meta.normalization.imageSize (1:2));
 im_ = im_ - net.meta.normalization.averageImage;
 
 % Calculer  la pr�diction  du mod�le
 res = vl_simplenn(net , im_);
 %R�cup�ration de la sortie de la couche ReLu 7 + norm L2 8
 SortieRelu7 = res(20).x;
 SortieRelu7 = SortieRelu7(:);
 
 SortieRelu7Norm = SortieRelu7./norm(SortieRelu7.');
 
 
 Xtrain(i,:)=SortieRelu7Norm;
 
 end
 
 % Charger et pr�parer les images de test
 for i = 1:size(X_test_path,1)
     
     
 imDir = strjoin(X_test_path(i,1));
 I = imread(imDir);
 im = repmat(I, 1, 1, 3);
 im_ = single(im); % Note : valeurs  entre 0 et 255
 im_ = imresize(im_ , net.meta.normalization.imageSize (1:2));
 im_ = im_ - net.meta.normalization.averageImage;
 
 % Calculer  la pr�diction  du mod�le
 res = vl_simplenn(net , im_);
 %R�cup�ration de la sortie de la couche ReLu 7 + norm L2 8
 SortieRelu7 = res(20).x;
 SortieRelu7 = SortieRelu7(:);
 
 
 SortieRelu7Norm = SortieRelu7./norm(SortieRelu7.');
 Xtest(i,:)=SortieRelu7Norm;
 end
save('Xtest.mat','Xtest');
save('Xtrain.mat','Xtrain');