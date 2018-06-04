function TP1()
    data = load('circles.mat')
    
    nx = 2; %
    nh = 6; %
    ny = 2; %
    net = initMLP (nx, nh, ny )
    
    Niter = 100; %
    eta = 0.1; %
    
    for i=1:Niter
        %train
        [ Yhat_train, out ] = forward (net, data.Xtrain');  
        net  = backward(net, out, data.Ytrain', eta);
        [ loss_train, acc_train ] = loss_accuracy (Yhat_train', data.Ytrain);
        
        %test
        [ Yhat_test, ~ ] = forward (net, data.Xtest');
        [ loss_test, acc_test ] = loss_accuracy (Yhat_test', data.Ytest);
    end
    show(net,data);
    
end

function [ net ] = initMLP (nx, nh, ny )
    net = struct;
    net.Wh = normrnd(0, 0.3, nh, nx);
    net.bh = randn(nh,1);
    net.Wy = normrnd(0, 0.3, ny, nh);
    net.by = randn(ny,1);       
end

function [ Yhat, out ] = forward (net,X )
    out.X = X;
    n = size(X,2);
    out.Htilde = net.Wh * X + repmat(net.bh,1,n);
    out.H=tanh(out.Htilde);
    out.Ytilde = net.Wy * out.H + repmat(net.by,1,n);
    out.Yhat = softmax(out.Ytilde); 
    Yhat = out.Yhat;
end


function [ L, acc ] = loss_accuracy (Yhat, Y)
    n = size(Y,1);  % no examples
    k = size(Y,2);  % no classes

    L = zeros(n,1);
    acc = 0;
    for i=1:n
        l = 0;
        max = 0;
        maxi = 0;
        for j=1:k
            l = l - Y(i,j) * log(Yhat(i,j));    
            if Yhat(i,j) > max
                max = Yhat(i,j);    % max prediction score
                maxi = j;           % predicted class = j
            end
        end
        L(i) = L(i) + l;
        if Y(i,maxi) == 1 % predicted class == real class
            acc = acc + 1;
        end        
    end
    acc = acc/n;
    L_sum = sum(L);
end

function [ net ] = backward(net, out, Y, eta)
    n = size(Y,2);
    gradBy = 0;
    gradWy = 0;
    gradBh = 0;
    gradWh = 0;
    for i=1:n
        y = Y(:,i);
        x = out.X(:,i);
        htilde = out.Htilde(:,i);
        h = out.H(:,i);
        ytilde = out.Ytilde(:,i);
        yhat = out.Yhat(:,i);
        
        gradYtilde = ytilde - y;

        gradBy = gradBy + gradYtilde;
        gradWy = gradWy + gradYtilde * h';

        gradHtilde = diag(1-h.^2) * net.Wy' * gradYtilde;
        
        gradBh = gradBh + gradHtilde;
        gradWh = gradWh + gradHtilde * x';
    end
    net.Wh = net.Wh - eta/n * gradWh;
    net.bh = net.bh - eta/n * gradBh;
    net.Wy = net.Wy - eta/n * gradWy;
    net.by = net.by - eta/n * gradBy;
end

function show(net, data)
%     [x1 x2] = meshgrid(-2:0.1:2);
%     x1 = reshape(x1,size(x1,1)^2,1);
%     x2 = reshape(x2,size(x2,1)^2,1);
%     Xgrid = [x1 x2];
%        
%     [Ygrid , ~] = forward(net, Xgrid');
%     grid = Ygrid(:,1);
%     l = sqrt(length(Ygrid));
%     %imagesc(reshape(Ygrid , l, l));
%     caxis([0.3,  0.7]);

    Xtrain = data.Xtrain;
    Ytrain = data.Ytrain;
    [Ytrain_hat , ~] = forward(net, Xtrain');
    Ytrain_hat = Ytrain_hat';

    figure()
    plot((Xtrain(Ytrain(:,1) >0.5,1) + 2) * 10, ...
         (Xtrain(Ytrain(:,1) >0.5,2) + 2) * 10, 'o', 'color', 'blue') %first class
    hold on
    plot((Xtrain(Ytrain(:,2) >0.5,1) + 2) * 10, ...
         (Xtrain(Ytrain(:,2) >0.5,2) + 2) * 10, 'o', 'color', 'red') %second class
    title('Vérité terrain')
     
    figure()
    plot((Xtrain(Ytrain_hat(:,1) >0.5,1) + 2) * 10, ...
         (Xtrain(Ytrain_hat(:,1) >0.5,2) + 2) * 10, 'o', 'color', 'blue') %first class
    hold on
    plot((Xtrain(Ytrain_hat(:,2) >0.5,1) + 2) * 10, ...
         (Xtrain(Ytrain_hat(:,2) >0.5,2) + 2) * 10, 'o', 'color', 'red') %second class
    title('Prediction')

end