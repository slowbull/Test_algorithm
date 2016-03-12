% function [ acc ] = test()
%TEST Summary of this function goes here
%   Detailed explanation goes here
% Param.opt  'sgd', 'sgd+mom'

% load('usps2');

% create Data.
d = 100;
n = 1000;
X = rand(d,n);
Y = rand(n,1);
w_opt = inv(X*X'+eye(d))*X*Y;
obj_opt = sum((w_opt'*X - Y').^2) + 0.5*(w_opt'*w_opt);

max_epochs = 50;

%% sgd
Param.max_epochs = max_epochs;
Param.opt = 'sgd';
Param.C = 1e0; 
Param.lr = 1e-1;
Param.step = 1;
Param.eps = 0.01;
Param.neighbor = 50

[w,obj1] = LS_train(X,Y,Param);

%% sgd+mom
Param.max_epochs = max_epochs;
Param.opt = 'sgd+mom';
Param.C = 1e0; 
Param.lr = 1e-1;
Param.step = 1;
Param.eps = 0.01;
Param.neighbor = 50

[w,obj2] = LS_train(X,Y,Param);


%% svrg
Param.max_epochs = max_epochs;
Param.opt = 'svrg';
Param.C = 1e0; 
Param.lr = 5e-5;
Param.step = 0;
Param.eps = 0.01;
Param.neighbor = 50

[w,obj3] = LS_train(X,Y,Param);



%%
err1 = abs(obj1-obj_opt);
err2 = abs(obj2-obj_opt);
err3 = abs(obj3-obj_opt);
len = length(err1);

epoch = [1:len];

figure;
plot(epoch,err1,epoch,err2,epoch,err3);
% end

                    
