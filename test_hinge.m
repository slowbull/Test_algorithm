% function [ acc ] = test()
%TEST Summary of this function goes here
%   Detailed explanation goes here
% Param.opt  'sgd', 'sgd+mom'

load('usps2');
Param.max_epochs = 100;
Param.opt = 'acc_svrg';
Param.C = 1e2; 
Param.lr = 1e-5;
Param.step = 0;
Param.eps = 0.01;
Param.neighbor = 50
Param.asvrg_t1 = 0.1;
Param.asvrg_t2 = 0.5;

train_X = [train_X,ones(length(train_Y),1)];
test_X = [test_X,ones(length(test_Y),1)];

[w,obj] = svm_train_hinge(train_X',train_Y,Param);
pred_Y = svm_predict(test_X',w);

acc = sum(pred_Y==test_Y)/length(test_Y)
% end

                    
