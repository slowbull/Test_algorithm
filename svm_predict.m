function [ y ] = svm_predict( X,w )
%SVM_PREDICT Summary of this function goes here
%   Detailed explanation goes here

fx = w'*X;
y = ones(length(fx),1);
y(fx<0) = -1;

end

