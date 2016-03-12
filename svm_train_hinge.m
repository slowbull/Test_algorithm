function [ w ,obj] = svm_train_hinge( X ,Y, Param )
%SVM Summary of this function goes here
%   hinge loss function with L2 norm regularization.
% function:
% \min 0.5 * w^Tw + C \sum_i=1^l max(1-y_iw^Tx_i,0)
% X \in d * n 
% Y \in n * 1
% w \in d * 1

max_epochs =  Param.max_epochs;
opt_method = Param.opt;
C = Param.C;
initial_lr = Param.lr;
lr = initial_lr;
step = Param.step;

epoch = 1;
[d,n] = size(X);
w = rand(d,1);

dual_w = rand(n,1);
dual_w_old = dual_w;
if strcmp(opt_method,'sdca')
  w = sum(repmat(Y',[d,1]).*repmat(dual_w',[d,1]).*X,2);
end

mom = zeros(d,1);
beta = 0.5;
alpha = repmat(w,[1,n]);
while epoch < max_epochs
    index_permutation = randperm(n);
    for iter = 1:n
        index = index_permutation(iter);
        if strcmp(opt_method,'sgd')
            gradient = compute_gradient(w,X(:,index),Y(index),C);
            w = w - lr * gradient;            
        elseif strcmp(opt_method,'sgd+mom')
            if beta < 0.9
                beta = beta + lr;
            end

            gradient = compute_gradient(w,X(:,index),Y(index),C);
            mom = beta * mom + lr * gradient;
            w = w - mom;
        elseif strcmp(opt_method,'svrg')
            % full gradient compute
            if iter == 1
                hat_w = w;
                gradient_full = zeros(d,1);
               for inner_iter = 1:n
                   gradient = compute_gradient(w,X(:,inner_iter),Y(inner_iter),C);
                   gradient_full = gradient_full + gradient;
               end
               gradient_full = gradient_full / n;
            end
            
            gradient = compute_gradient(w,X(:,index),Y(index),C);
            hat_gradient = compute_gradient(hat_w,X(:,index),Y(index),C);
            w = w - lr * (gradient - hat_gradient  + gradient_full);
        elseif strcmp(opt_method,'saga')
            if epoch == 1 && iter == 1
                for i_num  = 1:n
                    gradient = compute_gradient(w,X(:,index),Y(index),C);
                    alpha(:,i_num) = gradient;
                end
            end
            gradient = compute_gradient(w,X(:,index),Y(index),C);
            w = w - lr * (gradient - alpha(:,index) + sum(alpha,2)/n);
            alpha(:,index) = gradient;
       
        elseif strcmp(opt_method,'eNsaga') % to be completed
           if epoch == 1 && iter == 1
                graph = compute_neighbor(X,Param.neighbor);
           end
           
            neighbors = find(graph(:,index)==1);

            gradient = w + C * (-Y(index)/(1+exp(Y(index)*w'*X(:,index))))*X(:,index);
            w = w - lr * (gradient - alpha(:,index) + sum(alpha,2)/n);

           for i_neigh = 1:length(neighbors)
                eij = (exp(norm(X(:,index)-X(:,i_neigh))*norm(w)) - 1)/(1+exp(-w'*X(:,index)))*norm(X(:,i_neigh));
                if eij <= Param.eps
                    alpha(:,i_neigh) = gradient;
                else
                    tmp = w + C * (-Y(i_neigh)/(1+exp(Y(i_neigh)*w'*X(:,i_neigh))))*X(:,i_neigh);
                    alpha(:,i_neigh) = tmp;
                end
           end
        elseif strcmp(opt_method,'adamsgd')
            if epoch == 1
                mom_1 = zeros(d,1);
                mom_2 = zeros(d,1);
                beta1 = 0.9;
                beta2 = 0.999;
                epsilon = 1e-8;
            end
            gradient = compute_gradient(w,X(:,index),Y(index),C);
            mom_1 = beta1 * mom_1 + (1-beta1)*gradient;
            mom_2 = beta2 * mom_2 + (1-beta2)*gradient.^2;
            mom_1_bias = mom_1 / (1-beta1^iter);
            mom_2_bias = mom_2 / (1-beta2^iter);
            w = w - lr * mom_1_bias ./ (mom_2_bias.^0.5 + epsilon);                        

        elseif strcmp(opt_method,'sdca')
            G = Y(index)*w'*X(:,index) - 1;
            if dual_w(index) == 0 
                PG = min(G,0);
            elseif dual_w(index) == C
                PG = max(G,0);
            else
                PG = G;
            end
            
            if PG ~= 0 
                QII = Y(index)*Y(index)*X(:,index)'*X(:,index);
                dual_w_old(index) = dual_w(index);
                dual_w(index) = min(max(dual_w(index)-G/QII,0),C);
                w = w + (dual_w(index)-dual_w_old(index))*Y(index)*X(:,index);
            end

            if iter==1
                tmp = (compute_obj(w,X,Y,C) - compute_dual_obj(dual_w,X,Y,C))
                if tmp < 1e-3  
                   epoch = intmax;
                   break;
                end
            end
        end    
        
        if step ~= 0
            lr = initial_lr/(1+iter+n*(epoch-1))^step;
        end
    end
    obj(epoch) = compute_obj(w,X,Y,C);
    fprintf('epoch: %d, obj: %e\n',epoch, obj(epoch));
epoch = epoch +1;
end
end

function obj = compute_obj(w,X,Y,C)
obj = 0.5 * (w'*w);
n = length(Y);
obj = obj + C * sum(max(0, ones(1,n)-Y'.*(w'*X)));
end

function obj = compute_dual_obj(dual_w,X,Y,C)
Q = (Y*Y').*(X'*X);
obj = -0.5*dual_w'*Q*dual_w + sum(dual_w); 
end

function gradient = compute_gradient(w,X,Y,C)
tmp = 1 - Y*w'*X;
if tmp <= 0
    gradient = w;
else
    gradient = w - C * Y*X;
end
end
