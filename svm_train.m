function [ w ,obj] = svm_train( X ,Y, Param )
%SVM Summary of this function goes here
%   logistic regression loss function with L2 norm regularization.
% function:
% \min 0.5 * w^Tw + C \sum_i=1^l log(1+e^(-y_iw^Tx_i))
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
mom = zeros(d,1);
beta = 0.5;
alpha = repmat(w,[1,n]);
while epoch < max_epochs
    index_permutation = randperm(n);
    for iter = 1:n
        index = index_permutation(iter);
        if strcmp(opt_method,'sgd')
			gradient = compute_grad(w,X, Y, C index)
            w = w - lr * gradient;            
        elseif strcmp(opt_method,'sgd+mom')
            if beta < 0.9
                beta = beta + lr;
            end
			gradient = compute_grad(w,X, Y, C index)
            mom = beta * mom + lr * gradient;
            w = w - mom;
        elseif strcmp(opt_method,'svrg')
            % full gradient compute
            if iter == 1
                hat_w = w;
                gradient_full = zeros(d,1);
               for inner_iter = 1:n
            		gradient = w + C * (-Y(index)/(1+exp(Y(index)*w'*X(:,index))))*X(:,iner_iter);
                   gradient_full = gradient_full + gradient;
               end
               gradient_full = gradient_full / n;
            end
            
			gradient = compute_grad(w,X, Y, C index)
			hat_gradient = compute_grad(hat_w, X, Y, C index)
            w = w - lr * (gradient - hat_gradient  + gradient_full);
        elseif strcmp(opt_method,'saga')
            if epoch == 1 && iter == 1
                for i_num  = 1:n
					gradient = compute_grad(w,X, Y, C i_num)
                    alpha(:,i_num) = gradient;
                end
            end
			gradient = compute_grad(w,X, Y, C index)
            w = w - lr * (gradient - alpha(:,index) + sum(alpha,2)/n);
            alpha(:,index) = gradient;
       
        elseif strcmp(opt_method,'eNsaga')
           if epoch == 1 && iter == 1
                graph = compute_neighbor(X,Param.neighbor);
           end
           
            neighbors = find(graph(:,index)==1);

			gradient = compute_grad(w,X, Y, C index)
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
			gradient = compute_grad(w,X, Y, C index)
            mom_1 = beta1 * mom_1 + (1-beta1)*gradient;
            mom_2 = beta2 * mom_2 + (1-beta2)*gradient.^2;
            mom_1_bias = mom_1 / (1-beta1^iter);
            mom_2_bias = mom_2 / (1-beta2^iter);
            w = w - lr * mom_1_bias ./ (mom_2_bias.^0.5 + epsilon);                        
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
obj = 0.5 * w'*w;
obj = obj + C*sum(log(1+exp(-Y'.*(w'*X))));
end


function gradient = compute_grad(w,X, Y, C index)
	tmp = -Y(index)*w'*X(:,index);
	gradient = w + C * (-Y(index)*exp(tmp)/(1+exp(tmp)))*X(:,index);
end
