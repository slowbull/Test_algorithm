function [ graph ] = compute_neighbor(X, num_neighbors)
%COMPUTE_NEIGHBOR Summary of this function goes here
%   Detailed explanation goes here


n = size(X,2);
dis = zeros(n,n);
graph = zeros(n,n);
for row = 1:n
    for col = 1:n
        dis(row,col) = norm(X(:,row) - X(:,col));
    end
end

for node = 1:n
   [~,idx] = sort(dis(:,node));
   graph(node,idx(1:num_neighbors)) = 1;
end

end

