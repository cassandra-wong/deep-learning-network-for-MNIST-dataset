function J_ReLU = jac_ReLU(z)
% Usage: computes the Jacobian matrix of the ReLU activation
% function evaluated in weighted input vector z;
% this is a diagonal matrix (should work for input z of any size)

J = [];
for i = 1:length(z)
    if z(i)>=0
        J = [J, 1];
    else
        J = [J, 0];
    end
end

J_ReLU = diag(J);
end