function J_Softmax = jac_Softmax(y)
% Usage: computes the Jacobian matrix of the Softmax
% function evaluated in weighted input vector z, but we give
% the Softmax output y as input to the jac_Softmax(y) function
% because the Jacobian formulas can easily be expressed
% as a function of y; this matrix is not diagonal but it is symmetric % (should work for input y of any size)

J_Softmax = diag(y);

for i=1:length(J_Softmax)
    for k=1:size(J_Softmax)
        if i == k
            J_Softmax(i,k) = y(i)-(y(i)^2);
        else
            J_Softmax(i,k) = -y(i)*y(k);
        end
    end
end

end