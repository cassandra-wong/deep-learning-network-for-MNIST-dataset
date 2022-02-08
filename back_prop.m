function [grad_W1_f,grad_W2_f,grad_W3_f,grad_b1_f,grad_b2_f,grad_b3_f] = back_prop(W1,W2,W3,z1,h1,z2,h2,z3,y_i,train_y_actual_i,x_i)
    grad_b3_f=jac_Softmax(y_i)*(-2*(train_y_actual_i-y_i));
    grad_W3_f=grad_b3_f*h2';
    grad_b2_f=jac_ReLU(z2)*W3'*grad_b3_f;
    grad_W2_f=grad_b2_f*h1';
    grad_b1_f=jac_ReLU(z1)*W2'*grad_b2_f;
    grad_W1_f=grad_b1_f*x_i';
end
