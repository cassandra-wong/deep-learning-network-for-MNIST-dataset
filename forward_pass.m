 function [z1,h1,z2,h2,z3,y_i] = forward_pass(W1,W2,W3,b1,b2,b3,x_i)
    z1=(W1*x_i)+b1;
    h1=phi_ReLU(z1);
    z2=(W2*h1)+b2;
    h2=phi_ReLU(z2);
    z3=(W3*h2)+b3;
    y_i=phi_Softmax(z3);
end
