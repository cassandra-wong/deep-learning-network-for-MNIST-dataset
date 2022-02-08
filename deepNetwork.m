% initialize parameters
maxit=300000;
alpha=0.01;
n=60000;
n_test=10000;
out_interval=1000;

% load the training and test images + labels
trainimages=loadMNISTImagesAsVectors('fashion_train-images.idx3-ubyte');
testimages=loadMNISTImagesAsVectors('fashion_t10k-images.idx3-ubyte');
trainlabels=loadMNISTLabels('fashion_train-labels.idx1-ubyte');
testlabels=loadMNISTLabels('fashion_t10k-labels.idx1-ubyte');

% generate 2 arrays of one-hot label vectors
train_y_actual=(trainlabels==1:10)';
test_y_actual=(testlabels==1:10)';

% randomly initialize W's and b's
W1=randn(128,784)*0.3;
W2=randn(32,128)*0.3;
W3=randn(10,32)*0.3;
b1=randn(128,1)*0.3;
b2=randn(32,1)*0.3;
b3=rand(10,1)*0.3;

% before SGD training
before_accuracy=0;
for i=1:n
    h1=phi_ReLU(W1*(trainimages(:,i))+b1);
    h2=phi_ReLU(W2*h1+b2);
    yhat=phi_Softmax(W3*h2+b3);             % probability vector
    yhat_label=find(yhat==max(yhat));       % find label
    % loss and accuracy 
    if yhat_label==trainlabels(i)
        before_accuracy = before_accuracy+1;
    end
    Fi_before(i)=norm(yhat-train_y_actual(:,i))^2;
end

F(1)=sum(Fi_before)/n;
percent_accurate(1)=(before_accuracy)/n;
fprintf(['Before SGD Training \nLoss: ' num2str(F(1)) '\nAccuracy: '...
    num2str(percent_accurate(1)) '\n\n'])

% SGD training
SGD_accuracy=0;
for it=1:maxit
    i=randi(n);
    x_i=trainimages(:,i);
    % forward pass for image i
    [z1,h1,z2,h2,z3,y_i]=forward_pass(W1,W2,W3,b1,b2,b3,x_i);
    y_i_label=find(y_i==max(y_i));
    if y_i_label==trainlabels(i)
        SGD_accuracy = SGD_accuracy+1;
    end
    train_y_actual_i=train_y_actual(:,i);
    
    % back-propagate to compute gradients of f_i
    [grad_W1_f,grad_W2_f,grad_W3_f,grad_b1_f,grad_b2_f,grad_b3_f]=...
        back_prop(W1,W2,W3,z1,h1,z2,h2,z3,y_i,train_y_actual_i,x_i);
    
    % SGD update of the weights
    b3=b3-(alpha*grad_b3_f);
    W3=W3-(alpha*grad_W3_f);
    
    b2=b2-(alpha*grad_b2_f);
    W2=W2-(alpha*grad_W2_f);
    
    b1=b1-(alpha*grad_b1_f);
    W1=W1-(alpha*grad_W1_f);
    
    Fi_SGD(it)=(norm(y_i-train_y_actual_i))^2;
    if mod(it,out_interval)==0
        iteration=it/out_interval;
        F(iteration+1)=sum(Fi_SGD(it))/it;
        percent_accurate(iteration+1)=(SGD_accuracy)/it;
        fprintf(['During SGD Training \nIteration: ' num2str(it) '\nLoss: '...
            num2str(F(iteration+1)) '\nAccuracy: '...
            num2str(percent_accurate(iteration+1)) '\n\n'])
    end
end

% plot F as a function of iterations
subplot(211);
x=[0:out_interval:it];
plot(x,F)
xlabel('Iteration')
ylabel('Loss')

title('Training Progress: Loss')

% plot percent_accurate as a function of iterations
subplot(212);
plot(x,(percent_accurate*100))
xlabel('Iteration')
ylabel('Accuracy (%)')
title('Training Progress: Accuracy')

% test set
test_accuracy=0;
for j=1:n_test
    x_j=testimages(:,j);
    % forward pass for image j
    [z1,h1,z2,h2,z3,y_j]=forward_pass(W1,W2,W3,b1,b2,b3,x_j);
    y_j_label=find(y_j==max(y_j));
    if y_j_label==testlabels(j)
        test_accuracy = test_accuracy+1;
    end
    test_y_actual_j=test_y_actual(:,j);
    
    Fi_test(j)=(norm(y_j-test_y_actual_j))^2;
    if j==n_test
        F_test=sum(Fi_test(j))/j;
        percent_accurate_test=(test_accuracy)/j;
        fprintf(['Test Iteration: ' num2str(j) '\nTest Loss: '...
            num2str(F_test) '\nTest Accuracy: '...
            num2str(percent_accurate_test) '\n\n'])
    end
end