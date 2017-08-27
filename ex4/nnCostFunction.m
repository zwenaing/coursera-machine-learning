function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

bias = ones(m, 1); % 5000 x 1
X = [bias X]; % 5000 X 401
a2 = sigmoid(X * Theta1'); % 5000 x 25
a2 = [bias a2]; % 5000 x 26
hypothesis = sigmoid(a2 * Theta2'); % 5000 x 10

double_sum = 0.0;

for i = 1 : m
    vecY = zeros(num_labels,1);
    vecY(y(i)) = 1;
    for k = 1 : num_labels
        double_sum = double_sum + ((-vecY(k) * log(hypothesis(i,k))) - ((1 - vecY(k)) * log(1 - hypothesis(i,k))));
    end
end

J = double_sum / m;

sum1 = 0.0;
for j = 1 : hidden_layer_size
    for k = 2 : input_layer_size + 1
        sum1 = sum1 + Theta1(j,k)^2;
    end
end

sum2 = 0.0;
for j = 1 : num_labels
    for k = 2 : hidden_layer_size + 1
        sum2 = sum2 + Theta2(j,k)^2;
    end
end

J = J + ((lambda * (sum1 + sum2)) / (2 * m));

% -------------------------------------------------------------

vecY = zeros(m, num_labels); % 5000 x 10
for t = 1 : m
    vecY(t,y(t)) = 1;
end

delta3 = hypothesis - vecY; % 5000 x 10
z2 = X * Theta1'; % 5000 x 25
delta2 = delta3  * Theta2 .* [ones(m,1) sigmoidGradient(z2)]; % 5000 x 26

for t = 1 : m
    Theta1_grad = Theta1_grad + delta2(t,2:end)' * X(t,:);
    Theta2_grad = Theta2_grad + delta3(t,:)' * a2(t,:);
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
