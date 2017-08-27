function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% computing the logistic regression function
hypothesis = sigmoid(X * theta);
%hypothesis = 1 ./ (1 + exp(-(X * theta))); 

% computing the vectorized model of logistic cost function
J = (((-y') * log(hypothesis)) - ((1 - y)' * log (1-hypothesis))) * (1 / m);

grad = zeros(size(theta));

% computing different partial derivatives for different features of the
% cost function
 
for i = 1 : m
    difference = hypothesis(i) - y(i);
    grad(1) = grad(1) + difference * X(i,1);
    grad(2) = grad(2) + difference * X(i,2);
    grad(3) = grad(3) + difference * X(i,3);
end
grad(1) = grad(1) * (1 / m);
grad(2) = grad(2) * (1 / m);
grad(3) = grad(3) * (1 / m);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% =============================================================

end
