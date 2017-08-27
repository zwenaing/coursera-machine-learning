function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of theta values

% You need to return the following variables correctly 

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% computing the logistic regression function
hypothesis = 1 ./ (1 + exp(-(X * theta))); 

% computing the vectorized model of regularized logistic cost function

beforeReg = (((-y') * log(hypothesis)) - ((1 - y)' * log (1-hypothesis))) / m;
theta(1) = 0;
J = beforeReg + ((lambda / (2 * m)) * sum(theta .^ 2));

for i = 1 : m
    grad(1) = grad(1) + ((hypothesis(i) - y(i)) * X(i,1));
end
grad(1) = grad(1) / m;

for j = 2 : n
    for k = 1 : m
        difference = hypothesis(k) - y(k);
        grad(j) = grad(j) + (difference * X(k,j));
    end
    grad(j) = (grad(j) / m) + ((lambda / m) * theta(j));
end


% =============================================================

end
