function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calculate hypothesis
hypo = sigmoid(X * theta);

% Calculate regularization term for cost function
reg_term1 = theta(2:end)' * theta(2:end) * (0.5 * lambda / m);

% Calculate cost function using hypothesis calculated above
J = (y' * log(hypo) + (1 - y') * log(1 - hypo)) / (-m) + reg_term1;

% Calculate regularization term for cost function
reg_term2 = theta * lambda / m;
reg_term2(1) = 0;

% Calculate gradient using hypothesis calculated above
grad = X' * (hypo - y) / m + reg_term2;




% =============================================================

end
