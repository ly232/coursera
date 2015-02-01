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
X = [ones(size(X, 1), 1), X];  % X: 5000x401
X_hidden = sigmoid((Theta1 * X')');  % X_hidden: 5000x25
X_hidden = [ones(size(X_hidden, 1), 1), X_hidden];  % X_hidden: 5000x26
X_hidden = sigmoid(Theta2*X_hidden');  % X_hidden: 10x5000
y_formatted = zeros(size(X_hidden));
y_formatted = y_formatted(:);
offset = [0:length(y)-1]' * size(X_hidden, 1);
y_formatted(y+offset) = 1;  % y_formatted: 10x5000
y_formatted = reshape(y_formatted, size(X_hidden));
J = 1/m * sum(sum(-y_formatted.*log(X_hidden) - (1-y_formatted).*log(1-X_hidden), 1), 2) + ...
  lambda/(2*m) * (sum(sum(Theta1(:, 2:size(Theta1, 2)).^2)) + sum(sum(Theta2(:, 2:size(Theta2, 2)).^2)));

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t = 1:m
  % set input layer's values a1 to t-th training example x(t)
  a1 = X(t, :)';  % 401x1
  z2 = Theta1 * a1;  % 25x1
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  % for each output unit k in layer 3 (otuput layer), set delta:
  delta3 = a3-y_formatted(:, t);
  % set delta for hidden layer 2:
  delta2 = Theta2'*delta3.*[0; sigmoidGradient(z2)];
  % accumulate gradient:
  Delta1 = Delta1 + delta2(2:end)*(a1');
  Delta2 = Delta2 + delta3*(a2');
  % obtain unregularized gradient for neural net cost function:
  Theta1_grad = 1/m * Delta1;
  Theta1_grad(:, 2:end) += lambda/m*Theta1(:, 2:end);
  Theta2_grad = 1/m * Delta2;
  Theta2_grad(:, 2:end) += lambda/m*Theta2(:, 2:end);
end













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
