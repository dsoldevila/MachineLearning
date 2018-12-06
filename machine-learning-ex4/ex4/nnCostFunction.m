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

%1-----------
%Init y
recoded_y = zeros(num_labels, m);
for i=1:m,
  recoded_y(y(i), i) = 1; %10*5000
end;

%Feedforward
X = [ones(m,1), X];
Z = sigmoid(X*Theta1');
Z = [ones(m,1), Z];
H0 = sigmoid(Z*Theta2'); %5000*10

%Calculate cost
J = sum(sum(-recoded_y'.*log(H0)-(1-recoded_y').*log(1-H0)))/m;
regularization = lambda/(2*m)*(sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2)));
J += regularization;

%2-------------
for t=1:m,
  %Set input, training_set(t)
  a = X(t,:)'; %401*1
  %Feedforward (again)
  a2 = sigmoid(Theta1*a); %25*401 * 401*1 = 25*1
  a2 = [1; a2]; %26*1
  a3 = sigmoid(Theta2*a2); %10*26 * 26*1 = 10*1
  %Compute error
  delta_3 = a3-recoded_y(:,t); %10*1
  delta_2 = Theta2'*delta_3.*sigmoidGradient(a2); %26*10 * 10*1 .*26*1 = 26*1
  delta_2 = delta_2(2:end); %remove bias 25*1
  
  %Acumulate_error
  Theta1_grad += delta_2*a';  %25*1 * 1*401 = 25*401
  Theta2_grad += delta_3*a2'; %10*1 * 1*26 = 10*26
end;

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

grad = [reshape(Theta1_grad, 1,[]) reshape(Theta2_grad, 1, [])];


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
