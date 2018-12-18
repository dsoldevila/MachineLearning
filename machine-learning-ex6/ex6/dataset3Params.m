function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cs = [0.03 0.3 3 30];
sigmas = [0.01 0.1 1 10];

score_matrix = zeros(length(cs), length(sigmas));

for i=1:length(cs),
  for j=1:length(sigmas),
    model = svmTrain(X, y, cs(i), @(x1, x2) gaussianKernel(x1, x2, sigmas(j)));
    predictions = svmPredict(model, Xval);
    score_matrix(i,j) = mean(double(predictions ~= yval));
  endfor
endfor
score_matrix
[minval, row] = min(min(score_matrix,[],2));
[minval, col] = min(min(score_matrix,[],1));

C = cs(row);
sigma = sigmas(col);




% =========================================================================

end
