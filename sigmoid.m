function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z)); %initial value

g = (1./(1+exp(-z))); %applying sigmoid
% =============================================================

end
