function plotData(X, y)


% Create New Figure
figure; hold on;

P = find(y==1);N = find(y == 0);


plot(X(P, 1), X(P, 2), 'b+','LineWidth', 2, 'MarkerSize', 7);
plot(X(N, 1), X(N, 2), 'ro', 'MarkerFaceColor', 'y','MarkerSize', 7);



hold off;

end
