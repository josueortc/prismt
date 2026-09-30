function probabilities(ax, R)
%PROBABILITIES How confident the model is: predicted probability of the true class,
%one distribution per true class (a well-separated model puts most mass near 1).
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
y = R.Mat.y_true; P = R.Mat.prob; names = R.Mat.class_names;
edges = linspace(0, 1, 21);
for k = 1:numel(names)
    p = P(y == k, k);
    histogram(ax, p, edges, 'Normalization', 'probability', 'DisplayStyle', 'stairs', 'LineWidth', s.lineWidth, ...
        'EdgeColor', s.categorical(k, :), 'DisplayName', names(k) + " (n=" + numel(p) + ")");
end
xline(ax, 1 / numel(names), ':', 'Color', s.muted, 'HandleVisibility', 'off');
legend(ax, 'Location', 'northwest', 'Box', 'off');
xlabel(ax, "Predicted probability of the true class"); ylabel(ax, "Fraction of test trials");
title(ax, "Prediction confidence"); grid(ax, 'on');
end
