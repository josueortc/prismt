function learningCurves(ax, R, opts)
%LEARNINGCURVES Loss (or a validation score) per epoch, one line per fold.
%
%   prismt.plot.learningCurves(ax, R)                       % training and validation loss
%   prismt.plot.learningCurves(ax, R, Metric="val_balanced_accuracy")
%   The best epoch of each fold (the model that was kept) is marked with a dot.
arguments
    ax
    R struct
    opts.Metric (1, 1) string = "loss"
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
H = R.History;
if isempty(H) || height(H) == 0
    title(ax, "No training history yet"); return
end
folds = unique(H.fold)';
single = numel(folds) == 1;
for k = 1:numel(folds)
    rows = H(H.fold == folds(k), :);
    c = s.categorical(mod(k - 1, 8) + 1, :);
    if opts.Metric == "loss"
        if single
            plot(ax, rows.epoch, rows.train_loss, 'Color', s.categorical(1, :), 'LineWidth', s.lineWidth, 'DisplayName', "training");
            plot(ax, rows.epoch, rows.val_loss, 'Color', s.categorical(2, :), 'LineWidth', s.lineWidth, 'DisplayName', "validation");
            c = s.categorical(2, :);
        else
            plot(ax, rows.epoch, rows.val_loss, 'Color', c, 'LineWidth', s.lineWidth, 'DisplayName', "fold " + folds(k));
        end
        y = rows.val_loss;
    else
        if ~ismember(opts.Metric, string(rows.Properties.VariableNames))
            title(ax, "No " + opts.Metric + " in this run"); return
        end
        y = rows.(opts.Metric);
        plot(ax, rows.epoch, y, 'Color', c, 'LineWidth', s.lineWidth, 'DisplayName', "fold " + folds(k));
    end
    best = find(logical(rows.is_best), 1, 'last');
    if ~isempty(best)
        plot(ax, rows.epoch(best), y(best), 'o', 'MarkerSize', 7, 'MarkerFaceColor', c, 'MarkerEdgeColor', s.surface, ...
            'HandleVisibility', 'off');
    end
end
legend(ax, 'Location', 'best', 'Box', 'off', 'TextColor', s.ink);
grid(ax, 'on'); xlabel(ax, "Epoch");
if opts.Metric == "loss"
    ylabel(ax, "Loss"); title(ax, "Learning curves (dot = epoch kept)");
else
    ylabel(ax, replace(opts.Metric, "_", " ")); title(ax, replace(opts.Metric, "_", " ") + " per epoch");
end
end
