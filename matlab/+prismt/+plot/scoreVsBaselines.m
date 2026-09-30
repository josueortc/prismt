function scoreVsBaselines(ax, R)
%SCOREVSBASELINES The model's test score next to what simple reference models achieve.
%   Classification: balanced accuracy of the transformer, logistic regression, always
%   guessing the most common class, chance, and (when the run had baselines.permutations)
%   the 95th percentile of logistic regression with shuffled labels. Masked autoencoder: R² of hidden values
%   for each masking pattern, against the best simple baseline for that pattern.
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
M = R.Metrics;
if string(M.task) == "classify"
    names = ["PRISMT", "logistic regression", "majority class", "shuffled labels (95th pct)", "chance"];
    vals = [M.test.balanced_accuracy, NaN, M.baselines.majority.balanced_accuracy, NaN, M.baselines.chance.balanced_accuracy];
    if isfield(M.baselines, 'logistic'), vals(2) = M.baselines.logistic.balanced_accuracy; end
    if isfield(M.baselines, 'shuffled_labels'), vals(4) = M.baselines.shuffled_labels.p95; end
    keep = ~isnan(vals); names = names(keep); vals = vals(keep);
    b = barh(ax, vals, 0.6, 'FaceColor', 'flat', 'EdgeColor', 'none');
    b.CData = repmat(s.muted * 0.4 + 0.6, numel(vals), 1); b.CData(1, :) = s.categorical(1, :);
    yticks(ax, 1:numel(vals)); yticklabels(ax, names); ax.YDir = 'reverse';
    for k = 1:numel(vals)
        text(ax, vals(k) + 0.01, k, sprintf('%.2f', vals(k)), 'Color', s.ink, 'VerticalAlignment', 'middle');
    end
    xlim(ax, [0 1.1]); xline(ax, vals(end), ':', 'Color', s.muted);
    xlabel(ax, "Balanced accuracy on test trials");
    title(ax, string(M.headline.description));
else
    masks = R.Mat.mask_names; S = numel(masks);
    [best, bi] = max(R.Mat.baseline_r2, [], 2);
    X = [R.Mat.r2(:), best(:)];
    b = barh(ax, X, 'grouped', 'EdgeColor', 'none');
    b(1).FaceColor = s.categorical(1, :); b(2).FaceColor = s.muted * 0.4 + 0.6;
    lbl = replace(masks, "_", " ");
    yticks(ax, 1:S); yticklabels(ax, lbl); ax.YDir = 'reverse';
    for k = 1:S
        text(ax, max(X(k, 1), 0) + 0.02, k - 0.15, sprintf('%.2f', X(k, 1)), 'Color', s.ink);
        text(ax, max(X(k, 2), 0) + 0.02, k + 0.15, sprintf('%.2f (%s)', X(k, 2), replace(R.Mat.baseline_names(bi(k)), "_", " ")), ...
            'Color', s.muted, 'FontSize', 8);
    end
    legend(ax, ["PRISMT", "best simple baseline"], 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
    xline(ax, 0, '-', 'Color', s.muted, 'HandleVisibility', 'off');
    xlabel(ax, "R² of hidden values (test trials)"); title(ax, "Masked reconstruction vs simple baselines");
end
grid(ax, 'on'); ax.YGrid = 'off';
end
