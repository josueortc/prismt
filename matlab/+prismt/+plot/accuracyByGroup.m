function accuracyByGroup(ax, R, level)
%ACCURACYBYGROUP Test accuracy of each subject (or session), with chance marked.
%   One dot per subject shows how much the result depends on individuals; trials of one
%   subject are not independent, so this spread matters more than the trial count.
arguments
    ax
    R struct
    level (1, 1) string {mustBeMember(level, ["subject", "session"])} = "subject"
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
M = R.Metrics;
if ~isfield(M, 'by_group') || ~isfield(M.by_group, level)
    title(ax, "No " + level + " column in this dataset"); return
end
G = struct2table(M.by_group.(level), 'AsArray', true);
[~, order] = sort(G.accuracy);
G = G(order, :);
n = height(G);
for k = 1:n
    plot(ax, [0 G.accuracy(k)], [k k], '-', 'Color', s.grid, 'LineWidth', 1, 'HandleVisibility', 'off');
end
scatter(ax, G.accuracy, 1:n, 60, s.categorical(1, :), 'filled', 'MarkerEdgeColor', s.surface, 'LineWidth', 1.5);
chance = M.headline.chance;
xline(ax, chance, ':', "chance", 'Color', s.muted, 'LabelVerticalAlignment', 'bottom');
yticks(ax, 1:n); yticklabels(ax, string(G.group) + " (n=" + G.n + ")");
xlim(ax, [0 1]); xlabel(ax, "Accuracy on this " + level + "'s test trials");
title(ax, "Accuracy per " + level); grid(ax, 'on'); ax.YGrid = 'off';
end
