function confusion(ax, R, opts)
%CONFUSION Confusion matrix of the test trials (rows: true class, columns: predicted).
%   Normalize=true (default) shows each row as fractions (per-class recall on the diagonal)
%   with the trial counts in brackets.
arguments
    ax
    R struct
    opts.Normalize (1, 1) logical = true
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
C = R.Mat.confusion_test;
names = R.Mat.class_names;
F = C ./ max(sum(C, 2), 1);
V = C; if opts.Normalize, V = F; end
imagesc(ax, V);
colormap(ax, s.sequential);
if opts.Normalize, prismt.plot.setLimits(ax, [0 1]); else, prismt.plot.setLimits(ax, [0 max(V(:))]); end
for i = 1:size(C, 1)
    for j = 1:size(C, 2)
        col = s.ink; if F(i, j) > 0.55, col = [1 1 1]; end
        txt = sprintf('%.0f%%\n(%d)', 100 * F(i, j), C(i, j));
        if ~opts.Normalize, txt = sprintf('%d', C(i, j)); end
        text(ax, j, i, txt, 'HorizontalAlignment', 'center', 'Color', col, 'FontSize', 9);
    end
end
xticks(ax, 1:numel(names)); xticklabels(ax, names); yticks(ax, 1:numel(names)); yticklabels(ax, names);
ax.YDir = 'reverse'; axis(ax, 'image');
xlabel(ax, "Predicted class"); ylabel(ax, "True class");
title(ax, "Test trials: " + sprintf('%.0f%% balanced accuracy', 100 * mean(diag(F), 'omitnan')));
end
