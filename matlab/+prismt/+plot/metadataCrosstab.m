function T = metadataCrosstab(ax, ds, rowsBy, colsBy)
%METADATACROSSTAB Trial counts for two trial columns (e.g. phase x mouse), shown as a grid.
%   Empty cells (0 trials) are what to look for: a class missing from some subjects, or a
%   column that fully determines the label (a confound). Returns the counts as a table.
a = labelsOf(ds, rowsBy); b = labelsOf(ds, colsBy);
[ra, ~, ia] = unique(a); [cb, ~, ib] = unique(b);
C = accumarray([ia ib], 1, [numel(ra) numel(cb)]);
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
imagesc(ax, C, 'AlphaData', 0.9);
colormap(ax, s.sequential); prismt.plot.setLimits(ax, [0 max(C(:))]);
for i = 1:numel(ra)
    for j = 1:numel(cb)
        col = s.ink; if C(i, j) > 0.6 * max(C(:)), col = [1 1 1]; end
        txt = string(C(i, j)); if C(i, j) == 0, txt = "0"; col = s.critical; end
        text(ax, j, i, txt, 'HorizontalAlignment', 'center', 'Color', col, 'FontSize', 8);
    end
end
xticks(ax, 1:numel(cb)); xticklabels(ax, cb); yticks(ax, 1:numel(ra)); yticklabels(ax, ra);
xtickangle(ax, 45); ax.YDir = 'reverse'; axis(ax, 'tight');
xlabel(ax, colsBy); ylabel(ax, rowsBy); title(ax, "Trials per " + rowsBy + " and " + colsBy);
T = array2table(C, 'RowNames', cellstr(ra), 'VariableNames', cellstr(cb));
end

function g = labelsOf(ds, by)
v = ds.Trials.(by);
g = string(v);
if isnumeric(v)
    L = ds.ValueLabels(ds.ValueLabels.Column == by, :);
    for k = 1:height(L), g(v == L.Value(k)) = L.Label(k); end
end
g(ismissing(g)) = "(missing)";
end
