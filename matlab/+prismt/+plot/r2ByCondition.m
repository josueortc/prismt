function T = r2ByCondition(ax, R, ds, column, opts)
%R2BYCONDITION Predictability of hidden values per condition (e.g. early vs late learning).
%
%   T = prismt.plot.r2ByCondition(ax, R, ds, "phase", Modality="ach")
%   Pools the per-trial sums the run saved, so any trial column can be used after the fact.
%   Error bars: 2.5-97.5% range over 500 bootstrap resamples of sessions (or trials when
%   the dataset has no session column). Exploratory: conditions were not part of training.
arguments
    ax
    R struct
    ds (1, 1) prismt.Dataset
    column (1, 1) string
    opts.Mask (1, 1) string = ""
    opts.Modality = 1
    opts.Boot (1, 1) double = 500
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
M = R.Mat;
si = 1; if strlength(opts.Mask), si = find(M.mask_names == opts.Mask, 1); end
m = opts.Modality; if ~isnumeric(m), m = find(M.modality_names == string(m), 1); end
q = M.pairs(:, 2) == m;
n = squeeze(M.pt_n(si, :, q)); s1 = squeeze(M.pt_sum_y(si, :, q)); s2 = squeeze(M.pt_sum_y2(si, :, q));
sse = squeeze(M.pt_sse(si, :, q));
tri = M.pt_trial_index(:);
g = string(ds.Trials.(column)); g = g(tri);
if isnumeric(ds.Trials.(column))
    L = ds.ValueLabels(ds.ValueLabels.Column == column, :);
    v = ds.Trials.(column)(tri);
    for k = 1:height(L), g(v == L.Value(k)) = L.Label(k); end
end
sess = (1:numel(tri))';
if strlength(ds.Session)
    key = string(ds.Trials.(ds.Session)(tri));
    if strlength(ds.Subject), key = string(ds.Trials.(ds.Subject)(tri)) + "/" + key; end  % unique per animal
    [~, ~, sess] = unique(key);
end
levels = unique(g(~ismissing(g)), 'stable');
rs = RandStream('mt19937ar', 'Seed', 0);
T = table('Size', [numel(levels) 5], 'VariableTypes', ["string", "double", "double", "double", "double"], ...
    'VariableNames', ["Condition", "R2", "Low", "High", "Trials"]);
for k = 1:numel(levels)
    rows = g == levels(k);
    r2 = pooled(rows);
    us = unique(sess(rows));
    boot = nan(opts.Boot, 1);
    for b = 1:opts.Boot
        pick = us(randi(rs, numel(us), numel(us), 1));
        w = accumarray(pick, 1, [max(sess) 1]);
        boot(b) = pooled(rows, w(sess));
    end
    T(k, :) = {levels(k), r2, prctile2(boot, 2.5), prctile2(boot, 97.5), nnz(rows)};
end
for k = 1:height(T)
    errorbar(ax, k, T.R2(k), T.R2(k) - T.Low(k), T.High(k) - T.R2(k), 'o', 'Color', s.categorical(k, :), ...
        'MarkerFaceColor', s.categorical(k, :), 'LineWidth', s.lineWidth, 'MarkerSize', 8, 'CapSize', 0);
    text(ax, k + 0.12, T.R2(k), sprintf('%.2f', T.R2(k)), 'Color', s.ink);
end
xticks(ax, 1:height(T)); xticklabels(ax, T.Condition + " (n=" + T.Trials + ")");
xlim(ax, [0.5 height(T) + 0.5]); grid(ax, 'on'); ax.XGrid = 'off';
yl = ylim(ax); ylim(ax, [min(0, yl(1)), max(yl(2), 0.05)]);   % keep 0 in view so small differences look small
ylabel(ax, "R² of hidden values"); xlabel(ax, column);
title(ax, M.modality_names(m) + " predictability by " + column + " (exploratory)");

    function r = pooled(rows, w)
        if nargin < 2, w = ones(numel(rows), 1); end
        w = w(:) .* rows(:);
        N = sum(w .* n, 'all'); S1 = sum(w .* s1, 'all'); S2 = sum(w .* s2, 'all'); E = sum(w .* sse, 'all');
        sst = S2 - S1 ^ 2 / max(N, 1);
        r = 1 - E / sst;
        if N < 2 || sst <= 0, r = NaN; end
    end
end

function v = prctile2(x, p)
% Percentile without the Statistics Toolbox.
x = sort(x(isfinite(x)));
if isempty(x), v = NaN; return; end
v = interp1(linspace(0, 100, numel(x)), x, p);
end
