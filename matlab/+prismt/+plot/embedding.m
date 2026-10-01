function embedding(ax, R, colorBy, opts)
%EMBEDDING Test trials in the model's summary space (first two principal components).
%
%   prismt.plot.embedding(ax, R, ds.Trials.phase)             % colour by any trial column
%   prismt.plot.embedding(ax, R, ds.Trials.mouse, Max=3000)
%   prismt.plot.embedding(ax, R, ds.Trials.phase, Fold=2)     % with cross-validation
%   colorBy is indexed by dataset trial number; at most 8 groups are coloured, the rest
%   are drawn as "Other". Shapes repeat the colours, so groups are not told apart by
%   colour alone. With cross-validation every fold has its own model, and so its own
%   summary space: one fold is shown at a time (Fold, default 1).
arguments
    ax
    R struct
    colorBy = []
    opts.Max (1, 1) double = 5000
    opts.Fold (1, 1) double = 1
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
M = R.Mat;
if ~isfield(M, 'embedding_pcs')
    title(ax, "This run did not save embeddings"); return
end
P = M.embedding_pcs;
idx = M.embedding_trial_index(:);
ex = M.embedding_explained;
nFolds = 1;
if isfield(M, 'embedding_fold')
    f = M.embedding_fold(:);
    nFolds = max(f);
    fold = min(max(1, opts.Fold), nFolds);
    P = P(f == fold, :); idx = idx(f == fold);
    if size(ex, 1) >= fold && size(ex, 2) > 1, ex = ex(fold, :); end
end
n = size(P, 1);
keep = 1:n;
if n > opts.Max
    rs = RandStream('mt19937ar', 'Seed', 0);
    keep = sort(randperm(rs, n, opts.Max));
end
if isempty(colorBy)
    g = repmat("trials", n, 1);
else
    g = string(colorBy); g = g(idx);
end
levels = unique(g, 'stable');
if numel(levels) > 8
    counts = arrayfun(@(l) nnz(g == l), levels);
    [~, o] = sort(counts, 'descend');
    g(~ismember(g, levels(o(1:7)))) = "Other";
    levels = [levels(o(1:7)); "Other"];
end
markers = 'os^dvp<>';
for k = 1:numel(levels)
    rows = keep(g(keep) == levels(k));
    c = s.categorical(k, :); if levels(k) == "Other", c = s.missing; end
    scatter(ax, P(rows, 1), P(rows, 2), 14, c, markers(k), 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', levels(k) + " (" + numel(rows) + ")");
end
if numel(levels) > 1, legend(ax, 'Location', 'bestoutside', 'Box', 'off'); end
xlabel(ax, sprintf('PC 1 (%.0f%% of variance)', 100 * ex(1))); ylabel(ax, sprintf('PC 2 (%.0f%%)', 100 * ex(2)));
ttl = "Test trials in the model's summary space";
sub = "";
if nFolds > 1, sub = sprintf("fold %d of %d (each fold has its own model)", fold, nFolds); end
if n > opts.Max, ttl = ttl + sprintf(" (%d of %d shown)", opts.Max, n); end
if strlength(sub)
    title(ax, {char(ttl), char(sub)}, 'FontWeight', 'normal');
else
    title(ax, ttl);
end
grid(ax, 'on'); axis(ax, 'square');
end
