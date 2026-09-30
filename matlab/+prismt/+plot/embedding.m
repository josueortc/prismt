function embedding(ax, R, colorBy, opts)
%EMBEDDING Test trials in the model's summary space (first two principal components).
%
%   prismt.plot.embedding(ax, R, ds.Trials.phase)             % colour by any trial column
%   prismt.plot.embedding(ax, R, ds.Trials.mouse, Max=3000)
%   colorBy is indexed by dataset trial number; at most 8 groups are coloured, the rest
%   are drawn as "Other". Shapes repeat the colours, so groups are not told apart by
%   colour alone.
arguments
    ax
    R struct
    colorBy = []
    opts.Max (1, 1) double = 5000
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
M = R.Mat;
if ~isfield(M, 'embedding_pcs')
    title(ax, "This run did not save embeddings"); return
end
P = M.embedding_pcs;
idx = M.embedding_trial_index(:);
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
ex = M.embedding_explained;
xlabel(ax, sprintf('PC 1 (%.0f%% of variance)', 100 * ex(1))); ylabel(ax, sprintf('PC 2 (%.0f%%)', 100 * ex(2)));
ttl = "Test trials in the model's summary space";
if n > opts.Max, ttl = ttl + sprintf(" (%d of %d shown)", opts.Max, n); end
title(ax, ttl); grid(ax, 'on'); axis(ax, 'square');
end
