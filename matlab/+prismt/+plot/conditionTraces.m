function conditionTraces(ax, ds, by, opts)
%CONDITIONTRACES Mean +/- SEM over trials, one line per condition, for some channels.
%
%   prismt.plot.conditionTraces(ax, ds, "stim", Channels=["ch09","ch10"], Modality="calcium")
%   Channels are averaged together. SEM is computed across subjects when a subject
%   column is set (each subject contributes its mean), otherwise across trials.
arguments
    ax
    ds (1, 1) prismt.Dataset
    by (1, 1) string
    opts.Channels = []
    opts.Modality = 1
    opts.Trials = true(ds.N, 1)
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
m = opts.Modality;
if ~isnumeric(m), m = find(ds.ModalityNames == string(m), 1); end
ch = opts.Channels;
if isempty(ch), ch = ds.ModalityChannels{m}'; end
if ~isnumeric(ch), [~, ch] = ismember(string(ch), ds.ChannelNames); end
g = labelsOf(ds, by);
levels = unique(g(opts.Trials & ~ismissing(g)), 'stable');
if numel(levels) > 8
    error('prismt:E_PLOT', '%s has %d values; choose a column with at most 8 for traces.', by, numel(levels));
end
[t, tl] = prismt.plot.timeAxis(ds);
subj = [];
if strlength(ds.Subject), subj = string(ds.Trials.(ds.Subject)); end
for k = 1:numel(levels)
    rows = opts.Trials & g == levels(k);
    tr = squeeze(mean(ds.X(rows, ch, :, m), 2, 'omitnan'));
    tr = reshape(tr, nnz(rows), []);
    if ~isempty(subj) && numel(unique(subj(rows))) > 1
        [~, ~, j] = unique(subj(rows));
        per = splitapply(@(x) mean(x, 1, 'omitnan'), tr, j);
        mu = mean(per, 1, 'omitnan'); se = std(per, 0, 1, 'omitnan') / sqrt(size(per, 1));
    else
        mu = mean(tr, 1, 'omitnan'); se = std(tr, 0, 1, 'omitnan') / sqrt(size(tr, 1));
    end
    c = s.categorical(k, :);
    fill(ax, [t fliplr(t)], [mu + se fliplr(mu - se)], c, 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    plot(ax, t, mu, 'Color', c, 'LineWidth', s.lineWidth, 'DisplayName', levels(k) + " (n=" + nnz(rows) + ")");
end
xline(ax, 0, ':', 'Color', s.muted, 'HandleVisibility', 'off');
legend(ax, 'Location', 'best', 'Box', 'off', 'TextColor', s.ink);
chanText = strjoin(ds.ChannelNames(ch(1:min(end, 3))), ", ");
if numel(ch) > 3, chanText = chanText + sprintf(" +%d", numel(ch) - 3); end
xlabel(ax, tl); ylabel(ax, ds.ModalityNames(m) + unitsOf(ds, m));
title(ax, ds.ModalityNames(m) + " by " + by + " (" + chanText + ")");
grid(ax, 'on');
end

function g = labelsOf(ds, by)
v = ds.Trials.(by);
g = string(v);
if isnumeric(v)
    L = ds.ValueLabels(ds.ValueLabels.Column == by, :);
    for k = 1:height(L), g(v == L.Value(k)) = L.Label(k); end
end
end

function u = unitsOf(ds, m)
u = "";
if strlength(ds.ModalityUnits(m)), u = " (" + ds.ModalityUnits(m) + ")"; end
end
