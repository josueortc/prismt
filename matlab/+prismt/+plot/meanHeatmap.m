function h = meanHeatmap(ax, ds, opts)
%MEANHEATMAP Trial-averaged channels x time image of one modality.
%
%   prismt.plot.meanHeatmap(ax, ds)                          % all trials, first modality
%   prismt.plot.meanHeatmap(ax, ds, Modality="ach", Trials=ds.Trials.phase=="late")
%   prismt.plot.meanHeatmap(ax, ds, Difference={mask1, mask2}, Labels=["late","early"])
%   Missing values are ignored in the average; channels with no data are drawn gray.
arguments
    ax
    ds (1, 1) prismt.Dataset
    opts.Modality = 1
    opts.Trials = true(ds.N, 1)
    opts.Difference = {}
    opts.Labels (1, :) string = ["A", "B"]
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
m = modalityIndex(ds, opts.Modality);
chans = ds.ModalityChannels{m};
if isempty(opts.Difference)
    V = squeeze(mean(ds.X(opts.Trials, chans, :, m), 1, 'omitnan'));
    cmap = s.sequential; lim = [min(V(:)) max(V(:))];
    ttl = "Average " + ds.ModalityNames(m) + " (" + nnz(opts.Trials) + " trials)";
    if lim(1) < 0 && lim(2) > 0, cmap = s.diverging; lim = [-1 1] * max(abs(lim)); end
else
    a = squeeze(mean(ds.X(opts.Difference{1}, chans, :, m), 1, 'omitnan'));
    b = squeeze(mean(ds.X(opts.Difference{2}, chans, :, m), 1, 'omitnan'));
    V = a - b; cmap = s.diverging; lim = [-1 1] * max(abs(V(:)), [], 'omitnan');
    ttl = ds.ModalityNames(m) + ": " + opts.Labels(1) + " minus " + opts.Labels(2);
end
V = reshape(V, numel(chans), []);
[t, tl] = prismt.plot.timeAxis(ds);
h = imagesc(ax, t, 1:numel(chans), V, 'AlphaData', ~isnan(V));
colormap(ax, cmap);
prismt.plot.setLimits(ax, lim);
cb = colorbar(ax); cb.Color = s.muted;
cb.Label.String = unitLabel(ds, m);
ax.YDir = 'reverse';
axis(ax, 'tight');
names = ds.ChannelNames(chans);
if numel(chans) <= 40
    yticks(ax, 1:numel(chans)); yticklabels(ax, names);
end
xline(ax, 0, ':', 'Color', s.muted, 'LineWidth', 1);
xlabel(ax, tl); ylabel(ax, "Channel"); title(ax, ttl);
end

function m = modalityIndex(ds, m)
if isstring(m) || ischar(m)
    k = find(ds.ModalityNames == string(m), 1);
    if isempty(k), error('prismt:E_PLOT', 'No modality %s (there are: %s).', m, strjoin(ds.ModalityNames, ', ')); end
    m = k;
end
end

function u = unitLabel(ds, m)
u = ds.ModalityNames(m);
if strlength(ds.ModalityUnits(m)), u = u + " (" + ds.ModalityUnits(m) + ")"; end
end
