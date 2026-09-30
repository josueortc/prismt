function channelMap(ax, values, opts)
%CHANNELMAP Draw one value per channel on a map, or as bars when there is no layout.
%
%   prismt.plot.channelMap(ax, v, Atlas="grid82")               % label-image atlas
%   prismt.plot.channelMap(ax, v, X=ds.ChannelX, Y=ds.ChannelY)  % positions -> tiles
%   prismt.plot.channelMap(ax, v, Names=ds.ChannelNames)         % no layout -> bars
%   Signed values use the diverging map with symmetric limits; others the blue ramp.
arguments
    ax
    values (:, 1) double
    opts.Atlas (1, 1) string = ""
    opts.X double = []
    opts.Y double = []
    opts.Names (:, 1) string = strings(0, 1)
    opts.Title (1, 1) string = ""
    opts.Label (1, 1) string = ""
    opts.Limits double = []
    opts.Signed = []
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
signed = opts.Signed;
if isempty(signed), signed = any(values < 0) && any(values > 0); end
lim = opts.Limits;
if isempty(lim)
    if signed, lim = [-1 1] * max(abs(values), [], 'omitnan'); else, lim = [min(values, [], 'omitnan') max(values, [], 'omitnan')]; end
end
cmap = s.sequential; if signed, cmap = s.diverging; end
if strlength(opts.Atlas)
    A = prismt.atlas.load(opts.Atlas);
    if numel(values) ~= A.NChannels
        error('prismt:E_PLOT', 'Atlas %s has %d channels, but %d values were given.', opts.Atlas, A.NChannels, numel(values));
    end
    img = nan(size(A.Labels));
    inside = A.Labels > 0;
    img(inside) = values(A.Labels(inside));
    imagesc(ax, img, 'AlphaData', ~isnan(img));
    axis(ax, 'image', 'off'); ax.YDir = 'reverse';
elseif ~isempty(opts.X) && ~isempty(opts.Y) && any(isfinite(opts.X))
    ok = isfinite(opts.X) & isfinite(opts.Y);
    scatter(ax, opts.X(ok), opts.Y(ok), 900, values(ok), 's', 'filled', 'MarkerEdgeColor', s.surface, 'LineWidth', 2);
    if ~isempty(opts.Names)
        text(ax, opts.X(ok), opts.Y(ok), opts.Names(ok), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', s.ink);
    end
    ax.YDir = 'reverse'; axis(ax, 'equal', 'off');
    xlim(ax, [min(opts.X(ok)) - 0.7, max(opts.X(ok)) + 0.7]); ylim(ax, [min(opts.Y(ok)) - 0.7, max(opts.Y(ok)) + 0.7]);
else
    b = bar(ax, values, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.8);
    idx = round(rescale(values, 1, 256, 'InputMin', lim(1), 'InputMax', lim(2)));
    idx(~isfinite(idx)) = 1;
    b.CData = cmap(idx, :);
    if ~isempty(opts.Names) && numel(values) <= 60
        xticks(ax, 1:numel(values)); xticklabels(ax, opts.Names); xtickangle(ax, 60);
    end
    ylabel(ax, opts.Label); grid(ax, 'on'); ax.XGrid = 'off';
end
colormap(ax, cmap);
prismt.plot.setLimits(ax, lim);
cb = colorbar(ax); cb.Color = s.muted; cb.Label.String = opts.Label;
title(ax, opts.Title);
end
