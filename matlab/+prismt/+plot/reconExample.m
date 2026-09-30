function reconExample(axs, R, opts)
%RECONEXAMPLE One test trial: original, what the model saw, its reconstruction, the error.
%
%   f = figure; t = tiledlayout(f, 1, 4); axs = arrayfun(@(k) nexttile(t), 1:4);
%   prismt.plot.reconExample(axs, R, Example=1, Modality="ach")
%   Channels x time images in original units; hidden values are gray in the second panel.
arguments
    axs (1, 4)
    R struct
    opts.Example (1, 1) double = 1
    opts.Modality = 1
end
s = prismt.plot.style();
M = R.Mat;
m = opts.Modality;
if ~isnumeric(m), m = find(M.modality_names == string(m), 1); end
k = opts.Example;
orig = squeeze(M.example_original(k, :, :, m));
rec = squeeze(M.example_reconstruction(k, :, :, m));
hid = squeeze(M.example_hidden(k, :, :, m));
have = any(isfinite(orig), 2) | any(hid, 2);
orig = orig(have, :); rec = rec(have, :); hid = hid(have, :);
seen = orig; seen(hid) = NaN;
rec(~hid) = NaN;
err = rec - orig;
t = M.bin_centers_s(:)';
names = M.channel_names(have);
lim = [min(orig(:), [], 'omitnan') max(orig(:), [], 'omitnan')];
panels = {orig, seen, rec, err};
titles = ["Recorded", "Visible to the model", "Reconstructed (hidden values)", "Error (reconstructed - recorded)"];
for p = 1:4
    ax = axs(p);
    prismt.plot.prepareAxes(ax);
    V = panels{p};
    imagesc(ax, t, 1:size(V, 1), V, 'AlphaData', ~isnan(V));
    ax.Color = s.missing;
    if p < 4
        colormap(ax, s.sequential); if lim(1) < 0 && lim(2) > 0, colormap(ax, s.diverging); lim = [-1 1] * max(abs(lim)); end
        prismt.plot.setLimits(ax, lim);
    else
        colormap(ax, s.diverging); prismt.plot.setLimits(ax, [-1 1] * max(abs(err(:)), [], 'omitnan'));
    end
    colorbar(ax, 'Color', s.muted);
    ax.YDir = 'reverse'; axis(ax, 'tight');
    if p == 1 && numel(names) <= 40, yticks(ax, 1:numel(names)); yticklabels(ax, names); else, yticks(ax, []); end
    xlabel(ax, "Time (s)"); title(ax, titles(p), 'FontWeight', 'normal');
end
title(axs(1), "Trial " + M.example_trial_index(k) + ", " + M.modality_names(m) + ": recorded");
end
