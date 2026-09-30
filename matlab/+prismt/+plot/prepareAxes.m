function prepareAxes(ax)
%PREPAREAXES Recessive grid and axes, ink-colored text (shared by the plot functions).
s = prismt.plot.style();
cla(ax, 'reset');
hold(ax, 'on');
set(ax, 'Color', s.surface, 'XColor', s.muted, 'YColor', s.muted, 'GridColor', s.grid, 'GridAlpha', 1, ...
    'FontSize', s.font, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 0.75);
ax.Title.Color = s.ink; ax.XLabel.Color = s.muted; ax.YLabel.Color = s.muted;
f = ancestor(ax, 'figure');
if ~isempty(f) && isprop(f, 'Theme') && ~isempty(f.Theme) %#ok<*ALIGN> compat-ok (R2025a)
    try, f.Theme = 'light'; catch, end
end
end
