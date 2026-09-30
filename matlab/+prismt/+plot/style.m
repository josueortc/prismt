function s = style()
%STYLE Colors and line specs shared by every PRISMT plot (validated palette; light theme).
%   s.categorical   8 x 3, series colors in fixed order (never cycled; fold extras into "Other")
%   s.sequential    256 x 3 single-hue (blue) ramp for magnitudes, light -> dark
%   s.diverging     256 x 3 blue <-> gray <-> red, for signed values (use symmetric limits)
%   s.ink, s.muted, s.grid, s.surface; s.lineWidth; s.good/.warning/.serious/.critical
hex = @(h) sscanf(h(2:end), '%2x%2x%2x', [1 3]) / 255;
cats = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
s.categorical = cell2mat(arrayfun(@(c) hex(char(c)), cats', 'UniformOutput', false));
seq = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
s.sequential = ramp(cell2mat(arrayfun(@(c) hex(char(c)), seq', 'UniformOutput', false)), 256);
div = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2b3b2", "#e34948", "#9e2b2a"];
s.diverging = ramp(cell2mat(arrayfun(@(c) hex(char(c)), div', 'UniformOutput', false)), 256);
s.ink = hex('#0b0b0b'); s.muted = hex('#52514e'); s.grid = hex('#e4e3df'); s.surface = hex('#fcfcfb');
s.missing = hex('#d9d8d4');
s.good = hex('#0ca30c'); s.warning = hex('#fab219'); s.serious = hex('#ec835a'); s.critical = hex('#d03b3b');
s.lineWidth = 1.5;
s.font = 10;
end

function C = ramp(stops, n)
x = linspace(0, 1, size(stops, 1));
C = interp1(x, stops, linspace(0, 1, n));
end
