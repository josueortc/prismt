function st = syntheticStructure(profile)
%SYNTHETICSTRUCTURE The deterministic part of PRISMT's synthetic-data recipe.
%
%   st = prismt.demo.syntheticStructure("fast") returns the channel grid, latent loadings,
%   response kernel, planted channels and time axis of a synthetic profile ("tiny",
%   "fast" or "tutorial"). It is identical to prismt.data.synthetic.structure in Python;
%   tests compare both against tests/fixtures/synthetic_structure_v1.json. Channel
%   numbers are 1-based.
arguments
    profile (1, 1) string {mustBeMember(profile, ["tiny", "fast", "tutorial"])} = "fast"
end
switch profile
    case "tiny",     p = struct('n_mice', 4, 'n_sessions', 2, 'n_trials', 16, 'grid', [2 3], 'n_time', 6);
    case "fast",     p = struct('n_mice', 8, 'n_sessions', 2, 'n_trials', 30, 'grid', [3 4], 'n_time', 10);
    case "tutorial", p = struct('n_mice', 8, 'n_sessions', 4, 'n_trials', 40, 'grid', [4 6], 'n_time', 12);
end
c = constants();
nrow = p.grid(1); ncol = p.grid(2);
R = nrow * ncol; T = p.n_time;
r0 = (0:R - 1)';
rows = floor(r0 / ncol);
cols = mod(r0, ncol);
cx = (ncol - 1) * c.latent_centers(:, 1)';
cy = (nrow - 1) * c.latent_centers(:, 2)';
s = max(0.75, min(nrow, ncol) / 3);
loadings = exp(-((cols - cx) .^ 2 + (rows - cy) .^ 2) / (2 * s ^ 2));

onset = floor(T / 4);                      % 0-based index of the first post-stimulus sample
times = ((0:T - 1) - onset) / c.fs_hz;
tau = ((0:T - 1) - onset + 1) / c.fs_hz;
kernel = (tau > 0) .* (tau / c.kernel_peak_s) .* exp(1 - tau / c.kernel_peak_s);
kernel = kernel / max(kernel);

at = @(row, col) row * ncol + col + 1;     % 1-based channel number
patternA = [at(nrow - 1, 0), at(nrow - 1, 1)];
patternB = [at(0, ncol - 1), at(0, ncol - 2)];
noiseChannel = at(0, 0);
reserved = [patternA, patternB, noiseChannel];
free = setdiff(R:-1:1, reserved, 'stable');
c1 = free(1);
c2 = free(min(2, numel(free)));

st = struct();
st.recipe = 1;
st.profile = char(profile);
st.n_mice = p.n_mice;
st.n_sessions = p.n_sessions;
st.n_trials_per_session = p.n_trials;
st.grid = [nrow ncol];
st.shape = [p.n_mice * p.n_sessions * p.n_trials, R, T, 2];
st.channel_x = (cols + 1)';
st.channel_y = (rows + 1)';
st.loadings = loadings;
st.omegas = c.omegas;
st.ach_mix = c.ach_mix;
st.onset_index = onset + 1;
st.times_s = times;
st.kernel = kernel;
st.pattern_a = patternA;
st.pattern_b = patternB;
st.noise_channel = noiseChannel;
st.nan_channels = [2 c1; p.n_mice c2];
st.columns = ["mouse", "session", "phase", "stim", "response"];
st.modalities = ["calcium", "ach"];
end

function c = constants()
% Must match prismt/data/synthetic.py.
c.fs_hz = 10;
c.noise_sd = 0.35;
c.omegas = [0.5 1.0 1.5];
c.ach_mix = [1.0 0.7 0.5];
c.latent_centers = [0.15 0.2; 0.85 0.35; 0.45 0.9];
c.kernel_peak_s = 0.3;
end
