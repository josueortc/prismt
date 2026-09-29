function [ds, truth] = makeSyntheticDataset(opts)
%MAKESYNTHETICDATASET Demo data with known structure, for learning PRISMT.
%
%   [ds, truth] = prismt.demo.makeSyntheticDataset(Profile="fast", Difficulty="medium", Seed=0)
%
%   Imitates a two-colour widefield experiment: "calcium" and "ach" recorded on a small
%   grid of cortical channels, trials aligned to a stimulus (CS+ / CS-), several mice,
%   early and late learning sessions. What is planted, and what a working model should
%   therefore find (truth lists the channels):
%     - On CS+ trials, the two bottom-left channels show an evoked calcium response
%       (truth.StimulusChannels): classify "stim" to find it.
%     - In late sessions, the two top-right channels show an evoked ACh response
%       (truth.LearningChannels): classify "phase" (early vs late) across mice to find it.
%     - All channels mix three smooth latent factors and ACh is a lagged mixture of the
%       same factors, so a masked autoencoder can predict hidden values, and ACh from
%       calcium. The top-left channel (truth.NoiseChannel) is pure noise.
%     - Two (mouse, channel) pairs are entirely missing (NaN), plus ~1% of traces.
%
%   Profile     "tiny" (tests), "fast" (quick demos, default), "tutorial" (the tutorial)
%   Difficulty  "easy", "medium" (default) or "hard": size of the planted responses
%   Seed        random seed (uses its own stream; the global rng is not touched)
%
%   The same recipe is implemented in Python (python -m prismt synth); the structure is
%   identical, the random draws differ.
arguments
    opts.Profile (1, 1) string {mustBeMember(opts.Profile, ["tiny", "fast", "tutorial"])} = "fast"
    opts.Difficulty (1, 1) string {mustBeMember(opts.Difficulty, ["easy", "medium", "hard"])} = "medium"
    opts.Seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
end
st = prismt.demo.syntheticStructure(opts.Profile);
rs = RandStream('mt19937ar', 'Seed', opts.Seed);
difficulty = struct('easy', 6.0, 'medium', 3.0, 'hard', 1.5);
noiseSd = 0.35; offsetSd = 0.2; arousalSd = 0.3; nanFraction = 0.01;
delta = difficulty.(opts.Difficulty) * noiseSd;

S = st.n_mice; Q = st.n_sessions; n = st.n_trials_per_session;
N = st.shape(1); R = st.shape(2); T = st.shape(3); M = st.shape(4);
L = st.loadings; K = size(L, 2);
t = 0:T - 1;

mouseIdx = repelem((1:S)', Q * n);
sessionIdx = repmat(repelem((1:Q)', n), S, 1);
late = sessionIdx > floor(Q / 2);
stim = zeros(N, 1);
for k = 1:S * Q
    v = mod(0:n - 1, 2)';
    stim((k - 1) * n + (1:n)) = v(randperm(rs, n));
end
pCorrect = 0.6 + 0.25 * late;
correct = rand(rs, N, 1) < pCorrect;
response = zeros(N, 1);
response(stim == 1 & correct) = 1;      % hit
response(stim == 1 & ~correct) = 0;     % miss
response(stim == 0 & correct) = 2;      % correct rejection
response(stim == 0 & ~correct) = 3;     % false alarm

amp = randn(rs, N, K);
phase = 2 * pi * rand(rs, N, K);
f = zeros(N, K, T);
for k = 1:K
    f(:, k, :) = reshape(amp(:, k) .* cos(2 * pi * st.omegas(k) * t / T + phase(:, k)), N, 1, T);
end
fLag = cat(3, f(:, :, 1), f(:, :, 1:end - 1));
calcium = zeros(N, R, T);
ach = zeros(N, R, T);
for tt = 1:T
    calcium(:, :, tt) = f(:, :, tt) * L';
    ach(:, :, tt) = (fLag(:, :, tt) .* st.ach_mix) * L';
end
ach = ach + reshape(arousalSd * randn(rs, N, 1) .* (t / max(T - 1, 1)), N, 1, T);
calcium(:, st.noise_channel, :) = 0;
ach(:, st.noise_channel, :) = 0;

evoked = st.kernel .* (1 + 0.3 * randn(rs, N, 1));           % N x T
for r = st.pattern_a
    calcium(:, r, :) = calcium(:, r, :) + reshape(delta * evoked .* (stim == 1), N, 1, T);
end
evokedB = st.kernel .* (1 + 0.3 * randn(rs, N, 1));
for r = st.pattern_b
    ach(:, r, :) = ach(:, r, :) + reshape(delta * evokedB .* late, N, 1, T);
end

gain = 1 + 0.15 * randn(rs, S, 1);
offsets = offsetSd * randn(rs, S, Q, R, M);
X = cat(4, calcium, ach) .* gain(mouseIdx);
off = zeros(N, R, 1, M);
for i = 1:N
    off(i, :, 1, :) = reshape(offsets(mouseIdx(i), sessionIdx(i), :, :), 1, R, 1, M);
end
X = X + off + noiseSd * randn(rs, N, R, T, M);

for k = 1:size(st.nan_channels, 1)
    X(mouseIdx == st.nan_channels(k, 1), st.nan_channels(k, 2), :, :) = NaN;
end
eligible = setdiff(1:R, [st.pattern_a, st.pattern_b]);
nRandom = round(nanFraction * N * numel(eligible) * M);
for k = 1:nRandom
    X(randi(rs, N), eligible(randi(rs, numel(eligible))), :, randi(rs, M)) = NaN;
end

mice = compose("M%02d", mouseIdx);
phases = categorical(repmat("early", N, 1), ["early", "late"]);
phases(late) = "late";
trials = table(categorical(mice), sessionIdx, phases, stim, response, ...
    'VariableNames', {'mouse', 'session', 'phase', 'stim', 'response'});

truth = struct('Profile', opts.Profile, 'Difficulty', opts.Difficulty, 'Seed', opts.Seed, ...
    'StimulusChannels', st.pattern_a, 'LearningChannels', st.pattern_b, ...
    'NoiseChannel', st.noise_channel, 'MissingChannels', st.nan_channels, ...
    'OnsetIndex', st.onset_index, ...
    'Description', "Stimulus signal: calcium in StimulusChannels on CS+ trials. " + ...
    "Learning signal: ACh in LearningChannels in late sessions. NoiseChannel is unpredictable.");
provenance.synthetic = struct('recipe', 1, 'profile', char(opts.Profile), ...
    'difficulty', char(opts.Difficulty), 'seed', opts.Seed, ...
    'pattern_a', {prismt.internal.jsonArray(st.pattern_a)}, ...
    'pattern_b', {prismt.internal.jsonArray(st.pattern_b)}, ...
    'noise_channel', st.noise_channel, ...
    'nan_channels', {num2cell(st.nan_channels, 2)'}, ...
    'onset_index', st.onset_index, 'generator', 'matlab');

ds = prismt.makeDataset(X, trials, ...
    Times=st.times_s, Event="stimulus onset", ...
    ChannelNames=compose("ch%02d", (1:R)'), ChannelX=st.channel_x, ChannelY=st.channel_y, ...
    ModalityNames=["calcium", "ach"], ModalityUnits=["dF/F", "dF/F"], ModalityKinds=["neural", "neural"], ...
    Subject="mouse", Session="session", ...
    ValueLabels=struct('stim', {{0, "CS-"; 1, "CS+"}}, 'response', {{0, "miss"; 1, "hit"; 2, "CR"; 3, "FA"}}), ...
    Provenance=provenance);
end
