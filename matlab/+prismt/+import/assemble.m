function [ds, notes] = assemble(p, opts, file)
%ASSEMBLE Turn a reader's intermediate form into a prismt.Dataset.
notes = p.notes(:);
sigs = string(fieldnames(p.signals));
if ~isempty(opts.Signals)
    chosen = opts.Signals;
elseif strlength(opts.Signal)
    chosen = opts.Signal;
elseif ismember("dff", sigs)
    chosen = "dff";
else
    chosen = sigs(1);
end
missing = setdiff(matlab.lang.makeValidName(chosen), sigs);
if ~isempty(missing)
    error('prismt:E_IMPORT_NO_SIGNAL', 'There is no signal called %s. Signals in the file: %s.', ...
        strjoin(missing, ", "), strjoin(sigs, ", "));
end
chosen = matlab.lang.makeValidName(chosen);
if numel(sigs) > 1 && isscalar(chosen)
    notes(end + 1) = "Signal " + chosen + " was used (also available: " + strjoin(setdiff(sigs, chosen), ", ") + ").";
end
nS = numel(p.signals.(chosen(1)));
counts = cellfun(@(x) size(x, 1), p.signals.(chosen(1)));
N = sum(counts);
[R0, T] = deal(size(p.signals.(chosen(1)){1}, 2), size(p.signals.(chosen(1)){1}, 3));
for s = 1:nS
    for c = chosen
        sz = size(p.signals.(c){s});
        if sz(2) ~= R0 || sz(3) ~= T
            error('prismt:E_IMPORT_ROWS', 'Session %d (%s): %s is %s, but the first session is %d channels x %d time points.', ...
                s, p.sessionNames(s), c, mat2str(sz), R0, T);
        end
    end
end

% Neural part: one modality per chosen signal, or split channels into modalities.
X = zeros(N, R0, T, numel(chosen), 'single');
row = 0;
for s = 1:nS
    rows = row + (1:counts(s));
    for m = 1:numel(chosen)
        X(rows, :, :, m) = single(p.signals.(chosen(m)){s});
    end
    row = row + counts(s);
end
modNames = opts.ModalityNames;
if opts.Layout ~= "independent"
    K = opts.NModalities;
    if K < 2 || mod(R0, K)
        error('prismt:E_IMPORT_LAYOUT', 'Layout "%s" needs NModalities (2 or more) that divides the %d channels.', opts.Layout, R0);
    end
    if size(X, 4) > 1
        error('prismt:E_IMPORT_LAYOUT', 'Use either several Signals or a channel Layout, not both.');
    end
    Rk = R0 / K;
    Y = zeros(N, Rk, T, K, 'single');
    for k = 1:K
        if opts.Layout == "blocks", idx = (k - 1) * Rk + (1:Rk); else, idx = k:K:R0; end
        Y(:, :, :, k) = X(:, idx, :, 1);
    end
    X = Y;
    notes(end + 1) = sprintf("Channels split into %d modalities (%s).", K, opts.Layout);
end
if isempty(modNames)
    if numel(chosen) > 1
        modNames = chosen;
    elseif size(X, 4) > 1
        modNames = compose("signal%d", 1:size(X, 4));
    else
        modNames = "calcium";
        if chosen ~= "dff", modNames = chosen; end
    end
end
channelNames = compose("ch%02d", (1:size(X, 2))');
hemi = strings(0, 1);
atlas = opts.Atlas;
if strlength(atlas) == 0 && p.atlasHint == "grid82" && size(X, 2) == 82 && ~opts.AverageHemispheres
    atlas = "grid82";
    notes(end + 1) = "Channels placed on the 82-tile grid atlas (from the file's parcellation).";
end
if opts.AverageHemispheres
    if mod(size(X, 2), 2)
        error('prismt:E_IMPORT_LAYOUT', 'AverageHemispheres needs an even number of channels (pairs 2k-1, 2k).');
    end
    X = (X(:, 1:2:end, :, :) + X(:, 2:2:end, :, :)) / 2;
    channelNames = compose("region%02d", (1:size(X, 2))');
    notes(end + 1) = "Hemisphere pairs (channels 2k-1 and 2k) were averaged: " + 2 * size(X, 2) + " -> " + size(X, 2) + " channels.";
    if strlength(atlas) == 0 && p.atlasHint == "grid82", atlas = "grid41"; end
end
[x, y] = deal(zeros(0, 1));
if strlength(atlas)
    A = prismt.atlas.load(atlas);
    if A.NChannels ~= size(X, 2)
        error('prismt:E_IMPORT_ATLAS', 'Atlas %s has %d channels but the data have %d.', atlas, A.NChannels, size(X, 2));
    end
    x = A.X; y = A.Y;
    if atlas == "grid82"
        hemi = A.Hemisphere;
        channelNames = compose("tile%02d_%s", ceil((1:82)' / 2), hemi);
    end
end
Rn = size(X, 2);
M = size(X, 4);
units = opts.ModalityUnits;
if isempty(units), units = repmat("", 1, M); end
kinds = repmat("neural", 1, M);
chanSets = repmat({(1:Rn)'}, M, 1);

% Behavior: its own channels (named after the columns) in an extra modality.
beh = opts.Behavior;
if ~isempty(beh)
    beh = matlab.lang.makeValidName(beh);
    bad = setdiff(beh, string(fieldnames(p.behavior)));
    if ~isempty(bad)
        error('prismt:E_IMPORT_NO_SIGNAL', 'No per-trial time series called %s. Available: %s.', strjoin(bad, ", "), ...
            strjoin(string(fieldnames(p.behavior)), ", "));
    end
    B = numel(beh);
    Xb = nan(N, Rn + B, T, M + 1, 'single');
    Xb(:, 1:Rn, :, 1:M) = X;
    for b = 1:B
        Xb(:, Rn + b, :, M + 1) = reshape(single(vertcat(p.behavior.(beh(b)){:})), N, 1, T);
    end
    X = Xb;
    channelNames = [channelNames; beh(:)];
    if ~isempty(x), x = [x; nan(B, 1)]; y = [y; nan(B, 1)]; end
    if ~isempty(hemi), hemi = [hemi; repmat("", B, 1)]; end
    modNames = [modNames(:)', "behavior"];
    units = [units(:)', ""];
    kinds = [kinds, "behavior"];
    chanSets = [chanSets; {(Rn + 1:Rn + B)'}];
    notes(end + 1) = "Behavior (" + strjoin(beh, ", ") + ") added as a separate modality with its own channels.";
end

% Trial table: per-trial columns plus per-session values repeated for each trial.
cols = struct();
for c = string(fieldnames(p.trialCols))'
    parts = p.trialCols.(c);
    if numel(parts) ~= nS || any(cellfun(@numel, parts(:)) ~= counts(:))
        error('prismt:E_IMPORT_ROWS', 'Column %s does not have one value per trial in every session.', c);
    end
    cols.(c) = vertcat(parts{:});
end
sessionOf = repelem((1:nS)', counts);
for c = string(fieldnames(p.sessionCols))'
    v = p.sessionCols.(c);
    cols.(c) = v(sessionOf);
end
sessionCol = opts.Session;
if strlength(sessionCol) == 0
    sessionCol = "session";
    cols.session = p.sessionNames(sessionOf);
end
trials = struct2table(cols);
subject = opts.Subject;
if ~ismember(subject, string(trials.Properties.VariableNames))
    notes(end + 1) = "No '" + subject + "' column: animals cannot be kept apart (set Subject).";
    subject = "";
end
fs = opts.SamplingRate; t0 = opts.TimeZero;
if isempty(fs), fs = p.fs; end
if isempty(t0), t0 = p.t0; end
if isempty(fs), fs = 10; notes(end + 1) = "Sampling rate unknown; 10 Hz assumed (set SamplingRate)."; end
if isempty(t0), t0 = 0; notes(end + 1) = "Time of the first sample unknown; 0 s assumed (set TimeZero)."; end
labels = struct();
for c = string(fieldnames(p.labels))'
    if ismember(c, string(trials.Properties.VariableNames)) && isnumeric(trials.(c))
        labels.(c) = p.labels.(c);
    end
end
prov = struct('source_files', {{char(file)}}, 'importer', 'prismt.importData', ...
    'notes', {cellstr(notes(:)')}, 'signals', {cellstr(chosen)});
ds = prismt.makeDataset(X, trials, SamplingRate=fs, TimeZero=t0, Event="stimulus onset", ...
    ChannelNames=channelNames, ChannelX=x, ChannelY=y, Hemisphere=hemi, Atlas=atlas, ...
    ModalityNames=modNames, ModalityUnits=units, ModalityKinds=kinds, ModalityChannels=chanSets, ...
    Subject=subject, Session=sessionCol, ValueLabels=labels, Provenance=prov);
end
