function ds = makeDataset(X, trials, opts)
%MAKEDATASET Build a PRISMT dataset from your own arrays.
%
%   ds = prismt.makeDataset(X, trials, Name=Value, ...)
%
%   X       numeric array. By default its dimensions are trials x channels x time
%           (x modalities); use AxisOrder if yours are arranged differently. A channel is
%           anything recorded over time: a brain region or electrode, a behavior variable
%           (speed, pupil, a body-part coordinate), a physiological or stimulus signal.
%           A modality is a kind of signal; each can have its own channels
%           (ModalityChannels), e.g. 82 calcium channels and 3 behavior channels.
%   trials  table (or struct of column vectors) with one row per trial: subject, session,
%           condition, response, ... Any names and any number of columns. Use [] if you
%           have no per-trial information.
%
%   Several signals recorded on the same trials (e.g. neural activity and behavior): give X
%   as a struct with one field per signal, each trials x channels x time, or trials x time
%   for a single trace such as running speed. Signals may have different numbers of
%   channels; they must share trials and time points:
%     X = struct('neural', dff, 'pupil', pupil, 'speed', speed);
%     ds = prismt.makeDataset(X, trials, SamplingRate=30, Subject="mouse");
%   The field names become the signal names; channels are named "neural_01", ... (or the
%   signal name for one-channel signals) unless ChannelNames gives all of them in order.
%
%   Recordings with different channels (e.g. electrodes that differ between sessions):
%   make one dataset per recording and join them with prismt.combineDatasets, which
%   matches channels by name and marks the ones a recording lacks as missing.
%
%   Name=Value options
%     AxisOrder         order of the dimensions of X, e.g. "trials,time,channels".
%                       Words: trials, channels, time, modalities. Default
%                       "trials,channels,time,modalities".
%     SamplingRate      samples per second (Hz). Times are TimeZero + (0:T-1)/SamplingRate.
%     TimeZero          time of the first sample in seconds, e.g. -1.1 for 11 frames
%                       before the stimulus at 10 Hz. Default 0.
%     Times             explicit time of every sample in seconds (instead of the above).
%     Event             what time 0 is, e.g. "stimulus onset".
%     ChannelNames      one name per channel. Default "ch01", "ch02", ...
%     ChannelX, ChannelY  channel positions for maps (optional).
%     Hemisphere        "L"/"R" per channel (optional).
%     ChannelGroups     a group per channel (optional), e.g. brain areas, "left"/"right",
%                       or "arm"/"eye" for sensors. Runs can use some groups only
%                       (selection.channel_groups); "" = no group.
%     Atlas             atlas name, e.g. "grid82" (optional).
%     AtlasImage        label image, pixel value k = channel k (optional).
%     ModalityNames     e.g. ["calcium","ach"]. Default "signal1", ...
%     ModalityUnits     e.g. ["dF/F","dF/F"]. Default "".
%     ModalityKinds     what each modality is, e.g. "neural", "behavior", "physiology",
%                       "stimulus", "other" (any short text). Default "signal".
%     ModalityChannels  cell array: which channels exist for each modality (1-based).
%                       Default: all channels for every modality.
%     Subject           name of the trials column that identifies each subject (animal,
%                       participant...). Set it: results are then tested on new subjects.
%     Session           name of the trials column that identifies each session.
%     ValueLabels       struct, e.g. struct('stim', {{0,"CS-"; 1,"CS+"}}).
%     TrialUid          one unique id per trial (optional).
%     Provenance        struct saved with the dataset (where the data came from).
%
%   Example
%     X = rand(200, 82, 41);                          % trials x channels x time
%     T = table(repmat(["M1";"M2"],100,1), randi(2,200,1)-1, 'VariableNames', {'mouse','stim'});
%     ds = prismt.makeDataset(X, T, SamplingRate=10, TimeZero=-1.1, Subject="mouse", ...
%                             ModalityNames="calcium", ModalityUnits="dF/F");
%     prismt.writeDataset(ds, "mydata_prismt.mat");
arguments
    X {mustBeArrayOrSignals}
    trials = []
    opts.AxisOrder (1, 1) string = "trials,channels,time,modalities"
    opts.SamplingRate double = []
    opts.TimeZero (1, 1) double = 0
    opts.Times double = []
    opts.Event (1, 1) string = ""
    opts.ChannelNames string = strings(0, 1)
    opts.ChannelX double = []
    opts.ChannelY double = []
    opts.Hemisphere string = strings(0, 1)
    opts.ChannelGroups string = strings(0, 1)
    opts.Atlas (1, 1) string = ""
    opts.AtlasImage = []
    opts.ModalityNames string = strings(0, 1)
    opts.ModalityUnits string = strings(0, 1)
    opts.ModalityKinds string = strings(0, 1)
    opts.ModalityChannels cell = {}
    opts.Subject (1, 1) string = ""
    opts.Session (1, 1) string = ""
    opts.ValueLabels = struct()
    opts.TrialUid string = strings(0, 1)
    opts.Provenance struct = struct()
end

ds = prismt.Dataset();
if isstruct(X)
    [X, opts] = fromSignals(X, opts);
end
ds.X = single(orderAxes(X, opts.AxisOrder));
[N, R, T, M] = size(ds.X);

if isempty(trials)
    trials = table('Size', [N 0], 'VariableTypes', {});
elseif isstruct(trials)
    trials = struct2table(trials);
elseif ~istable(trials)
    error('prismt:E_DATA_COLUMNS', 'trials must be a table, a struct of column vectors, or [].');
end
ds.Trials = normalizeColumns(trials);

ds.ChannelNames = defaultNames(opts.ChannelNames, R, "ch%02d");
ds.ChannelX = opts.ChannelX(:);
ds.ChannelY = opts.ChannelY(:);
ds.Hemisphere = opts.Hemisphere(:);
ds.ChannelGroups = opts.ChannelGroups(:);
ds.Atlas = opts.Atlas;
if ~isempty(opts.AtlasImage), ds.AtlasImage = uint16(opts.AtlasImage); end
ds.ModalityNames = defaultNames(opts.ModalityNames, M, "signal%d");
ds.ModalityUnits = fillTo(opts.ModalityUnits, M, "");
ds.ModalityKinds = fillTo(opts.ModalityKinds, M, "signal");
if isempty(opts.ModalityChannels)
    ds.ModalityChannels = repmat({(1:R)'}, M, 1);
else
    ds.ModalityChannels = cellfun(@(c) double(c(:)), opts.ModalityChannels(:), 'UniformOutput', false);
end

if ~isempty(opts.Times)
    ds.Times = double(opts.Times(:));
elseif ~isempty(opts.SamplingRate)
    ds.Times = opts.TimeZero + (0:T - 1)' / opts.SamplingRate;
else
    ds.Times = (0:T - 1)';
    ds.Provenance.time_unknown = true;
end
ds.Event = opts.Event;
ds.Subject = opts.Subject;
ds.Session = opts.Session;
ds.TrialUid = opts.TrialUid(:);
ds.ValueLabels = labelsTable(opts.ValueLabels);
provenance = opts.Provenance;
for f = string(fieldnames(ds.Provenance))'
    provenance.(f) = ds.Provenance.(f);
end
ds.Provenance = provenance;

prismt.internal.throwIfErrors(ds.validate(), "The dataset is not valid");
end

function Y = orderAxes(X, order)
words = strtrim(split(lower(order), ","))';
words(words == "") = [];
canonical = ["trials", "channels", "time", "modalities"];
aliases = struct('trial', "trials", 'channel', "channels", 'regions', "channels", 'region', "channels", ...
    'times', "time", 'timepoints', "time", 'frames', "time", 'modality', "modalities", 'signals', "modalities");
for k = 1:numel(words)
    if isfield(aliases, words(k)), words(k) = aliases.(words(k)); end
end
if numel(words) < ndims(X) || any(~ismember(words, canonical)) || numel(unique(words)) < numel(words)
    error('prismt:E_DATA_SHAPE', ['AxisOrder "%s" must name each dimension of X once, using ' ...
        'trials, channels, time and modalities.'], order);
end
missing = setdiff(canonical, words, 'stable');
words = [words, missing];               % absent axes have size 1
[~, perm] = ismember(canonical, words);
Y = permute(X, perm);
end

function names = defaultNames(names, n, pattern)
names = names(:);
if isempty(names)
    names = compose(pattern, (1:n)');
end
end

function v = fillTo(v, n, default)
v = v(:);
if isempty(v)
    v = repmat(string(default), n, 1);
elseif isscalar(v) && n > 1
    v = repmat(v, n, 1);
end
end

function T = normalizeColumns(T)
% Text columns become categorical (what PRISMT stores); everything else is kept.
for c = 1:width(T)
    col = T.(c);
    if iscellstr(col) || ischar(col) || isstring(col)
        s = string(col);
        s(strlength(s) == 0) = missing;
        T.(c) = categorical(s);
    end
end
end

function L = labelsTable(v)
L = table(strings(0, 1), zeros(0, 1), strings(0, 1), 'VariableNames', {'Column', 'Value', 'Label'});
if istable(v)
    L = [L; v(:, {'Column', 'Value', 'Label'})];
    return
end
for f = string(fieldnames(v))'
    pairs = v.(f);
    if isa(pairs, 'containers.Map')
        keys = pairs.keys; vals = pairs.values;
        pairs = [keys(:), vals(:)];
    end
    for k = 1:size(pairs, 1)
        L(end + 1, :) = {f, double(pairs{k, 1}), string(pairs{k, 2})}; %#ok<AGROW>
    end
end
end

function mustBeArrayOrSignals(X)
if ~(isnumeric(X) || islogical(X) || (isstruct(X) && isscalar(X)))
    error('prismt:E_DATA_SHAPE', 'X must be a numeric array, or a struct with one numeric field per signal.');
end
end

function [Y, opts] = fromSignals(S, opts)
% One field per signal -> trials x (all channels) x time x signals, missing where a signal has no channel.
names = string(fieldnames(S));
if isempty(names), error('prismt:E_DATA_SHAPE', 'X has no signals (the struct is empty).'); end
order = erase(lower(opts.AxisOrder), [",modalities", "modalities,", "modalities"]);
parts = cell(numel(names), 1);
for k = 1:numel(names)
    x = S.(names(k));
    if ~(isnumeric(x) || islogical(x))
        error('prismt:E_DATA_SHAPE', 'Signal %s must be numeric.', names(k));
    end
    if ismatrix(x)
        x = reshape(x, size(x, 1), 1, size(x, 2));      % trials x time: one channel
    else
        x = orderAxes(x, order);
    end
    parts{k} = x;
end
N = size(parts{1}, 1); T = size(parts{1}, 3);
for k = 2:numel(names)
    if size(parts{k}, 1) ~= N || size(parts{k}, 3) ~= T
        error('prismt:E_DATA_SHAPE', ['Signal %s has %d trials x %d time points but %s has %d x %d. Signals given ' ...
            'together must share trials and time points (resample them to a common time base first).'], ...
            names(k), size(parts{k}, 1), size(parts{k}, 3), names(1), N, T);
    end
end
counts = cellfun(@(x) size(x, 2), parts);
R = sum(counts);
Y = nan(N, R, T, numel(names), 'single');
chanSets = cell(numel(names), 1);
chanNames = strings(R, 1);
first = 0;
for k = 1:numel(names)
    idx = first + (1:counts(k));
    Y(:, idx, :, k) = single(parts{k});
    chanSets{k} = idx(:);
    if counts(k) == 1
        chanNames(idx) = names(k);
    else
        chanNames(idx) = compose(names(k) + "_%02d", (1:counts(k))');
    end
    first = first + counts(k);
end
if isempty(opts.ChannelNames), opts.ChannelNames = chanNames; end
if isempty(opts.ModalityNames), opts.ModalityNames = names; end
if isempty(opts.ModalityChannels), opts.ModalityChannels = chanSets; end
opts.AxisOrder = "trials,channels,time,modalities";
end
