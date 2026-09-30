function [ds, notes] = combineDatasets(datasets, opts)
%COMBINEDATASETS Join datasets whose channels, signals or trial columns differ.
%
%   ds = prismt.combineDatasets({ds1, ds2, ds3})
%   [ds, notes] = prismt.combineDatasets({dsA, dsB}, Source="recording", Names=["A", "B"])
%
%   Use it when recordings do not share the same channels: electrodes that differ between
%   sessions, a camera that tracked other body parts, a signal only recorded in some
%   animals. Make one dataset per recording (prismt.makeDataset or prismt.importData),
%   then combine them:
%     - channels are matched by name (ChannelNames); a channel a recording lacks is
%       missing (NaN) for its trials, and the model skips missing values;
%     - signals (modalities) are matched by name in the same way;
%     - trial columns are joined; a column a dataset lacks is missing for its trials;
%     - all datasets must have the same time axis. Resample=true interpolates every dataset
%       onto the time points of the first one (within the time range they all cover).
%
%   Name them so they match: two recordings of the same electrode must use the same
%   channel name, and different electrodes different names. notes says what was done.
%
%   Options
%     Source    name of a new trial column saying which dataset each trial came from ("": none)
%     Names     one name per dataset for that column (default "1", "2", ...)
%     Resample  interpolate onto the first dataset's time points (default false)
arguments
    datasets
    opts.Source (1, 1) string = ""
    opts.Names string = strings(0, 1)
    opts.Resample (1, 1) logical = false
end
if ~iscell(datasets), datasets = num2cell(datasets); end
K = numel(datasets);
if K == 0, error('prismt:E_COMBINE', 'Give at least one dataset.'); end
names = opts.Names(:);
if isempty(names), names = string(1:K)'; end
if numel(names) ~= K, error('prismt:E_COMBINE', 'Give one name per dataset (%d).', K); end
notes = strings(0, 1);

% ---- time ----------------------------------------------------------------------------
times = datasets{1}.Times;
for k = 2:K
    tk = datasets{k}.Times;
    same = numel(tk) == numel(times) && max(abs(tk - times)) <= 1e-6 * max(1, max(abs(times)));
    if same, continue; end
    if ~opts.Resample
        error('prismt:E_COMBINE_TIME', ['Dataset %d has %d time points (%.3g to %.3g s) but dataset 1 has %d ' ...
            '(%.3g to %.3g s). Make them match, or use Resample=true to interpolate onto the time points ' ...
            'of dataset 1.'], k, numel(tk), tk(1), tk(end), numel(times), times(1), times(end));
    end
end
if opts.Resample
    lo = max(cellfun(@(d) d.Times(1), datasets));
    hi = min(cellfun(@(d) d.Times(end), datasets));
    keep = times >= lo - 1e-9 & times <= hi + 1e-9;
    if ~any(keep), error('prismt:E_COMBINE_TIME', 'The datasets share no time range.'); end
    if ~all(keep), notes(end + 1, 1) = sprintf("Time limited to %.3g to %.3g s, the range all datasets cover.", lo, hi); end
    times = times(keep);
end
T = numel(times);

% ---- channels and signals, matched by name --------------------------------------------
chan = strings(0, 1); mods = strings(0, 1);
for k = 1:K
    chan = [chan; col(setdiff(datasets{k}.ChannelNames, chan, 'stable'))]; %#ok<AGROW>
    mods = [mods; col(setdiff(datasets{k}.ModalityNames, mods, 'stable'))]; %#ok<AGROW>
end
R = numel(chan); M = numel(mods);
counts = cellfun(@(d) d.N, datasets(:));
N = sum(counts);
X = nan(N, R, T, M, 'single');
present = false(R, M);
[units, kinds] = deal(strings(M, 1));
unitsSet = false(M, 1);
cx = nan(R, 1); cy = nan(R, 1); hemi = strings(R, 1); groups = strings(R, 1);
row = 0;
for k = 1:K
    d = datasets{k};
    rows = row + (1:d.N);
    [~, ci] = ismember(d.ChannelNames, chan);
    [~, mi] = ismember(d.ModalityNames, mods);
    Xk = d.X;
    if opts.Resample && ~isequal(d.Times, times)
        Xk = resampleTime(Xk, d.Times, times);
    elseif opts.Resample
        Xk = Xk(:, :, keepIndex(d.Times, times), :);
    end
    for m = 1:d.M
        cs = d.ModalityChannels{m};
        X(rows, ci(cs), :, mi(m)) = Xk(:, cs, :, m);
        present(ci(cs), mi(m)) = true;
        if ~unitsSet(mi(m))
            units(mi(m)) = d.ModalityUnits(m); kinds(mi(m)) = d.ModalityKinds(m); unitsSet(mi(m)) = true;
        elseif units(mi(m)) ~= d.ModalityUnits(m)
            notes(end + 1, 1) = "Signal " + mods(mi(m)) + " has units """ + units(mi(m)) + """ in one dataset and """ + ...
                d.ModalityUnits(m) + """ in dataset " + k + "; the first was kept (check they are comparable)."; %#ok<AGROW>
        end
    end
    fill = @(target, src) iff(isempty(src), target, src);
    if ~isempty(d.ChannelX), cx(ci) = fill(cx(ci), d.ChannelX); cy(ci) = fill(cy(ci), d.ChannelY); end
    if ~isempty(d.Hemisphere), hemi(ci) = d.Hemisphere; end
    if ~isempty(d.ChannelGroups)
        g = d.ChannelGroups; take = strlength(groups(ci)) == 0;
        groups(ci(take)) = g(take);
    end
    missing = setdiff(chan, d.ChannelNames);
    if ~isempty(missing)
        notes(end + 1, 1) = "Dataset " + names(k) + " lacks " + numel(missing) + " of the " + R + " channels (" + ...
            listSome(missing) + "); they are missing for its " + d.N + " trials."; %#ok<AGROW>
    end
    row = row + d.N;
end
chanSets = arrayfun(@(m) find(present(:, m)), (1:M)', 'UniformOutput', false);

% ---- trials ----------------------------------------------------------------------------
cols = strings(0, 1);
for k = 1:K
    cols = [cols; col(setdiff(string(datasets{k}.Trials.Properties.VariableNames)', cols, 'stable'))]; %#ok<AGROW>
end
trials = table();
for c = cols'
    parts = cell(K, 1);
    for k = 1:K
        d = datasets{k};
        if ismember(c, string(d.Trials.Properties.VariableNames))
            parts{k} = d.Trials.(c);
        else
            parts{k} = [];
        end
    end
    trials.(c) = joinColumn(parts, counts, c);
end
if strlength(opts.Source)
    trials.(opts.Source) = repelem(names, counts);
end
if isempty(cols) && strlength(opts.Source) == 0, trials = []; end
subject = firstRole(datasets, "Subject");
session = firstRole(datasets, "Session");

labels = struct();
for k = 1:K
    L = datasets{k}.ValueLabels;
    for i = 1:height(L)
        c = L.Column(i);
        if ~isfield(labels, c), labels.(c) = cell(0, 2); end
        v = L.Value(i);
        hit = find(cellfun(@(x) isequal(x, v), labels.(c)(:, 1)), 1);
        if isempty(hit)
            labels.(c)(end + 1, :) = {v, L.Label(i)};
        elseif string(labels.(c){hit, 2}) ~= L.Label(i)
            error('prismt:E_COMBINE_LABELS', 'Column %s: value %g means "%s" in one dataset and "%s" in dataset %d.', ...
                c, v, labels.(c){hit, 2}, L.Label(i), k);
        end
    end
end

uids = strings(0, 1);
if all(cellfun(@(d) numel(d.TrialUid) == d.N, datasets))
    uids = vertcat(cellfun(@(d) d.TrialUid, datasets(:), 'UniformOutput', false));
    uids = vertcat(uids{:});
    if numel(unique(uids)) < numel(uids), uids = repelem(names, counts) + "/" + uids; end
end
atlas = "";
a = cellfun(@(d) string(d.Atlas), datasets);
if all(a == a(1)) && strlength(a(1)) && R == datasets{1}.R, atlas = a(1); end
events = unique(cellfun(@(d) string(d.Event), datasets));
event = events(1);
if numel(events) > 1, notes(end + 1, 1) = "Time 0 is described differently (" + strjoin(events, ", ") + "); kept """ + event + """."; end
prov = struct('combined_from', {cellstr(names)}, 'sources', {cellfun(@(d) d.Uid, datasets, 'UniformOutput', false)}, ...
    'notes', {cellstr(notes)});
xy = {};
if any(~isnan(cx)), xy = {'ChannelX', cx, 'ChannelY', cy}; end
extra = {};
if any(strlength(hemi)), extra = [extra, {'Hemisphere', hemi}]; end
if any(strlength(groups)), extra = [extra, {'ChannelGroups', groups}]; end
if ~isempty(uids), extra = [extra, {'TrialUid', uids}]; end
args = [{'Times', times, 'Event', event, 'ChannelNames', chan}, xy, extra, ...
    {'Atlas', atlas, 'ModalityNames', mods, 'ModalityUnits', units, 'ModalityKinds', kinds, ...
    'ModalityChannels', chanSets, 'Subject', subject, 'Session', session, 'ValueLabels', labels, 'Provenance', prov}];
ds = prismt.makeDataset(X, trials, args{:});
if K > 1
    notes = ["Combined " + K + " datasets: " + N + " trials, " + R + " channels, " + M + " signal(s)."; notes];
end
end

function x = col(x)
% A column (setdiff's output orientation differs between releases).
x = x(:);
end

function v = iff(c, a, b)
if c, v = a; else, v = b; end
end

function s = listSome(names)
s = strjoin(names(1:min(end, 5)), ", ");
if numel(names) > 5, s = s + ", ..."; end
end

function idx = keepIndex(t, target)
[~, idx] = min(abs(t(:) - target(:)'), [], 1);
end

function Y = resampleTime(X, t, target)
[N, R, T, M] = size(X);
Z = reshape(permute(X, [3 1 2 4]), T, []);
Y = interp1(double(t), double(Z), double(target), 'linear');
Y = permute(reshape(single(Y), numel(target), N, R, M), [2 3 1 4]);
end

function v = joinColumn(parts, counts, name)
% Concatenate one trial column across datasets, with missing values where a dataset lacks it.
kinds = strings(numel(parts), 1);
for k = 1:numel(parts)
    p = parts{k};
    if isempty(p) && counts(k) > 0, kinds(k) = "none";
    elseif isnumeric(p) || islogical(p), kinds(k) = "numeric";
    else, kinds(k) = "text";
    end
end
if all(kinds == "none" | kinds == "numeric") && any(kinds == "numeric")
    v = nan(sum(counts), 1);
    row = 0;
    for k = 1:numel(parts)
        if kinds(k) == "numeric", v(row + (1:counts(k))) = double(parts{k}(:)); end
        row = row + counts(k);
    end
    if all(kinds == "numeric") && all(cellfun(@islogical, parts)), v = logical(v); end
    return
end
v = strings(sum(counts), 1);
v(:) = missing;
row = 0;
for k = 1:numel(parts)
    if kinds(k) ~= "none", v(row + (1:counts(k))) = string(parts{k}(:)); end
    row = row + counts(k);
end
if any(kinds == "numeric") && any(kinds == "text")
    warning('prismt:combine', 'Column %s is numbers in some datasets and text in others; it was joined as text.', name);
end
end

function r = firstRole(datasets, role)
vals = cellfun(@(d) string(d.(role)), datasets);
vals = vals(strlength(vals) > 0);
r = "";
if isempty(vals), return; end
r = vals(1);
if any(vals ~= r)
    error('prismt:E_COMBINE_ROLES', 'The %s column is called %s in the datasets; rename them to one name first.', ...
        lower(role), strjoin(unique(vals), " and "));
end
end
