function p = fromTable(S, opts)
%FROMTABLE Read a table T with one row per session (the widefield tableForModeling files).
p = prismt.import.emptyParts();
names = string(fieldnames(S));
T = S.(names(find(arrayfun(@(n) istable(S.(n)), names), 1)));
nS = height(T);
order = opts.AxisOrder;
if strlength(order) == 0
    order = "trials,channels,time";
    p.notes(end + 1) = "Signal arrays read as trials x channels x time, the layout of tableForModeling " + ...
        "(set AxisOrder if yours differ).";
end
vars = string(T.Properties.VariableNames);
nTrials = [];
% 1. per-session 3-D signals
for v = vars
    col = T.(v);
    if iscell(col) && all(cellfun(@(x) isnumeric(x) && ndims(x) == 3, col))
        p.signals.(matlab.lang.makeValidName(v)) = cellfun(@(x) toTrialsChannelsTime(x, order), col, 'UniformOutput', false);
        if isempty(nTrials), nTrials = cellfun(@(x) size(toTrialsChannelsTime(x, order), 1), col); end
    end
end
if isempty(fieldnames(p.signals))
    error('prismt:E_IMPORT_NO_SIGNAL', 'The table has no per-session 3-D signal (like dff: trials x channels x time).');
end
sigNames = string(fieldnames(p.signals));
T0 = size(p.signals.(sigNames(1)){1}, 3);
for v = vars
    key = matlab.lang.makeValidName(v);
    if isfield(p.signals, key), continue; end
    col = T.(v);
    if iscell(col) && all(cellfun(@(x) isnumeric(x) || islogical(x), col))
        sz = cellfun(@(x) size(x), col, 'UniformOutput', false);
        rows = cellfun(@(s) s(1), sz);
        ncol = cellfun(@(s) prod(s(2:end)), sz);
        if all(rows == nTrials(:)) && all(ncol == 1)
            p.trialCols.(key) = cellfun(@(x) double(x(:)), col, 'UniformOutput', false);
        elseif all(rows == nTrials(:)) && all(ncol == T0)
            p.behavior.(key) = cellfun(@(x) double(reshape(x, size(x, 1), [])), col, 'UniformOutput', false);
        elseif all(cellfun(@numel, col) == 1)
            p.sessionCols.(key) = cellfun(@double, col);
        else
            bad = find(rows ~= nTrials(:), 1);
            if ~isempty(bad) && all(ncol == 1)
                error('prismt:E_IMPORT_ROWS', 'Row %d: column %s has %d values but the signal has %d trials.', ...
                    bad, v, rows(bad), nTrials(bad));
            end
            p.notes(end + 1) = "Column " + v + " was skipped (its size does not match trials or time).";
        end
    elseif iscell(col) && all(cellfun(@(x) ischar(x) || isstring(x) || iscategorical(x), col))
        p.sessionCols.(key) = string(cellfun(@(x) char(string(x)), col, 'UniformOutput', false));
    elseif (isnumeric(col) || islogical(col)) && size(col, 1) == nS && size(col, 2) == 1
        p.sessionCols.(key) = double(col);
    elseif (isstring(col) || iscategorical(col) || iscellstr(col)) && numel(col) == nS
        p.sessionCols.(key) = string(col);
    elseif ischar(col) && size(col, 1) == nS
        p.sessionCols.(key) = strtrim(string(col));
    else
        p.notes(end + 1) = "Column " + v + " was skipped (unsupported type " + class(col) + ").";
    end
end
p = readMeta(p, S);
subj = matlab.lang.makeValidName(opts.Subject);
if isfield(p.sessionCols, subj), who = string(p.sessionCols.(subj)); else, who = repmat("s", nS, 1); end
p.sessionNames = compose("%s_r%03d", who, (1:nS)');
if isfield(p.sessionCols, "sessionNum")
    p.notes(end + 1) = "Each table row is one session (sessionNum is kept as a column).";
else
    p.notes(end + 1) = "Each table row is treated as one session.";
end
end

function X = toTrialsChannelsTime(x, order)
w = strtrim(split(lower(order), ","))';
[~, perm] = ismember(["trials", "channels", "time"], w);
if any(perm == 0)
    error('prismt:E_IMPORT_AXES', 'AxisOrder "%s" must name trials, channels and time.', order);
end
X = permute(x, perm);
end

function p = readMeta(p, S)
if ~isfield(S, 'meta') || ~isstruct(S.meta), return; end
m = S.meta;
if isfield(m, 'frameRateHz'), p.fs = double(m.frameRateHz); end
if isfield(m, 'windowSeconds'), p.t0 = double(m.windowSeconds(1)); end
if isfield(m, 'parcellation') && startsWith(string(m.parcellation), "Grids82")
    p.atlasHint = "grid82";
end
for f = string(fieldnames(m))'
    if endsWith(f, "Coding") && (ischar(m.(f)) || isstring(m.(f)))
        col = extractBefore(f, strlength(f) - 5);
        pairs = regexp(char(m.(f)), '(-?\d+(\.\d+)?)\s*=\s*([^,;]+)', 'tokens');
        lab = cell(numel(pairs), 2);
        for k = 1:numel(pairs)
            lab(k, :) = {str2double(pairs{k}{1}), strtrim(string(pairs{k}{end}))};
        end
        if ~isempty(lab), p.labels.(col) = lab; end
    end
end
p.notes(end + 1) = "Sampling rate, time window and value labels were taken from the file's meta struct.";
end
