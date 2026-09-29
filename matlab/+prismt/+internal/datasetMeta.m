function meta = datasetMeta(ds)
%DATASETMETA The meta_json description of a dataset, ready for jsonencode.
%   Every list is wrapped with jsonArray so it stays a JSON array even with one element,
%   and NaN is used where JSON needs null. The format is specified in docs/data-format.md.
[N, R, T, M] = size(ds.X);
v = prismt.version();
arr = @prismt.internal.jsonArray;

meta.format = 'prismt.dataset';
meta.version = double(v.Formats.dataset);
meta.dataset_uid = char(ds.Uid);
meta.created = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
meta.created_by = sprintf('prismt-matlab %s (MATLAB %s)', v.Package, version('-release'));
meta.shape = arr([N R T M]);

meta.channels = struct('names', {arr(ds.ChannelNames)}, 'x', {orNull(ds.ChannelX)}, ...
    'y', {orNull(ds.ChannelY)}, 'hemisphere', {orNull(ds.Hemisphere)}, 'atlas', {textOrNull(ds.Atlas)});

chan = cell(1, M);
for m = 1:M
    chan{m} = arr(ds.ModalityChannels{m});
end
meta.modalities = struct('names', {arr(ds.ModalityNames)}, 'units', {arr(ds.ModalityUnits)}, ...
    'kinds', {arr(ds.ModalityKinds)}, 'channels', {chan});

meta.time = struct('times_s', {arr(ds.Times)}, 'event', char(ds.Event));
if T > 1
    dt = diff(ds.Times);
    if max(abs(dt - median(dt))) <= 1e-6 * max(1, abs(median(dt)))
        meta.time.fs_hz = 1 / median(dt);
        meta.time.t0_s = ds.Times(1);
    end
end

names = string(ds.Trials.Properties.VariableNames);
columns = cell(1, numel(names));
for c = 1:numel(names)
    columns{c} = columnMeta(names(c), ds.Trials.(names(c)), ds.ValueLabels);
end
meta.trials = struct('columns', {columns}, ...
    'roles', struct('subject', {textOrNull(ds.Subject)}, 'session', {textOrNull(ds.Session)}), ...
    'uid', {orNull(ds.TrialUid)});

meta.orientation_probes = probes(ds.X, [N R T M]);
meta.provenance = ds.Provenance;
end

function col = columnMeta(name, values, labels)
arr = @prismt.internal.jsonArray;
col = struct('name', char(name));
if islogical(values)
    col.type = 'bool';
    col.values = arr(values);
elseif isnumeric(values)
    col.type = 'numeric';
    col.values = arr(double(values));
    rows = labels(labels.Column == name, :);
    if height(rows) > 0
        lab = cell(1, height(rows));
        for k = 1:height(rows)
            lab{k} = struct('value', rows.Value(k), 'label', char(rows.Label(k)));
        end
        col.labels = lab;
    end
else
    col.type = 'categorical';
    if iscategorical(values)
        col.values = arr(string(values));
        col.categories = arr(string(categories(values)));
    else
        s = string(values);
        s(strlength(s) == 0) = missing;
        col.values = arr(s);
    end
end
end

function p = probes(X, sz)
% Values at a few fixed positions, so Python can prove it restored the dimensions.
n = numel(X);
k = unique(round(linspace(1, n, 8)));
finite = find(isfinite(X));
if ~isempty(finite)
    k = unique([k, finite(1), finite(end), finite(ceil(end / 2))]);
end
p = cell(1, numel(k));
for i = 1:numel(k)
    [a, b, c, d] = ind2sub(sz, k(i));
    p{i} = struct('index', {prismt.internal.jsonArray([a b c d] - 1)}, 'value', double(X(k(i))));
end
end

function v = orNull(x)
if isempty(x)
    v = NaN;
else
    v = prismt.internal.jsonArray(x);
end
end

function v = textOrNull(s)
if strlength(s) == 0
    v = NaN;
else
    v = char(s);
end
end
