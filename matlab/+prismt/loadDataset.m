function ds = loadDataset(file)
%LOADDATASET Read a PRISMT dataset written by prismt.writeDataset (or prismt synth).
%
%   ds = prismt.loadDataset(file) returns a prismt.Dataset.
arguments
    file (1, 1) string
end
if ~isfile(file)
    error('prismt:E_DATA_MISSING', 'The dataset file does not exist: %s', file);
end
try
    info = whos('-file', file);
catch
    error('prismt:E_DATA_UNREADABLE', '%s is not a readable MATLAB file.', file);
end
vars = string({info.name});
if ~ismember("prismt_format", vars)
    error('prismt:E_DATA_NOT_PRISMT', ['%s is not a PRISMT dataset (it contains: %s). ' ...
        'Convert it with prismt.importData or the Data tab (Import...).'], file, strjoin(vars(1:min(end, 8)), ', '));
end
S = load(file);
if ~strcmp(char(S.prismt_format(:)'), 'prismt.dataset')
    error('prismt:E_DATA_NOT_PRISMT', '%s has an unknown format tag.', file);
end
fmt = double(S.prismt_version);
if fmt > prismt.version().Formats.dataset
    error('prismt:E_DATA_NEWER_FORMAT', ['This dataset uses format version %d; this copy of PRISMT ' ...
        'reads up to version %d. Update PRISMT.'], fmt, prismt.version().Formats.dataset);
end
meta = jsondecode(native2unicode(reshape(S.meta_json, 1, []), 'UTF-8'));
shape = double(meta.shape(:)');

ds = prismt.Dataset();
ds.X = reshape(single(S.X), shape);
ds.Uid = textOr(field(meta, 'dataset_uid'), "");
ch = meta.channels;
ds.ChannelNames = string(ch.names);
ds.ChannelX = numOr(field(ch, 'x'));
ds.ChannelY = numOr(field(ch, 'y'));
ds.Hemisphere = strOr(field(ch, 'hemisphere'));
ds.Atlas = textOr(field(ch, 'atlas'), "");
if isfield(S, 'atlas_image')
    ds.AtlasImage = uint16(S.atlas_image);
end
mo = meta.modalities;
ds.ModalityNames = string(mo.names);
ds.ModalityUnits = strOr(field(mo, 'units'));
ds.ModalityKinds = strOr(field(mo, 'kinds'));
ds.ModalityChannels = channelSets(field(mo, 'channels'), shape);
ds.Times = double(meta.time.times_s(:));
ds.Event = textOr(field(meta.time, 'event'), "");

[ds.Trials, ds.ValueLabels] = trialsTable(meta.trials, shape(1));
roles = field(meta.trials, 'roles');
ds.Subject = textOr(field(roles, 'subject'), "");
ds.Session = textOr(field(roles, 'session'), "");
ds.TrialUid = strOr(field(meta.trials, 'uid'));
p = field(meta, 'provenance');
if isstruct(p)
    ds.Provenance = p;
end
end

function v = field(s, name)
if isstruct(s) && isfield(s, name)
    v = s.(name);
else
    v = [];
end
end

function v = numOr(x)
if isempty(x), v = zeros(0, 1); else, v = double(x(:)); end
end

function v = strOr(x)
if isempty(x), v = strings(0, 1); else, v = string(x(:)); end
end

function v = textOr(x, default)
if isempty(x) || (isnumeric(x) && all(isnan(x))), v = string(default); else, v = string(x); end
end

function sets = channelSets(x, shape)
% jsondecode turns [[1,2],[3,4]] into a matrix and [[1,2],[3]] into a cell array.
M = shape(4);
if isempty(x)
    sets = repmat({(1:shape(2))'}, M, 1);
elseif iscell(x)
    sets = cellfun(@(c) double(c(:)), x(:), 'UniformOutput', false);
elseif isvector(x) && M == 1
    sets = {double(x(:))};
else
    sets = num2cell(double(x), 2);
    sets = cellfun(@(c) c(:), sets(:), 'UniformOutput', false);
end
end

function [T, labels] = trialsTable(trials, N)
labels = table(strings(0, 1), zeros(0, 1), strings(0, 1), 'VariableNames', {'Column', 'Value', 'Label'});
cols = field(trials, 'columns');
if isempty(cols)
    T = table('Size', [N 0], 'VariableTypes', {});
    return
end
if isstruct(cols)
    cols = num2cell(cols);
end
values = cell(1, numel(cols));
names = strings(1, numel(cols));
for k = 1:numel(cols)
    c = cols{k};
    names(k) = string(c.name);
    raw = c.values;
    switch c.type
        case 'numeric'
            if iscell(raw)
                raw(cellfun(@isempty, raw)) = {NaN};
                raw = cell2mat(raw);
            end
            values{k} = double(raw(:));
            for lab = reshape(asCell(field(c, 'labels')), 1, [])
                labels(end + 1, :) = {names(k), double(lab{1}.value), string(lab{1}.label)}; %#ok<AGROW>
            end
        case 'bool'
            values{k} = logical(raw(:));
        otherwise
            if iscell(raw)
                s = strings(numel(raw), 1);
                for i = 1:numel(raw)
                    if isempty(raw{i}), s(i) = missing; else, s(i) = string(raw{i}); end
                end
            else
                s = string(raw(:));
            end
            cats = field(c, 'categories');
            if isempty(cats)
                values{k} = categorical(s);
            else
                values{k} = categorical(s, string(cats));
            end
    end
end
T = table(values{:}, 'VariableNames', cellstr(names));
end

function c = asCell(x)
if isempty(x), c = {}; elseif iscell(x), c = x; else, c = num2cell(x); end
end
