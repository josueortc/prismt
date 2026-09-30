function lines = toCode(value, name)
%TOCODE MATLAB statements that rebuild a value, e.g. for exported scripts.
%
%   lines = prismt.internal.toCode(cfg, "cfg")   % ["cfg = struct();"; "cfg.task = 'classify';"; ...]
%   Scalar structs become one assignment per field; everything else a single literal.
if nargin < 2, name = "x"; end
name = string(name);
if isstruct(value) && isscalar(value)
    lines = name + " = struct();";
    lines = [lines; fields(value, name)];
else
    lines = name + " = " + literal(value) + ";";
end
end

function lines = fields(s, prefix)
lines = strings(0, 1);
for f = string(fieldnames(s))'
    v = s.(f);
    if isstruct(v) && isscalar(v) && ~isempty(fieldnames(v))
        lines = [lines; fields(v, prefix + "." + f)]; %#ok<AGROW>
    else
        lines(end + 1, 1) = prefix + "." + f + " = " + literal(v) + ";"; %#ok<AGROW>
    end
end
end

function t = literal(v)
if ischar(v)
    t = "'" + replace(string(v), "'", "''") + "'";
elseif isstring(v)
    if isscalar(v)
        t = """" + replace(v, """", """""") + """";
    else
        t = "[" + strjoin(arrayfun(@literal, v(:)'), ", ") + "]";
    end
elseif islogical(v)
    if isscalar(v)
        t = string(mat2str(v));
    else
        t = string(mat2str(v));
    end
elseif isnumeric(v)
    if isempty(v)
        t = "[]";
    elseif isscalar(v)
        t = string(num2str(double(v), 15));
    else
        t = string(mat2str(double(v), 15));
    end
elseif iscell(v)
    if isempty(v), t = "{}"; return; end
    parts = cellfun(@literal, v, 'UniformOutput', false);
    t = "{" + strjoin([parts{:}], ", ") + "}";
elseif isstruct(v)
    if isempty(fieldnames(v)), t = "struct()"; return; end
    if ~isscalar(v)
        parts = arrayfun(@literal, v, 'UniformOutput', false);
        t = "[" + strjoin([parts{:}], ", ") + "]";
        return
    end
    args = strings(0, 1);
    for f = string(fieldnames(v))'
        x = literal(v.(f));
        if iscell(v.(f)), x = "{" + x + "}"; end    % struct() would otherwise make an array
        args(end + 1) = "'" + f + "', " + x; %#ok<AGROW>
    end
    t = "struct(" + strjoin(args, ", ") + ")";
else
    error('prismt:toCode', 'Cannot write a %s as code.', class(v));
end
end
