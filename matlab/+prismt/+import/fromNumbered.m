function p = fromNumbered(S, opts)
%FROMNUMBERED Read files with numbered variables: dff_001, stim_001, ..., mice, phases.
p = prismt.import.emptyParts();
f = string(fieldnames(S));
ids = sort(str2double(extractAfter(f(~cellfun(@isempty, regexp(f, '^dff_\d+$', 'once'))), "dff_")));
order = opts.AxisOrder;
if strlength(order) == 0
    order = "trials,channels,time";
    p.notes(end + 1) = "Signal arrays read as trials x channels x time (set AxisOrder if yours differ).";
end
w = strtrim(split(lower(order), ","))';
[~, perm] = ismember(["trials", "channels", "time"], w);
for k = 1:numel(ids)
    tag = sprintf('%03d', ids(k));
    for sig = ["dff", "zscore"]
        v = sig + "_" + tag;
        if isfield(S, v), p.signals.(sig){k, 1} = permute(double(S.(v)), perm); end
    end
    for c = ["stim", "response"]
        v = c + "_" + tag;
        if isfield(S, v), p.trialCols.(c){k, 1} = double(S.(v)(:)); end
    end
end
for c = ["mice", "phases"]
    if isfield(S, c)
        v = S.(c);
        if ischar(v), v = cellstr(v); end
        name = extractBefore(c, strlength(c));   % mice -> mouse, phases -> phase
        if c == "mice", name = "mouse"; end
        p.sessionCols.(name) = strtrim(string(v(:)));
    end
end
p.sessionNames = compose("session_%03d", ids(:));
end
