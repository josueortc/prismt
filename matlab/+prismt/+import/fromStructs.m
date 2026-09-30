function p = fromStructs(S, kind, opts)
%FROMSTRUCTS Read processed_data / standardized_data (dataset_001, dataset_002, ...).
p = prismt.import.emptyParts();
if kind == "processed", D = S.processed_data; else, D = S.standardized_data; end
names = string(fieldnames(D));
sets = sort(names(startsWith(names, "dataset_")));
order = opts.AxisOrder;
if strlength(order) == 0
    order = "trials,time,channels";
    p.notes(end + 1) = "Signal arrays read as trials x time x channels, the layout written by the old " + ...
        "preprocessing scripts (set AxisOrder if yours differ).";
end
w = strtrim(split(lower(order), ","))';
[~, perm] = ismember(["trials", "channels", "time"], w);
for k = 1:numel(sets)
    d = D.(sets(k));
    for sig = ["dff", "zscore"]
        if isfield(d, sig) && ~isempty(d.(sig))
            p.signals.(sig){k, 1} = permute(double(d.(sig)), perm);
        end
    end
    n = size(p.signals.dff{k}, 1);
    for c = ["stim", "response", "label"]
        if isfield(d, c) && numel(d.(c)) == n
            p.trialCols.(c){k, 1} = double(d.(c)(:));
        elseif isfield(d, c) && numel(d.(c)) == 1
            p.trialCols.(c){k, 1} = repmat(double(d.(c)), n, 1);
        end
    end
    for c = ["phase", "mouse"]
        if isfield(d, c)
            v = d.(c);
            if iscell(v) || (isstring(v) && numel(v) == n) || (ischar(v) && size(v, 1) == n && n > 1)
                p.trialCols.(c){k, 1} = strtrim(string(v(:)));
            else
                p.sessionCols.(c)(k, 1) = strtrim(string(v));
            end
        end
    end
end
p.sessionNames = compose("dataset_%03d", (1:numel(sets))');
end
