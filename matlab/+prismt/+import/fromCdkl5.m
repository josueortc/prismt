function p = fromCdkl5(S, opts)
%FROMCDKL5 Read CDKL5 structs (one element per animal, continuous regions x time recordings)
%and cut them into windows of opts.Window samples (step opts.Stride, default = Window).
p = prismt.import.emptyParts();
stride = opts.Stride;
if isempty(stride), stride = opts.Window; end
k = 0;
for group = ["wt", "mut"]
    name = "cdkl5_m_" + group + "_struct";
    if ~isfield(S, name), continue; end
    A = S.(name);
    if istable(A), A = table2struct(A); end
    for i = 1:numel(A)
        a = A(i);
        data = [];
        for fld = ["allen_parcels", "dff", "data", "calcium"]
            if isfield(a, fld), data = a.(fld); break; end
        end
        if iscell(data), data = data{1}; end
        if isempty(data), continue; end
        k = k + 1;
        % Continuous recordings are regions x time; windowing wants time x regions.
        W = prismt.import.windowContinuous(double(data'), opts.Window, stride);
        p.signals.dff{k, 1} = W;
        mouse = sprintf('%s_%d', group, i);
        for fld = ["mouse", "mouse_id", "animal_id"]
            if isfield(a, fld), v = a.(fld); if iscell(v), v = v{1}; end, mouse = char(string(v)); break; end
        end
        p.sessionCols.mouse(k, 1) = string(mouse);
        p.sessionCols.genotype(k, 1) = group;
    end
end
p.sessionNames = compose("%s_rec", p.sessionCols.mouse);
p.notes(end + 1) = sprintf("Continuous recordings cut into windows of %d samples every %d samples.", opts.Window, stride);
end
