function kind = detect(S)
%DETECT Which known layout a loaded .mat struct has.
f = string(fieldnames(S));
if ismember("prismt_format", f)
    kind = "prismt";
elseif any(arrayfun(@(n) istable(S.(n)), f))
    kind = "table";
elseif ismember("processed_data", f)
    kind = "processed";
elseif ismember("standardized_data", f)
    kind = "standardized";
elseif any(startsWith(f, "cdkl5_"))
    kind = "cdkl5";
elseif any(~cellfun(@isempty, regexp(f, '^dff_\d+$', 'once')))
    kind = "numbered";
else
    kind = "unknown";
end
end
