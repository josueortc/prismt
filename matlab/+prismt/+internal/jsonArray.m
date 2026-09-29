function c = jsonArray(v)
%JSONARRAY Wrap a vector so jsonencode writes a JSON array, even with one element.
%   jsonencode(5) gives 5 but jsonencode(prismt.internal.jsonArray(5)) gives [5].
%   Missing strings and NaN become null.
if isempty(v)
    c = {};
elseif iscell(v)
    c = reshape(v, 1, []);
elseif isstring(v) || ischar(v) || iscategorical(v)
    s = reshape(string(v), 1, []);
    c = cellstr(s);
    c(ismissing(s)) = {NaN};
elseif islogical(v)
    c = num2cell(reshape(v, 1, []));
elseif isnumeric(v)
    c = num2cell(double(reshape(v, 1, [])));
else
    error('prismt:internal:jsonArray', 'Cannot write a %s as a JSON array.', class(v));
end
end
