function [S, file] = readSource(source)
%READSOURCE Load a .mat file (any version) into a struct, or accept a struct/table directly.
file = "";
if isstruct(source)
    S = source;
elseif istable(source)
    S = struct('T', source);
else
    file = string(source);
    if ~isfile(file)
        error('prismt:E_DATA_MISSING', 'The file does not exist: %s', file);
    end
    S = load(file);
end
end
