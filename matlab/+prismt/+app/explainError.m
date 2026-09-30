function [message, title] = explainError(err)
%EXPLAINERROR Plain-language text for an error raised by an app action.
%   PRISMT's own errors already say what to do; anything else is shown with where it came
%   from so it can be reported.
title = "Something went wrong";
message = string(err.message);
id = string(err.identifier);
if startsWith(id, "prismt:")
    title = "PRISMT could not do this";
    return
end
if ~isempty(err.stack)
    s = err.stack(1);
    message = message + newline + newline + "(in " + s.name + ", line " + s.line + ")";
end
end
