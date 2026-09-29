function issues = addIssue(issues, level, code, message, hint, field)
%ADDISSUE Append one issue. level is "error", "warning" or "info".
if nargin < 5, hint = ""; end
if nargin < 6, field = ""; end
issues(end + 1) = struct('Level', string(level), 'Code', string(code), ...
    'Message', string(message), 'Hint', string(hint), 'Field', string(field));
end
