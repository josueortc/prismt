function throwIfErrors(issues, title)
%THROWIFERRORS Raise one error that lists every error-level issue, with its hint.
%   The identifier is prismt:<code of the first error>, so callers and tests can check it.
if nargin < 2, title = "PRISMT could not continue"; end
if isempty(issues), return; end
errs = issues(string({issues.Level}) == "error");
if isempty(errs), return; end
lines = strings(numel(errs), 1);
for k = 1:numel(errs)
    lines(k) = "  - " + errs(k).Message;
    if strlength(errs(k).Hint) > 0
        lines(k) = lines(k) + newline + "    " + errs(k).Hint;
    end
end
msg = string(title) + ":" + newline + strjoin(lines, newline);
throwAsCaller(MException("prismt:" + errs(1).Code, "%s", msg));
end
