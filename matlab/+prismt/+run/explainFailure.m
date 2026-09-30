function err = explainFailure(runDir, exitCode)
%EXPLAINFAILURE A plain-language explanation of why a run stopped.
%   Uses the structured error Python wrote in status.json when there is one; otherwise
%   matches the end of the log against known problems (known_errors.json next to this file).
if nargin < 2, exitCode = []; end
err = struct('code', "E_UNKNOWN", 'title', "The run stopped unexpectedly", 'message', "", 'hint', ...
    "Open the log for details.", 'field', "", 'log_tail', "");
f = fullfile(runDir, "status.json");
if isfile(f)
    try
        s = jsondecode(fileread(f));
        if isfield(s, 'error') && isstruct(s.error)
            e = s.error;
            for k = ["code", "title", "message", "hint", "field"]
                if isfield(e, k), err.(k) = string(e.(k)); end
            end
        end
    catch
    end
end
tail = "";
for name = ["log.txt", "stdout.txt"]
    lf = fullfile(runDir, name);
    if isfile(lf)
        lines = splitlines(string(fileread(lf)));
        tail = tail + strjoin(lines(max(1, end - 40):end), newline) + newline;
    end
end
err.log_tail = tail;
if err.code ~= "E_UNKNOWN" && err.code ~= "E_INTERNAL", return; end
known = jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')), 'known_errors.json')));
for k = 1:numel(known)
    if ~isempty(regexp(tail, known(k).pattern, 'once'))
        err.code = string(known(k).code);
        err.title = string(known(k).title);
        err.hint = string(known(k).hint);
        err.message = string(regexp(tail, known(k).pattern, 'match', 'once'));
        return
    end
end
if ~isempty(exitCode)
    err.message = "Python stopped with exit code " + exitCode + ".";
end
end
