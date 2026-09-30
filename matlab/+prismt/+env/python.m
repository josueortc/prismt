function py = python(newPath)
%PYTHON The Python executable PRISMT uses (and remembers).
%
%   py = prismt.env.python()          % the saved choice, or the first working candidate
%   prismt.env.python("/path/to/python")  % check it with doctor and remember it
%
%   Errors with a plain explanation if no working PRISMT Python can be found.
if nargin == 1
    report = prismt.env.doctor(string(newPath));
    if ~report.ok
        error('prismt:E_ENV_PYTHON', 'This Python cannot run PRISMT: %s', report.checks(find(~[report.checks.ok], 1)).message);
    end
    prismt.internal.settings("python", char(newPath));
    py = string(newPath);
    return
end
saved = prismt.internal.settings("python");
if ~isempty(saved) && isfile(saved)
    py = string(saved);
    return
end
if ~isempty(getenv('PRISMT_PYTHON')) && isfile(getenv('PRISMT_PYTHON'))
    py = string(getenv('PRISMT_PYTHON'));
    return
end
[py, ~] = prismt.env.findPython();
if strlength(py) == 0
    error('prismt:E_ENV_PYTHON', ['No Python environment for PRISMT was found. Open the app''s Setup tab ' ...
        '(run_prismt_gui) and click "Create environment", or run prismt.env.createEnvironment().']);
end
end
