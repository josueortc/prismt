function report = doctor(python, opts)
%DOCTOR Check that a Python can run PRISMT ("python -m prismt doctor").
%
%   report = prismt.env.doctor(pythonPath) returns the decoded report: report.ok, report.checks
%   (name, ok, message, hint), report.device ("cuda", "mps" or "cpu") and report.warnings.
%   A Python that cannot even start gives ok = false with an explanation instead of an error.
arguments
    python (1, 1) string
    opts.Timeout (1, 1) double = 240
end
if ~isfile(python)
    report = failed(python, "No Python at " + python + ".", "Find or create the PRISMT environment.");
    return
end
[report, status, log] = prismt.run.runPython("doctor", Python=python, Timeout=opts.Timeout);
if isempty(report)
    tail = strtrim(extractAfter(log, max(0, strlength(log) - 600)));
    hint = "This Python does not have PRISMT's packages. Use Create environment on the Setup tab.";
    if contains(log, "No module named 'prismt'")
        hint = "PRISMT's code was not found by this Python; reinstall with Create environment.";
    elseif status == -1
        hint = "Python took too long to start; try again.";
    end
    report = failed(python, "This Python could not run PRISMT: " + tail, hint);
end
report.python_path = python;
v = prismt.version().Package;
if isfield(report, 'prismt') && isfield(report.prismt, 'version') && string(report.prismt.version) ~= v
    report.warnings = [report.warnings(:); struct('code', "W_ENV_VERSION", 'message', ...
        "Python runs PRISMT " + report.prismt.version + " but MATLAB has " + v + ".", 'hint', "")];
end
end

function r = failed(python, message, hint)
r = struct('ok', false, 'python_path', python, 'device', "", 'warnings', struct('code', {}, 'message', {}, 'hint', {}), ...
    'checks', struct('name', "python", 'ok', false, 'message', message, 'hint', hint));
end
