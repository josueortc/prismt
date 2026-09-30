function report = check(cfg, opts)
%CHECK Check run settings against their dataset without training (python -m prismt check).
%
%   report = prismt.check(cfg)               % selection, class counts, split plan, size
%   report = prismt.check(cfg, Timing=true)  % also measures a few steps to estimate run time
%
%   report.ok is false when the settings cannot work; report.error then says why and how to
%   fix it (report.error.field names the setting). report.warnings lists things to look at.
arguments
    cfg struct
    opts.Timing (1, 1) logical = false
    opts.Python (1, 1) string = ""
end
f = string(tempname) + ".json";
cleanup = onCleanup(@() delete(f));
prismt.internal.atomicWrite(f, jsonencode(cfg));
args = ["check", "--config", f];
if opts.Timing, args(end + 1) = "--timing"; end
[report, status, log] = prismt.run.runPython(args, Python=opts.Python, Timeout=600);
if isempty(report)
    report = struct('ok', false, 'error', struct('code', "E_INTERNAL", 'title', "The check could not run", ...
        'message', log, 'hint', "", 'field', ""));
end
report.exit_code = status;
end
