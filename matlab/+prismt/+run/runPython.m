function [out, status, log] = runPython(args, opts)
%RUNPYTHON Run "python -m prismt <args> --json" and return the decoded JSON.
%
%   [out, status, log] = prismt.run.runPython(["validate", file])
%
%   out     struct decoded from the command's JSON output ([] if there was none)
%   status  exit code (0 ok, 2 problem with data or settings, 4 environment, ...)
%   log     what the command wrote to stderr (its log)
%   Errors are not thrown: callers decide what to show. Waits at most Timeout seconds.
arguments
    args (1, :) string
    opts.Python (1, 1) string = ""
    opts.Timeout (1, 1) double = 600
    opts.Json (1, 1) logical = true
end
py = opts.Python;
if strlength(py) == 0
    py = prismt.env.python();
end
tmp = string(tempname);
stdoutFile = tmp + ".out";
stderrFile = tmp + ".err";
cleanup = onCleanup(@() cellfun(@deleteIfFile, {stdoutFile, stderrFile}));
if opts.Json
    args = [args, "--json"];
end
full = ["-m", "prismt", args];
if usejava('jvm')
    env = prismt.run.Process.cleanEnvironment(py, struct());
    pb = java.lang.ProcessBuilder(cellstr([py, full]));
    jenv = pb.environment();
    keys = cell(jenv.keySet().toArray());
    for k = 1:numel(keys)
        if ~isKey(env, keys{k}), jenv.remove(keys{k}); end
    end
    names = env.keys();
    for k = 1:numel(names)
        jenv.put(names{k}, env(names{k}));
    end
    pb.redirectOutput(java.io.File(char(stdoutFile)));
    pb.redirectError(java.io.File(char(stderrFile)));
    proc = pb.start();
    proc.getOutputStream().close();
    finished = proc.waitFor(opts.Timeout, java.util.concurrent.TimeUnit.SECONDS);
    if ~finished
        proc.destroyForcibly();
        out = [];
        status = -1;
        log = "Timed out after " + opts.Timeout + " s: " + strjoin(full, " ");
        return
    end
    status = double(proc.exitValue());
else
    q = @(a) prismt.internal.shellQuote(a, ternary(ispc, "cmd", "posix"));
    cmd = strjoin(arrayfun(q, [py, full]), " ") + " > " + q(stdoutFile) + " 2> " + q(stderrFile);
    if ~ispc
        cmd = "env -u PYTHONHOME PYTHONPATH=" + q(fullfile(prismt.internal.repoRoot(), 'src')) + ...
            " PYTHONNOUSERSITE=1 " + cmd;
    end
    status = system(cmd);
end
text = readText(stdoutFile);
log = readText(stderrFile);
out = [];
if opts.Json && strlength(strtrim(text)) > 0
    lines = splitlines(strtrim(text));
    try
        out = jsondecode(char(lines(end)));
    catch
        out = [];
        log = log + newline + text;
    end
elseif ~opts.Json
    out = text;
end
end

function t = readText(f)
if isfile(f), t = string(fileread(f)); else, t = ""; end
end

function deleteIfFile(f)
if isfile(f), delete(f); end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
