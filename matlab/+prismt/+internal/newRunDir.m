function runDir = newRunDir(cfg, runsFolder, name, kind)
%NEWRUNDIR A new, never-reused run folder: <runs>/<task>-<yyyyMMdd-HHmmss>-<id>[-name].
%   Named by time and a random id, never by settings, so two runs can never collide.
if strlength(runsFolder) == 0
    if isfield(cfg, 'output') && isfield(cfg.output, 'root')
        runsFolder = string(cfg.output.root);
    else
        runsFolder = fullfile(prismt.internal.projectFolder(), "runs");
    end
end
prefix = string(cfg.task);
if kind == "hpo", prefix = "hpo-" + prefix; end
id = extractBefore(prismt.internal.newUid(), 5);
stem = prefix + "-" + string(datetime('now', 'Format', 'yyyyMMdd-HHmmss')) + "-" + id;
if strlength(name), stem = stem + "-" + regexprep(name, '[^A-Za-z0-9_.-]+', '_'); end
runDir = fullfile(runsFolder, stem);
mkdir(runDir);
end
