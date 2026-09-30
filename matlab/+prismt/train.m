function run = train(cfg, opts)
%TRAIN Start training on this computer, in the background.
%
%   run = prismt.train(cfg)                      % returns at once; see run.status()
%   run = prismt.train(cfg, Wait=true)           % blocks until finished, printing progress
%   R = prismt.loadResults(run.RunDir)           % then plot with prismt.plot.*
%
%   The run gets its own folder, <RunsFolder>/<task>-<date>-<time>-<id>, which holds its
%   settings, progress (status.json, history.csv), results (metrics.json, results.mat,
%   predictions.csv) and the trained model. MATLAB stays usable while it trains.
arguments
    cfg struct
    opts.RunsFolder (1, 1) string = ""
    opts.Name (1, 1) string = ""
    opts.Wait (1, 1) logical = false
    opts.Python (1, 1) string = ""
    opts.Kind (1, 1) string = "train"
end
runDir = prismt.internal.newRunDir(cfg, opts.RunsFolder, opts.Name, opts.Kind);
run = prismt.run.LocalRun.start(cfg, runDir, Kind=opts.Kind, Python=opts.Python);
if opts.Wait
    last = "";
    while run.isActive()
        s = run.status();
        if isfield(s, 'message') && string(s.message) ~= last
            fprintf('%s\n', s.message);
            last = string(s.message);
        end
        pause(2);
    end
    s = run.status();
    if string(s.state) == "failed"
        e = s.error;
        error(sprintf('prismt:%s', char(e.code)), '%s: %s %s (run folder: %s)', e.title, e.message, e.hint, runDir);
    end
    m = fullfile(runDir, "metrics.json");
    if isfile(m)
        lines = jsondecode(fileread(m)).summary_lines;
        fprintf('%s\n', string(lines));
    end
end
end
