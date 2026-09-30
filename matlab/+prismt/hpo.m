function run = hpo(cfg, opts)
%HPO Tune settings automatically on this computer, then retrain the best ones.
%
%   cfg.hpo = struct('n_trials', 20);  run = prismt.hpo(cfg)
%   Tried settings are compared on validation trials only; the best settings are then
%   retrained with several seeds and tested once. See hpo_summary.json in the run folder.
arguments
    cfg struct
    opts.RunsFolder (1, 1) string = ""
    opts.Name (1, 1) string = ""
    opts.Wait (1, 1) logical = false
    opts.Python (1, 1) string = ""
end
run = prismt.train(cfg, RunsFolder=opts.RunsFolder, Name=opts.Name, Wait=opts.Wait, Python=opts.Python, Kind="hpo");
end
