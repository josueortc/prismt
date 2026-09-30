function R = loadResults(folder)
%LOADRESULTS Everything a finished (or running) PRISMT run wrote, in MATLAB form.
%
%   R = prismt.loadResults(runFolder)    % also accepts a cluster job folder (uses results/)
%
%   R.Status, R.Config, R.Metrics   decoded JSON (R.Metrics.summary_lines: plain summary)
%   R.History                       table, one row per epoch (and fold)
%   R.Mat                           results.mat: predictions, probabilities, reconstructions,
%                                   R² per channel, embeddings, channel/time information
%   R.Predictions, R.Splits         tables (per test trial / every trial's split)
%   R.Tuning                        hpo_summary.json for tuning runs
folder = string(folder);
if isfolder(fullfile(folder, "results")) && isfile(fullfile(folder, "job.env"))
    folder = fullfile(folder, "results");
end
if ~isfolder(folder)
    error('prismt:E_RUN_MISSING', 'No run folder at %s.', folder);
end
R = struct('Folder', folder, 'Status', readJson(folder, "status.json"), 'Config', readJson(folder, "config.json"), ...
    'Metrics', readJson(folder, "metrics.json"), 'Tuning', readJson(folder, "hpo_summary.json"), ...
    'History', readCsv(folder, "history.csv"), 'Predictions', readCsv(folder, "predictions.csv"), ...
    'Splits', readCsv(folder, "splits.csv"), 'Mat', struct());
if isempty(R.Metrics) && ~isempty(R.Tuning)
    kept = string(R.Tuning.kept);
    if isfolder(kept)
        inner = prismt.loadResults(kept);
        [R.Metrics, R.Mat, R.Predictions, R.History] = deal(inner.Metrics, inner.Mat, inner.Predictions, inner.History);
    end
end
m = fullfile(folder, "results.mat");
if isfile(m)
    R.Mat = load(m);
    for f = ["channel_names", "modality_names", "class_names", "mask_names", "baseline_names", "subject", "session", "split"]
        if isfield(R.Mat, f)
            R.Mat.(f) = strtrim(string(R.Mat.(f)));
            R.Mat.(f) = R.Mat.(f)(:);
        end
    end
end
end

function s = readJson(folder, name)
f = fullfile(folder, name);
s = [];
if isfile(f)
    try, s = jsondecode(fileread(f)); catch, end
end
end

function t = readCsv(folder, name)
f = fullfile(folder, name);
t = table();
if isfile(f)
    try, t = readtable(f, 'TextType', 'string', 'VariableNamingRule', 'preserve'); catch, end
end
end
