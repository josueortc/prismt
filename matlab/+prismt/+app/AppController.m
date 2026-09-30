classdef AppController < handle
    %APPCONTROLLER Everything the PRISMT app knows and does, without any graphics.
    %   The app's tabs call these methods and redraw when an event fires, so the same
    %   actions can be scripted or tested without a window.

    properties (SetAccess = private)
        Python string = ""
        Doctor = []                 % last prismt.env.doctor report
        Dataset = []                % prismt.Dataset
        DatasetFile string = ""
        ImportNotes string = strings(0, 1)
        Config struct = struct()    % the run settings being edited (partial; Python fills the rest)
        Check = []                  % last prismt.check report
        Run = []                    % prismt.run.LocalRun being watched
        EnvJob = []                 % environment creation in progress
        Schema struct = struct()
    end

    events
        EnvChanged
        DataChanged
        ConfigChanged
        CheckChanged
        RunChanged
    end

    methods
        function c = AppController()
            schema = jsondecode(fileread(fullfile(prismt.internal.repoRoot(), 'src', 'prismt', 'resources', ...
                'config_schema.json')));
            schema.fields = uniform(schema.fields, ["path", "type", "label", "help", "default", "choices", ...
                "min", "max", "advanced", "nullable"]);
            schema.presets = uniform(schema.presets, ["name", "label", "description", "values"]);
            for k = 1:numel(schema.presets)
                schema.presets(k).values = uniform(schema.presets(k).values, ["path", "value"]);
            end
            schema.sections = uniform(schema.sections, ["key", "label"]);
            c.Schema = schema;
            c.Config = struct('task', 'classify', 'preset', 'quick');
        end

        % ---- Python environment -------------------------------------------------------
        function ok = findPython(c)
            [py, rep] = prismt.env.findPython();
            ok = strlength(py) > 0;
            if ok, c.Python = py; c.Doctor = rep; else, c.Doctor = []; end
            notify(c, 'EnvChanged');
        end

        function ok = setPython(c, path)
            rep = prismt.env.doctor(string(path));
            c.Doctor = rep;
            ok = rep.ok;
            if ok
                c.Python = string(path);
                prismt.internal.settings("python", char(path));
            end
            notify(c, 'EnvChanged');
        end

        function job = createEnvironment(c, allowDownload)
            job = prismt.env.createEnvironment(Download=allowDownload);
            c.EnvJob = job;
            notify(c, 'EnvChanged');
        end

        function done = pollEnvironment(c)
            %POLLENVIRONMENT True once environment creation has finished (then checks it).
            done = isempty(c.EnvJob) || ~c.EnvJob.Process.isRunning();
            if done && ~isempty(c.EnvJob)
                py = c.EnvJob.Python;
                c.EnvJob = [];
                if isfile(py), c.setPython(py); else, notify(c, 'EnvChanged'); end
            end
        end

        function tf = pythonReady(c)
            tf = ~isempty(c.Doctor) && c.Doctor.ok;
        end

        % ---- Data ---------------------------------------------------------------------
        function openDataset(c, file)
            c.Dataset = prismt.loadDataset(file);
            c.DatasetFile = string(file);
            c.ImportNotes = "Opened " + file;
            c.datasetChanged();
        end

        function importFile(c, file, opts)
            %IMPORTFILE Convert a lab file and save it as a PRISMT dataset in the project folder.
            if nargin < 3, opts = {}; end
            [ds, rep] = prismt.importData(file, opts{:});
            c.acceptImport(ds, rep, file);
        end

        function acceptImport(c, ds, rep, file)
            %ACCEPTIMPORT Save an imported dataset (from prismt.importData) and use it.
            [~, stem] = fileparts(file);
            out = fullfile(prismt.internal.projectFolder(), "datasets", stem + "_prismt.mat");
            prismt.writeDataset(ds, out);
            c.Dataset = prismt.loadDataset(out);
            c.DatasetFile = out;
            c.ImportNotes = [string(rep.notes(:)); "Saved as " + out];
            c.datasetChanged();
        end

        function info = inspectFile(c, file) %#ok<INUSL>
            %INSPECTFILE What a lab file contains (signals, behavior columns, timing), without importing.
            [~, info] = prismt.importData(file, Inspect=true);
        end

        function [ds, rep] = previewImport(~, file, opts)
            %PREVIEWIMPORT Import in memory with options (name-value cell), to check before saving.
            [ds, rep] = prismt.importData(file, opts{:});
        end

        function notes = addDatasets(c, files)
            %ADDDATASETS Combine the current dataset with other PRISMT datasets (channels,
            %signals and trial columns matched by name) and use the result.
            if isempty(c.Dataset), error('prismt:app', 'Open a dataset first.'); end
            files = string(files);
            parts = [{c.Dataset}; arrayfun(@(f) prismt.loadDataset(f), files(:), 'UniformOutput', false)];
            [~, names] = arrayfun(@fileparts, [c.DatasetFile; files(:)]);
            names = erase(names, "_prismt");
            [ds, notes] = prismt.combineDatasets(parts, Source="source_dataset", Names=matlab.lang.makeUniqueStrings(names));
            out = fullfile(prismt.internal.projectFolder(), "datasets", names(1) + "_combined_prismt.mat");
            out = uniqueFile(out);
            prismt.writeDataset(ds, out);
            c.Dataset = prismt.loadDataset(out);
            c.DatasetFile = out;
            c.ImportNotes = [notes(:); "Saved as " + out];
            c.datasetChanged();
        end

        function setChannelGroups(c, groups)
            %SETCHANNELGROUPS Give each channel a group (e.g. area, side, sensor) and save the dataset.
            ds = c.Dataset;
            groups = strtrim(string(groups(:)));
            if numel(groups) ~= ds.R, error('prismt:app', 'Give one group per channel (%d).', ds.R); end
            ds.ChannelGroups = groups;
            prismt.writeDataset(ds, c.DatasetFile);
            c.Dataset = prismt.loadDataset(c.DatasetFile);
            c.ImportNotes = [c.ImportNotes(:); "Channel groups saved: " + strjoin(unique(groups(strlength(groups) > 0))', ", ")];
            if isfield(c.Config, 'selection') && isfield(c.Config.selection, 'channel_groups')
                c.Config.selection = rmfield(c.Config.selection, 'channel_groups');
            end
            c.Check = [];
            notify(c, 'DataChanged');
            notify(c, 'ConfigChanged');
        end

        function useDemo(c, profile)
            if nargin < 2, profile = "fast"; end
            [ds, truth] = prismt.demo.makeSyntheticDataset(Profile=profile);
            out = fullfile(prismt.internal.projectFolder(), "datasets", "demo_" + profile + "_prismt.mat");
            prismt.writeDataset(ds, out);
            c.Dataset = prismt.loadDataset(out);
            c.DatasetFile = out;
            c.ImportNotes = ["Demo data (" + profile + "): a stimulus response in channels " + ...
                strjoin(ds.ChannelNames(truth.StimulusChannels), ", ") + " (classify stim), a learning effect in " + ...
                strjoin(ds.ChannelNames(truth.LearningChannels), ", ") + " (classify phase)."; "Saved as " + out];
            c.datasetChanged();
        end

        % ---- Task and settings --------------------------------------------------------
        function setTask(c, task)
            %SETTASK "classify", "mae" or "finetune" (classify starting from an autoencoder run).
            task = string(task);
            if task == "finetune"
                c.Config.task = 'classify';
                if ~isfield(c.Config, 'model') || ~isfield(c.Config.model, 'init_from')
                    c.Config.model.init_from = '';
                end
            else
                c.Config.task = char(task);
                if isfield(c.Config, 'model') && isfield(c.Config.model, 'init_from')
                    c.Config.model = rmfield(c.Config.model, 'init_from');
                end
            end
            c.configChanged();
        end

        function t = task(c)
            t = string(c.Config.task);
            if t == "classify" && isfield(c.Config, 'model') && isfield(c.Config.model, 'init_from'), t = "finetune"; end
        end

        function setLabel(c, column)
            c.Config.labels.column = char(column);
            c.Config.labels.classes = {};
            c.configChanged();
        end

        function setClasses(c, values, names)
            %SETCLASSES Values of the label column to compare; values sharing a name are merged.
            values = string(values);
            if nargin < 3, names = values; end
            names = string(names);
            u = unique(names, 'stable');
            classes = cell(1, numel(u));
            for k = 1:numel(u)
                classes{k} = struct('name', char(u(k)), 'values', {cellstr(values(names == u(k)))});
            end
            c.Config.labels.classes = classes;
            c.configChanged();
        end

        function setFilter(c, column, values)
            %SETFILTER Keep only trials whose column is one of values (empty values removes it).
            f = c.filters();
            keep = cellfun(@(x) ~strcmp(x.column, column), f);
            f = f(keep);
            if ~isempty(values)
                f{end + 1} = struct('column', char(column), 'op', 'in', 'values', {cellstr(string(values))});
            end
            c.Config.selection.filters = f;
            c.configChanged();
        end

        function f = filters(c)
            f = {};
            if isfield(c.Config, 'selection') && isfield(c.Config.selection, 'filters')
                f = c.Config.selection.filters;
                if isstruct(f), f = num2cell(f); end
            end
        end

        function setValue(c, path, value)
            %SETVALUE Set any setting by its dotted path, e.g. "train.lr" (empty = use default).
            parts = split(string(path), ".");
            cfg = c.Config;
            if isempty(value) || (isstring(value) && all(strlength(value) == 0))
                cfg = removePath(cfg, parts);
            else
                cfg = setPath(cfg, parts, value);
            end
            c.Config = cfg;
            c.configChanged();
        end

        function v = value(c, path, default)
            v = default;
            node = c.Config;
            for p = split(string(path), ".")'
                if isstruct(node) && isfield(node, p), node = node.(p); else, return; end
            end
            v = node;
        end

        function setPreset(c, name)
            c.Config.preset = char(name);
            c.configChanged();
        end

        function cfg = runConfig(c, runsFolder)
            %RUNCONFIG The settings as they will be sent to Python.
            cfg = c.Config;
            cfg.dataset = struct('path', char(c.DatasetFile));
            if nargin > 1, cfg.output.root = char(runsFolder); end
        end

        function rep = runCheck(c, timing)
            %RUNCHECK Ask Python what would happen (split, class counts, size, time).
            if nargin < 2, timing = false; end
            rep = prismt.check(c.runConfig(), Timing=timing, Python=c.Python);
            c.Check = rep;
            notify(c, 'CheckChanged');
        end

        function keep = keptTrials(c)
            %KEPTTRIALS Trials that pass the trial filters (logical, one per dataset trial).
            keep = [];
            if isempty(c.Dataset), return; end
            ds = c.Dataset;
            keep = true(ds.N, 1);
            for f = c.filters()
                ff = f{1};
                keep = keep & ismember(prismt.app.ui.labels(ds, string(ff.column)), string(ff.values));
            end
        end

        function T = classCounts(c)
            %CLASSCOUNTS Trials and subjects for each value of the label column (after filters).
            T = table(strings(0, 1), zeros(0, 1), zeros(0, 1), 'VariableNames', {'Value', 'Trials', 'Subjects'});
            col = string(c.value("labels.column", ""));
            if isempty(c.Dataset) || strlength(col) == 0 || ~ismember(col, string(c.Dataset.Trials.Properties.VariableNames))
                return
            end
            ds = c.Dataset;
            keep = c.keptTrials();
            g = prismt.app.ui.labels(ds, col);
            vals = unique(g(keep & ~ismissing(g)), 'stable');
            vals = sort(vals);
            n = arrayfun(@(v) nnz(keep & g == v), vals);
            subjects = zeros(size(vals));
            if strlength(ds.Subject)
                s = string(ds.Trials.(ds.Subject));
                subjects = arrayfun(@(v) numel(unique(s(keep & g == v))), vals);
            end
            T = table(vals(:), n(:), subjects(:), 'VariableNames', {'Value', 'Trials', 'Subjects'});
        end

        % ---- Running ------------------------------------------------------------------
        function run = startRun(c, kind, name)
            if nargin < 2, kind = "train"; end
            if nargin < 3, name = ""; end
            runs = fullfile(prismt.internal.projectFolder(), "runs");
            c.Run = prismt.train(c.runConfig(runs), RunsFolder=runs, Name=name, Kind=kind, Python=c.Python);
            run = c.Run;
            notify(c, 'RunChanged');
        end

        function stopRun(c)
            if ~isempty(c.Run), c.Run.stop(); end
            notify(c, 'RunChanged');
        end

        function watch(c, runDir)
            c.Run = prismt.run.LocalRun.attach(runDir);
            notify(c, 'RunChanged');
        end

        function job = makeClusterJob(c, profile, remoteDataset, mode)
            if nargin < 4, mode = "train"; end
            job = prismt.makeClusterJob(c.runConfig(), profile, RemoteDataset=remoteDataset, Mode=mode, Python=c.Python);
            prismt.internal.settings("cluster_profile", profile);
        end

        function file = exportScript(c, file, cfg)
            %EXPORTSCRIPT A MATLAB script that repeats a run without the app.
            %   exportScript(file) uses the current settings; exportScript(file, cfg) any settings.
            if nargin < 3, cfg = c.runConfig(); end
            if isfield(cfg, 'output') && isfield(cfg.output, 'root'), cfg.output = rmfield(cfg.output, 'root'); end
            if isfield(cfg, 'output') && isempty(fieldnames(cfg.output)), cfg = rmfield(cfg, 'output'); end
            kind = "train";
            if isfield(cfg, 'hpo') && ~isempty(cfg.hpo), kind = "hpo"; end
            lines = [
                "%% PRISMT run, exported from the app on " + string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm'))
                "% Runs the same analysis without the app. Change any setting below, then run the script."
                "% Every setting is explained in prismt.defaultConfig's help and on the app's tooltips."
                "addpath('" + replace(fullfile(prismt.internal.repoRoot(), "matlab"), "'", "''") + "');"
                ""
                prismt.internal.toCode(cfg, "cfg")
                ""
                "report = prismt.check(cfg);          % what will happen: trials per class, split, size"
                "if ~report.ok, error('%s %s', report.error.message, report.error.hint); end"
                "run = prismt." + kind + "(cfg, Wait=true);   % trains in the background; Wait prints progress"
                ""
                "R = prismt.loadResults(run.RunDir);"
                "disp(join(string(R.Metrics.summary_lines), newline))"
                "figure; prismt.plot.scoreVsBaselines(gca, R);"
                "figure; prismt.plot.learningCurves(gca, R);"];
            prismt.internal.atomicWrite(file, strjoin(lines, newline) + newline);
        end

        function T = finishedRuns(c, task)
            %FINISHEDRUNS Finished runs of one task ("mae" for fine-tuning), newest first.
            T = prismt.listRuns();
            if isempty(T) || height(T) == 0, return; end
            T = T(T.State == "finished" & T.Task == task, :);
        end
    end

    methods (Access = private)
        function datasetChanged(c)
            if isfield(c.Config, 'selection')
                for f = ["modalities", "channel_groups", "channels"]
                    if isfield(c.Config.selection, f), c.Config.selection = rmfield(c.Config.selection, f); end
                end
            end
            ds = c.Dataset;
            cols = string(ds.Trials.Properties.VariableNames);
            label = string(c.value("labels.column", ""));
            if ~ismember(label, cols)
                pick = intersect(["phase", "stim", "condition", "genotype", "response"], cols, 'stable');
                if isempty(pick)
                    % otherwise the first column with a few values that is not subject or session
                    for v = setdiff(prismt.app.ui.columnChoices(ds), [ds.Subject, ds.Session], 'stable')
                        n = numel(unique(prismt.app.ui.labels(ds, v)));
                        if n >= 2 && n <= 10, pick = v; break, end
                    end
                end
                if ~isempty(pick), c.Config.labels.column = char(pick(1)); else, c.Config.labels.column = ''; end
                c.Config.labels.classes = {};
            end
            c.Config.selection.filters = {};
            c.Check = [];
            notify(c, 'DataChanged');
            notify(c, 'ConfigChanged');
        end

        function configChanged(c)
            c.Check = [];
            notify(c, 'ConfigChanged');
        end
    end
end

function f = uniqueFile(f)
% f, or f with _2, _3... added so an existing file is never overwritten.
[d, n, e] = fileparts(f);
k = 1;
while isfile(f)
    k = k + 1;
    f = fullfile(d, n + "_" + k + e);
end
end

function out = uniform(list, names)
% jsondecode gives a cell array when objects have different keys; make a struct array.
if isstruct(list), list = num2cell(list); end
out = repmat(cell2struct(cell(numel(names), 1), cellstr(names), 1), numel(list), 1);
for k = 1:numel(list)
    for n = names
        if isfield(list{k}, n), out(k).(n) = list{k}.(n); end
    end
end
end

function s = setPath(s, parts, value)
if isscalar(parts)
    s.(parts(1)) = value;
    return
end
if ~isfield(s, parts(1)) || ~isstruct(s.(parts(1))), s.(parts(1)) = struct(); end
s.(parts(1)) = setPath(s.(parts(1)), parts(2:end), value);
end

function s = removePath(s, parts)
if ~isfield(s, parts(1)), return; end
if isscalar(parts)
    s = rmfield(s, parts(1));
else
    s.(parts(1)) = removePath(s.(parts(1)), parts(2:end));
end
end
