classdef SetupTab < handle
    %SETUPTAB Python environment, project folder and (optionally) a cluster profile.

    properties (SetAccess = private)
        App
        Lamp
        Status
        PythonPath
        Checks
        Log
        Buttons struct = struct()
        Project
        Cluster struct = struct()
    end
    properties (Constant)
        ClusterFields = ["host", "user", "remote_root", "partition", "account", "qos", "gpus", "cpus", ...
            "mem", "time", "modules", "conda_env", "runtime"]
        ClusterLabels = ["Login host", "User name", "Folder on the cluster", "Partition", "Account", "QOS", ...
            "GPUs", "CPUs", "Memory", "Time limit", "Modules to load", "Conda environment", "Runtime"]
        ClusterHelp = ["e.g. mccleary.ycrc.yale.edu", "your cluster user name", "job folders are copied here", ...
            "e.g. gpu", "billing account, if your cluster needs one", "", "0 for CPU only", "", "e.g. 32G", ...
            "hh:mm:ss", "e.g. miniconda", "created once by setup_env.sh", "conda or apptainer"]
    end

    methods
        function t = SetupTab(app, parent)
            t.App = app;
            g = prismt.app.ui.grid(parent, {'fit', 'fit', '1x'}, {'1x', '1x'});
            h = prismt.app.ui.heading(g, "Set up PRISMT on this computer");
            h.Layout.Column = [1 2];
            n = prismt.app.ui.text(g, "PRISMT trains its models in Python. It needs a Python environment " + ...
                "with PyTorch; it can find an existing one or create it for you (about 2 GB, 5-15 minutes). " + ...
                "This is needed once per computer.");
            n.Layout.Column = [1 2];

            % --- Python environment
            p = prismt.app.ui.panel(g, "Python environment");
            pg = prismt.app.ui.grid(p, {'fit', 'fit', 'fit', '1x', 90}, {22, '1x'});
            t.Lamp = uilamp(pg, 'Color', [0.6 0.6 0.6]);
            t.Status = uilabel(pg, 'Text', "Not checked yet", 'FontWeight', 'bold', 'WordWrap', 'on');
            t.PythonPath = prismt.app.ui.note(pg, "");
            t.PythonPath.Layout.Column = [1 2];
            b = uigridlayout(pg, [1 4], 'Padding', 0, 'ColumnSpacing', 6, 'RowHeight', {'fit'});
            b.Layout.Column = [1 2];
            b.BackgroundColor = pg.BackgroundColor;
            t.Buttons.find = prismt.app.ui.button(b, "Find automatically", @() app.safely(@t.findPython), ...
                'Tooltip', "Look for a Python environment that has PRISMT's packages");
            t.Buttons.choose = prismt.app.ui.button(b, "Choose Python...", @() app.safely(@t.choosePython), ...
                'Tooltip', "Pick the python (python.exe on Windows) of an environment yourself");
            t.Buttons.check = prismt.app.ui.button(b, "Check again", @() app.safely(@t.checkAgain));
            t.Buttons.create = prismt.app.ui.button(b, "Create environment", @() app.safely(@t.createEnvironment), ...
                'Tooltip', "Create a new 'prismt' environment from environment.yml (uses conda, or downloads micromamba)");
            t.Checks = uitable(pg, 'ColumnName', {'Part', 'Result', 'What to do'}, 'RowName', {}, ...
                'ColumnWidth', {80, 300, 'auto'}, 'Data', cell(0, 3));
            t.Checks.Layout.Column = [1 2];
            t.Log = uitextarea(pg, 'Editable', 'off', 'Visible', 'off', 'FontName', 'Menlo', 'FontSize', 10);
            t.Log.Layout.Column = [1 2];
            p.Layout.Row = 3; p.Layout.Column = 1;

            % --- Project folder and cluster
            right = uigridlayout(g, [2 1], 'RowHeight', {'fit', '1x'}, 'Padding', 0);
            right.BackgroundColor = g.BackgroundColor;
            right.Layout.Row = 3; right.Layout.Column = 2;
            p = prismt.app.ui.panel(right, "Project folder");
            pg = prismt.app.ui.grid(p, {'fit', 'fit'}, {'1x', 'fit', 'fit'});
            t.Project = uilabel(pg, 'Text', "", 'WordWrap', 'on');
            prismt.app.ui.button(pg, "Change...", @() app.safely(@t.changeProject));
            prismt.app.ui.button(pg, "Show", @() app.safely(@() prismt.app.ui.openFolder(prismt.internal.projectFolder())));
            nn = prismt.app.ui.note(pg, "Datasets, runs, cluster job folders and exported scripts are kept here " + ...
                "(in datasets/, runs/, cluster/ and scripts/).");
            nn.Layout.Column = [1 3];

            p = prismt.app.ui.panel(right, "Cluster (optional)");
            pg = prismt.app.ui.grid(p, [repmat({22}, 1, numel(t.ClusterFields)), {'fit', 'fit'}], {130, '1x'}, ...
                'RowSpacing', 4, 'Scrollable', 'on');
            prof = t.clusterProfile();
            for k = 1:numel(t.ClusterFields)
                f = t.ClusterFields(k);
                uilabel(pg, 'Text', t.ClusterLabels(k), 'Tooltip', t.ClusterHelp(k));
                v = "";
                if isfield(prof, f), v = string(prof.(f)); end
                if f == "runtime"
                    t.Cluster.(f) = uidropdown(pg, 'Items', ["conda", "apptainer"], 'Value', ternary(v == "apptainer", "apptainer", "conda"));
                else
                    t.Cluster.(f) = prismt.app.ui.placeholder(uieditfield(pg, 'Value', v, 'Tooltip', t.ClusterHelp(k)), t.ClusterHelp(k));
                end
            end
            b = uigridlayout(pg, [1 3], 'Padding', 0, 'ColumnSpacing', 6, 'RowHeight', {'fit'});
            b.Layout.Column = [1 2];
            b.BackgroundColor = pg.BackgroundColor;
            prismt.app.ui.button(b, "Save", @() app.safely(@t.saveCluster));
            prismt.app.ui.button(b, "Import...", @() app.safely(@t.importCluster), 'Tooltip', "Load a profile shared by your lab (.json)");
            prismt.app.ui.button(b, "Export...", @() app.safely(@t.exportCluster), 'Tooltip', "Save this profile to share it (.json)");
            nn = prismt.app.ui.note(pg, "Only needed to train on a SLURM cluster. The Run tab writes a job folder " + ...
                "and the exact commands to copy it up, submit it and fetch the results; nothing is sent automatically.");
            nn.Layout.Column = [1 2];

            app.listen('EnvChanged', @t.refresh);
        end

        function refresh(t)
            c = t.App.Controller;
            s = prismt.plot.style();
            t.Project.Text = prismt.internal.projectFolder();
            job = c.EnvJob;
            if ~isempty(job)
                t.Lamp.Color = s.warning;
                t.Status.Text = "Creating the environment... (this window stays usable)";
            elseif isempty(c.Doctor)
                t.Lamp.Color = [0.6 0.6 0.6];
                t.Status.Text = "No Python environment yet: press Find automatically or Create environment.";
            elseif c.Doctor.ok
                t.Lamp.Color = s.good;
                t.Status.Text = "Ready: " + describe(c.Doctor);
            else
                t.Lamp.Color = s.critical;
                t.Status.Text = "Not usable: " + firstProblem(c.Doctor);
            end
            if ~isempty(c.Doctor), t.PythonPath.Text = "Python: " + string(c.Doctor.python_path); end
            t.Checks.Data = checkRows(c.Doctor);
            busy = ~isempty(job);
            for f = string(fieldnames(t.Buttons))', t.Buttons.(f).Enable = onoff(~busy); end
        end

        function tick(t)
            c = t.App.Controller;
            if isempty(c.EnvJob), return; end
            t.Log.Visible = 'on';
            t.Log.Value = tailText(c.EnvJob.Log, 40);
            if c.pollEnvironment()
                if c.pythonReady()
                    t.Log.Value = [t.Log.Value; ""; "Done: the environment works."];
                else
                    t.Log.Value = [t.Log.Value; ""; "The environment was not created. The messages above say why."];
                end
            end
        end

        function findPython(t)
            h = t.App.Dialogs.busy("Looking for a Python environment with PRISMT's packages...");
            cleanup = onCleanup(@() delete(h));
            if ~t.App.Controller.findPython()
                t.Status.Text = "No usable Python was found. Press Create environment to make one.";
            end
        end

        function choosePython(t)
            if ispc, filter = {'python.exe', 'python.exe'}; else, filter = {'python*', 'python'}; end
            f = t.App.Dialogs.getFile(filter, "Choose the Python of an environment");
            if strlength(f) == 0, return; end
            h = t.App.Dialogs.busy("Checking " + f + "...");
            cleanup = onCleanup(@() delete(h));
            t.App.Controller.setPython(f);
        end

        function checkAgain(t)
            c = t.App.Controller;
            if strlength(c.Python) == 0, t.findPython(); return; end
            h = t.App.Dialogs.busy("Checking " + c.Python + "...");
            cleanup = onCleanup(@() delete(h));
            c.setPython(c.Python);
        end

        function createEnvironment(t)
            try
                t.App.Controller.createEnvironment(false);
            catch err
                if err.identifier ~= "prismt:E_ENV_NO_CONDA", rethrow(err); end
                a = t.App.Dialogs.ask("Conda is not installed. PRISMT can download micromamba (a single " + ...
                    "15 MB program from github.com/mamba-org) into its own folder and use it; no administrator " + ...
                    "rights are needed.", "Create environment", ["Download micromamba", "Cancel"]);
                if a ~= "Download micromamba", return; end
                t.App.Controller.createEnvironment(true);
            end
            t.Log.Visible = 'on';
            t.Log.Value = "Starting...";
        end

        function changeProject(t)
            f = t.App.Dialogs.getFolder("Choose the PRISMT project folder", prismt.internal.projectFolder());
            if strlength(f) == 0, return; end
            prismt.internal.projectFolder(f);
            t.refresh();
        end

        function prof = clusterProfile(~)
            prof = prismt.internal.settings("cluster_profile");
            if isempty(prof), prof = struct('remote_root', "~/prismt_jobs", 'partition', "gpu", 'gpus', 1, ...
                    'cpus', 4, 'mem', "32G", 'time', "04:00:00", 'modules', "miniconda", 'conda_env', "prismt", ...
                    'runtime', "conda"); end
        end

        function prof = readCluster(t)
            %READCLUSTER The profile as typed (numbers for GPUs and CPUs; empty fields left out).
            prof = struct();
            for f = t.ClusterFields
                v = strtrim(string(t.Cluster.(f).Value));
                if strlength(v) == 0, continue; end
                if ismember(f, ["gpus", "cpus"])
                    n = str2double(v);
                    if isnan(n) || n < 0 || n ~= round(n)
                        error('prismt:cluster', '%s must be a whole number (0 or more), not "%s".', ...
                            t.ClusterLabels(t.ClusterFields == f), v);
                    end
                    prof.(f) = n;
                else
                    prof.(f) = char(v);
                end
            end
        end

        function saveCluster(t)
            prismt.internal.settings("cluster_profile", t.readCluster());
        end

        function importCluster(t)
            f = t.App.Dialogs.getFile({'*.json', 'Cluster profile (*.json)'}, "Import a cluster profile");
            if strlength(f) == 0, return; end
            prof = jsondecode(fileread(f));
            for k = string(fieldnames(prof))'
                if isfield(t.Cluster, k), t.Cluster.(k).Value = string(prof.(k)); end
            end
            t.saveCluster();
        end

        function exportCluster(t)
            f = t.App.Dialogs.putFile({'*.json', 'Cluster profile (*.json)'}, "Export the cluster profile", "cluster_profile.json");
            if strlength(f) == 0, return; end
            prismt.internal.atomicWrite(f, jsonencode(t.readCluster(), 'PrettyPrint', true));
        end
    end
end

function text = describe(rep)
parts = strings(0, 1);
if isfield(rep, 'prismt'), parts(end + 1) = "PRISMT " + rep.prismt.version; end
if isfield(rep, 'torch') && isfield(rep.torch, 'version'), parts(end + 1) = "PyTorch " + rep.torch.version; end
switch string(rep.device)
    case "cuda", parts(end + 1) = "NVIDIA GPU";
    case "mps", parts(end + 1) = "Apple GPU";
    otherwise, parts(end + 1) = "CPU only (training will be slower)";
end
text = strjoin(parts, "  ·  ");
end

function text = firstProblem(rep)
bad = rep.checks(~[rep.checks.ok]);
text = "";
if ~isempty(bad), text = string(bad(1).message); end
end

function rows = checkRows(rep)
rows = cell(0, 3);
if isempty(rep), return; end
for k = 1:numel(rep.checks)
    ck = rep.checks(k);
    mark = "OK  ";
    if ~ck.ok, mark = "FAIL  "; end
    rows(end + 1, :) = {char(ck.name), char(mark + ck.message), char(ck.hint)}; %#ok<AGROW>
end
w = rep.warnings;
for k = 1:numel(w)
    rows(end + 1, :) = {'warning', char(w(k).message), char(w(k).hint)}; %#ok<AGROW>
end
end

function lines = tailText(file, n)
lines = "";
if ~isfile(file), return; end
try
    txt = splitlines(string(fileread(file)));
    txt = regexprep(txt, '.*\r', '');     % progress bars rewrite the line with carriage returns
    txt = txt(strlength(strtrim(txt)) > 0);
    lines = txt(max(1, end - n + 1):end);
    if isempty(lines), lines = ""; end
catch
end
end

function s = onoff(tf)
if tf, s = 'on'; else, s = 'off'; end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end

