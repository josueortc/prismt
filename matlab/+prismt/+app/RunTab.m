classdef RunTab < handle
    %RUNTAB Start and follow a run on this computer, or write a job folder for a cluster.

    properties (SetAccess = private)
        App
        Empty
        Content
        Where
        Local struct = struct()
        Cluster struct = struct()
        Checks
        LastState string = ""
        LastRows double = -1
        Job = []
    end

    methods
        function t = RunTab(app, parent)
            t.App = app;
            outer = uigridlayout(parent, [1 1], 'Padding', 0);
            outer.BackgroundColor = prismt.plot.style().surface;
            t.Empty = prismt.app.ui.text(outer, "Load data first (Data tab), then choose what to learn (Task tab).");
            t.Empty.HorizontalAlignment = 'center'; t.Empty.VerticalAlignment = 'center';
            t.Content = prismt.app.ui.grid(outer, {'fit', '1x'}, {'1x', 380});
            t.Content.Layout.Row = 1; t.Content.Layout.Column = 1;

            top = uigridlayout(t.Content, [1 4], 'Padding', 0, 'ColumnWidth', {'fit', 260, 'fit', '1x'}, 'RowHeight', {26});
            top.BackgroundColor = t.Content.BackgroundColor;
            top.Layout.Column = [1 2];
            prismt.app.ui.heading(top, "Run");
            t.Where = uidropdown(top, 'Items', ["On this computer", "On a cluster (SLURM)"], 'ItemsData', ["local", "cluster"], ...
                'ValueChangedFcn', @(~, ~) t.refresh());
            prismt.app.ui.button(top, "Export as MATLAB script...", @() app.safely(@t.exportScript), ...
                'Tooltip', "A script that repeats this run without the app (to keep with your analysis)");

            main = uigridlayout(t.Content, [1 1], 'Padding', 0);
            main.BackgroundColor = t.Content.BackgroundColor;
            t.buildLocal(main);
            t.buildCluster(main);
            t.Checks = prismt.app.ChecksPanel(app, t.Content);
            t.Checks.Panel.Layout.Row = 2; t.Checks.Panel.Layout.Column = 2;
            app.listen('RunChanged', @t.refresh);
            app.listen('DataChanged', @t.refresh);
            app.listen('ConfigChanged', @t.refreshButtons);
            app.listen('CheckChanged', @t.refreshButtons);
        end

        function refresh(t)
            c = t.App.Controller;
            has = ~isempty(c.Dataset) || ~isempty(c.Run);
            t.Empty.Visible = onoff(~has);
            t.Content.Visible = onoff(has);
            local = t.Where.Value == "local";
            t.Local.panel.Visible = onoff(local);
            t.Cluster.panel.Visible = onoff(~local);
            t.Checks.refresh();
            t.refreshButtons();
            t.LastRows = -1;
            t.tick();
        end

        function refreshButtons(t)
            c = t.App.Controller;
            active = ~isempty(c.Run) && c.Run.isActive();
            ok = ~t.Checks.hasErrors() && ~isempty(c.Dataset);
            t.Local.start.Enable = onoff(ok && ~active);
            t.Local.stop.Enable = onoff(active && ~isempty(c.Run.Process));
            t.Cluster.create.Enable = onoff(~isempty(c.Dataset) && c.pythonReady());
            tuning = isfield(c.Config, 'hpo');
            t.Local.start.Text = ternary(tuning, "Start tuning", "Start training");
        end

        function start(t)
            %START Check the settings (if not yet done) and start a run in the background.
            c = t.App.Controller;
            if isempty(c.Check), t.Checks.runCheck(); end
            if t.Checks.hasErrors()
                t.App.Dialogs.alert("Fix the problems marked ✖ under Checks first.", "Cannot start yet");
                return
            end
            kind = "train";
            if isfield(c.Config, 'hpo'), kind = "hpo"; end
            c.startRun(kind, strtrim(string(t.Local.name.Value)));
            t.LastState = "";
        end

        function stop(t)
            a = t.App.Dialogs.ask("Stop after the current step? The best model so far is kept and evaluated " + ...
                "(this can take a minute).", "Stop the run", ["Stop", "Cancel"]);
            if a ~= "Stop", return; end
            h = t.App.Dialogs.busy("Stopping...");
            cleanup = onCleanup(@() delete(h));
            t.App.Controller.stopRun();
        end

        function tick(t)
            %TICK Update progress, curves and log from the run folder (called every second).
            c = t.App.Controller;
            run = c.Run;
            if isempty(run)
                t.Local.status.Text = "No run yet. Check the settings, then press Start. MATLAB stays usable " + ...
                    "while the model trains, and the run continues if you close the app.";
                t.Local.banner.Visible = 'off';
                return
            end
            s = run.status();
            state = string(s.state);
            t.Local.folder.Text = "Run folder: " + run.RunDir;
            t.Local.status.Text = progressText(s);
            h = run.history();
            if height(h) ~= t.LastRows
                t.LastRows = height(h);
                prismt.plot.learningCurves(clearAxes(t.Local.curves), struct('History', h));
            end
            t.Local.log.Value = tailLog(run.RunDir, 30);
            if state ~= t.LastState
                t.LastState = state;
                t.showOutcome(s);
                t.refreshButtons();
                if ismember(state, ["finished", "failed", "cancelled"]) && isfield(t.App.Tabs, "Results")
                    t.App.Tabs.Results.refreshRuns();
                end
            end
        end

        function job = createJob(t)
            c = t.App.Controller;
            prof = t.App.show("Setup").readCluster();
            t.App.show("Run");
            remote = strtrim(string(t.Cluster.remote.Value));
            mode = "train";
            if isfield(c.Config, 'hpo'), mode = "hpo"; end
            h = t.App.Dialogs.busy("Writing the job folder...");
            cleanup = onCleanup(@() delete(h));
            job = c.makeClusterJob(prof, remote, mode);
            t.Job = job;
            t.Cluster.readme.Value = splitlines(string(job.Readme));
            t.Cluster.folder.Text = "Job folder: " + job.Folder;
            t.Cluster.show.Enable = 'on';
        end

        function file = exportScript(t)
            c = t.App.Controller;
            file = t.App.Dialogs.putFile({'*.m', 'MATLAB script (*.m)'}, "Export as MATLAB script", ...
                fullfile(prismt.internal.projectFolder(), "scripts", "prismt_" + c.task() + "_run.m"));
            if strlength(file) == 0, return; end
            c.exportScript(file);
            if ~t.App.Dialogs.Headless, edit(char(file)); end
        end
    end

    methods (Access = private)
        function buildLocal(t, parent)
            p = prismt.app.ui.panel(parent, "On this computer");
            p.Layout.Row = 1; p.Layout.Column = 1;
            t.Local.panel = p;
            g = prismt.app.ui.grid(p, {26, 36, 70, 18, '1x', 110}, {'fit', 200, 'fit', 'fit', '1x'});
            uilabel(g, 'Text', 'Run name');
            t.Local.name = prismt.app.ui.placeholder(uieditfield(g, 'Tooltip', ...
                "Optional; added to the run folder's name (runs are never overwritten)"), "optional");
            t.Local.start = prismt.app.ui.primary(g, "Start training", @() t.App.safely(@t.start));
            t.Local.stop = prismt.app.ui.button(g, "Stop", @() t.App.safely(@t.stop), ...
                'Tooltip', "Stop after the current step and keep the best model so far");
            uilabel(g, 'Text', '');
            t.Local.status = prismt.app.ui.text(g, "", 'FontWeight', 'bold');
            t.Local.status.Layout.Column = [1 5];
            b = uigridlayout(g, [2 4], 'Padding', 0, 'ColumnWidth', {'fit', 'fit', 'fit', '1x'}, 'RowHeight', {36, 26}, ...
                'RowSpacing', 4);
            b.BackgroundColor = g.BackgroundColor;
            b.Layout.Column = [1 5];
            t.Local.banner = b;
            t.Local.message = prismt.app.ui.text(b, "");
            t.Local.message.Layout.Column = [1 4];
            t.Local.open = prismt.app.ui.button(b, "Open results", @() t.App.safely(@t.openResults));
            t.Local.details = prismt.app.ui.button(b, "Technical details", @() t.App.safely(@t.showDetails));
            prismt.app.ui.button(b, "Show run folder", @() t.App.safely(@() prismt.app.ui.openFolder(t.App.Controller.Run.RunDir)));
            t.Local.folder = prismt.app.ui.note(g, "");
            t.Local.folder.Layout.Column = [1 5];
            t.Local.curves = uiaxes(g);
            t.Local.curves.Layout.Column = [1 5];
            t.Local.log = uitextarea(g, 'Editable', 'off', 'FontName', 'Menlo', 'FontSize', 10);
            t.Local.log.Layout.Column = [1 5];
        end

        function buildCluster(t, parent)
            p = prismt.app.ui.panel(parent, "On a cluster (SLURM)");
            p.Layout.Row = 1; p.Layout.Column = 1;
            t.Cluster.panel = p;
            g = prismt.app.ui.grid(p, {50, 26, 18, 20, '1x', 26}, {'fit', '1x', 'fit'});
            n = prismt.app.ui.text(g, "PRISMT writes a job folder with your settings, a copy of the code and " + ...
                "the SLURM scripts, and the exact commands to run. You copy it to the cluster yourself (this works " + ...
                "with Duo / two-factor login). The cluster settings are on the Setup tab.");
            n.Layout.Column = [1 3];
            uilabel(g, 'Text', 'Dataset on the cluster');
            t.Cluster.remote = prismt.app.ui.placeholder(uieditfield(g, 'Tooltip', ...
                "Where the dataset file will be on the cluster. Empty: it is copied inside the job folder."), ...
                "e.g. ~/prismt_data/mydata_prismt.mat (empty: copy it with the job)");
            t.Cluster.create = prismt.app.ui.primary(g, "Create job folder", @() t.App.safely(@t.createJob));
            t.Cluster.folder = prismt.app.ui.note(g, "");
            t.Cluster.folder.Layout.Row = 3; t.Cluster.folder.Layout.Column = [1 3];
            h = uilabel(g, 'Text', 'Commands (copy and paste them in a terminal, one step at a time):', 'FontWeight', 'bold');
            h.Layout.Row = 4; h.Layout.Column = [1 3];
            t.Cluster.readme = uitextarea(g, 'Editable', 'off', 'FontName', 'Menlo', 'FontSize', 11, ...
                'Value', "Press Create job folder; the commands for your cluster appear here.");
            t.Cluster.readme.Layout.Row = 5; t.Cluster.readme.Layout.Column = [1 3];
            b = uigridlayout(g, [1 3], 'Padding', 0, 'ColumnWidth', {'fit', 'fit', '1x'}, 'RowHeight', {'1x'});
            b.BackgroundColor = g.BackgroundColor;
            b.Layout.Row = 6; b.Layout.Column = [1 3];
            t.Cluster.show = prismt.app.ui.button(b, "Show job folder", @() t.App.safely(@() prismt.app.ui.openFolder(t.Job.Folder)), ...
                'Enable', 'off');
            prismt.app.ui.button(b, "I downloaded the results...", @() t.App.safely(@t.downloaded), ...
                'Tooltip', "Open the results of a job folder whose results/ folder you copied back");
        end

        function showOutcome(t, s)
            state = string(s.state);
            t.Local.banner.Visible = onoff(ismember(state, ["finished", "failed", "cancelled"]));
            t.Local.open.Visible = onoff(state == "finished");
            t.Local.details.Visible = onoff(state ~= "finished");
            switch state
                case "finished"
                    m = fullfile(t.App.Controller.Run.RunDir, "metrics.json");
                    txt = "Done.";
                    if isfile(m)
                        try
                            lines = string(jsondecode(fileread(m)).summary_lines);
                            txt = "Done. " + strjoin(lines(1:min(end, 2)), " ");
                        catch
                        end
                    end
                    prismt.app.ui.setBanner(t.Local.message, "good", txt);
                case {"failed", "cancelled"}
                    e = s.error;
                    txt = ternary(state == "cancelled", "Cancelled", "Error") + ": " + string(e.title);
                    if isfield(e, 'message') && strlength(string(e.message)), txt = txt + ". " + string(e.message); end
                    if isfield(e, 'hint') && strlength(string(e.hint)), txt = txt + newline + "What to do: " + string(e.hint); end
                    prismt.app.ui.setBanner(t.Local.message, "critical", txt);
            end
        end

        function openResults(t)
            r = t.App.show("Results");
            r.openRun(t.App.Controller.Run.RunDir);
        end

        function showDetails(t)
            d = t.App.Controller.Run.RunDir;
            f = fullfile(d, "log.txt");
            if ~isfile(f), f = fullfile(d, "stdout.txt"); end
            if isfile(f), edit(char(f)); end
        end

        function downloaded(t)
            f = t.App.Dialogs.getFolder("Choose the job folder (the one with results/ inside)", ...
                fullfile(prismt.internal.projectFolder(), "cluster"));
            if strlength(f) == 0, return; end
            r = t.App.show("Results");
            r.openRun(f);
        end
    end
end

function text = progressText(s)
state = string(s.state);
switch state
    case "queued", text = "Starting Python...";
    case "preparing", text = "Preparing: " + string(s.message);
    case "training"
        text = "Training: " + string(s.message);
        if isfield(s, 'epoch') && isfield(s, 'epochs') && isfield(s, 'n_folds') && ~isempty(s.epochs)
            frac = ((double(s.fold) - 1) + double(s.epoch) / double(s.epochs)) / double(s.n_folds);
            text = text + sprintf("  (%.0f%%)", 100 * frac);
        elseif isfield(s, 'trials_done') && isfield(s, 'n_trials')
            text = text + sprintf("  (%.0f%%)", 100 * double(s.trials_done) / double(s.n_trials));
        end
        if isfield(s, 'eta_s') && ~isempty(s.eta_s) && s.eta_s > 0
            text = text + "  ·  at most " + humanTime(s.eta_s) + " left in this fold (stops earlier when it stops improving)";
        end
    case "evaluating", text = "Evaluating on the test trials: " + string(s.message);
    case "finished", text = "Finished";
    case "cancelled", text = "Cancelled";
    otherwise, text = "Failed";
end
if isfield(s, 'elapsed_s') && ~isempty(s.elapsed_s)
    if ismember(state, ["finished", "failed", "cancelled"]), w = "took "; else, w = "running for "; end
    text = text + "  ·  " + w + humanTime(s.elapsed_s);
end
end

function d = humanTime(sec)
sec = double(sec);
if sec < 90, d = sprintf("%.0f s", sec); elseif sec < 5400, d = sprintf("%.0f min", sec / 60); else, d = sprintf("%.1f h", sec / 3600); end
end

function lines = tailLog(folder, n)
lines = "";
f = fullfile(folder, "log.txt");
if ~isfile(f), f = fullfile(folder, "stdout.txt"); end
if ~isfile(f), return; end
try
    txt = splitlines(string(fileread(f)));
    txt = txt(strlength(strtrim(txt)) > 0);
    lines = txt(max(1, end - n + 1):end);
    if isempty(lines), lines = ""; end
catch
end
end

function ax = clearAxes(ax)
cla(ax, 'reset');
end

function s = onoff(tf)
if tf, s = 'on'; else, s = 'off'; end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
