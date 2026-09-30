classdef LocalRun < handle
    %LOCALRUN A PRISMT run (training or tuning) going on in the background on this computer.
    %
    %   run = prismt.train(cfg)            % starts one
    %   s = run.status()                   % state, epoch, latest scores, ETA, error
    %   h = run.history()                  % table: one row per epoch (and fold)
    %   run.stop()                         % stop gracefully (keeps the best model so far)
    %   run.wait()                         % block until it ends
    %   run = prismt.run.LocalRun.attach(folder)   % reconnect after restarting MATLAB
    %
    %   Closing MATLAB does not stop a run; attach to it again later.

    properties (SetAccess = private)
        RunDir string
        Kind string = "train"      % train | hpo
        Process = []               % prismt.run.Process, empty after attach
    end
    properties (Access = private)
        LastStatus = []
    end

    methods (Static)
        function run = start(cfg, runDir, opts)
            arguments
                cfg struct
                runDir (1, 1) string
                opts.Kind (1, 1) string {mustBeMember(opts.Kind, ["train", "hpo"])} = "train"
                opts.Python (1, 1) string = ""
            end
            py = opts.Python;
            if strlength(py) == 0, py = prismt.env.python(); end
            if ~isfolder(runDir), mkdir(runDir); end
            runJson = fullfile(runDir, "run.json");
            prismt.internal.atomicWrite(runJson, jsonencode(cfg, 'PrettyPrint', true));
            args = ["-u", "-m", "prismt", opts.Kind, "--config", runJson, "--run-dir", runDir];
            writeLaunchScript(runDir, py, args);
            run = prismt.run.LocalRun();
            run.RunDir = runDir;
            run.Kind = opts.Kind;
            run.Process = prismt.run.Process.start(py, args, Folder=runDir, Log=fullfile(runDir, "stdout.txt"));
        end

        function run = attach(runDir)
            run = prismt.run.LocalRun();
            run.RunDir = string(runDir);
            if isfile(fullfile(runDir, "config.json"))
                c = jsondecode(fileread(fullfile(runDir, "config.json")));
                if isfield(c, 'hpo') && isfolder(fullfile(runDir, "hpo")), run.Kind = "hpo"; end
            end
        end
    end

    methods
        function s = status(run)
            %STATUS Latest status.json (keeps the last good copy if a write is caught half-done).
            f = fullfile(run.RunDir, "status.json");
            if ~isfile(f)
                s = struct('state', "queued", 'message', "Starting Python...", 'error', []);
            else
                % Always re-read: the file is small, and its timestamp has 1 s resolution.
                try
                    s = jsondecode(fileread(f));
                    run.LastStatus = s;
                catch
                    s = run.LastStatus;   % caught mid-write; use the last good copy
                    if isempty(s), s = struct('state', "queued", 'message', "", 'error', []); end
                end
            end
            if ~isempty(run.Process) && ~run.Process.isRunning() && ~ismember(string(s.state), ["finished", "failed", "cancelled"])
                pause(0.5);            % the final status may be written just after the process exits
                try, s = jsondecode(fileread(f)); catch, end
            end
            if ~isempty(run.Process) && ~run.Process.isRunning() && ~ismember(string(s.state), ["finished", "failed", "cancelled"])
                s.state = "failed";
                s.error = prismt.run.explainFailure(run.RunDir, run.Process.exitCode());
                s.message = s.error.title;
            end
        end

        function tf = isActive(run)
            s = run.status();
            tf = ~ismember(string(s.state), ["finished", "failed", "cancelled"]);
        end

        function h = history(run)
            %HISTORY One row per epoch (all folds; for a cluster fold job, that fold).
            f = fullfile(run.RunDir, "history.csv");
            h = table();
            if ~isfile(f)
                folds = dir(fullfile(run.RunDir, "fold_*", "history.csv"));
                if isempty(folds), return; end
                f = fullfile(folds(end).folder, folds(end).name);
            end
            try
                h = readtable(f, 'TextType', 'string');
            catch
            end
        end

        function stop(run, opts)
            %STOP Ask the run to stop after the current step; results are written from the best
            %model so far. Stop(Force=true) ends the process at once (no results).
            arguments
                run
                opts.Force (1, 1) logical = false
                opts.Timeout (1, 1) double = 30
            end
            prismt.internal.atomicWrite(fullfile(run.RunDir, "STOP"), "stop");
            if isempty(run.Process), return; end
            if opts.Force
                run.Process.terminate(true);
                return
            end
            t0 = tic;
            while run.Process.isRunning() && toc(t0) < opts.Timeout
                pause(0.5);
            end
            if run.Process.isRunning(), run.Process.terminate(false); end
            pause(2);
            if run.Process.isRunning(), run.Process.terminate(true); end
        end

        function s = wait(run, timeout)
            %WAIT Block until the run ends; returns the final status.
            if nargin < 2, timeout = inf; end
            t0 = tic;
            while run.isActive() && toc(t0) < timeout
                pause(1);
            end
            s = run.status();
        end

        function R = results(run)
            R = prismt.loadResults(run.RunDir);
        end
    end
end

function writeLaunchScript(runDir, py, args)
% A copy-paste way to rerun this run from a terminal (never used by the app itself).
q = @prismt.internal.shellQuote;
if ispc
    text = "@echo off" + newline + "set PYTHONPATH=" + fullfile(prismt.internal.repoRoot(), "src") + newline + ...
        strjoin(arrayfun(@(a) q(a, "cmd"), [py, args]), " ") + newline;
    prismt.internal.atomicWrite(fullfile(runDir, "launch.cmd"), text);
else
    text = "#!/bin/sh" + newline + "# Rerun this PRISMT run from a terminal." + newline + ...
        "export PYTHONPATH=" + q(fullfile(prismt.internal.repoRoot(), "src"), "posix") + newline + ...
        strjoin(arrayfun(@(a) q(a, "posix"), [py, args]), " ") + newline;
    prismt.internal.atomicWrite(fullfile(runDir, "launch.sh"), text);
end
end
