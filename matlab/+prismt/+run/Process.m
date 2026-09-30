classdef Process < handle
    %PROCESS A background process started without a shell, with a clean environment.
    %
    %   p = prismt.run.Process.start(exe, args, Folder=..., Log=..., Env=struct(...))
    %
    %   Uses java.lang.ProcessBuilder when the JVM is available: arguments are passed as a
    %   list, so paths with spaces or quotes need no escaping. The child does not inherit
    %   the variables MATLAB adds to its own environment (KMP_*, GFORTRAN_*, MATLAB*, ...)
    %   or MATLAB's library folders, which otherwise break conda's PyTorch. Without a JVM a
    %   launch script is written and started with system(). Output goes to the Log file.

    properties (SetAccess = private)
        Log string = ""
        Command string = ""
    end
    properties (Access = private)
        Java = []
        PidFile string = ""
        StartedShell logical = false
    end

    methods (Static)
        function p = start(exe, args, opts)
            arguments
                exe (1, 1) string
                args (1, :) string = strings(1, 0)
                opts.Folder (1, 1) string = string(pwd)
                opts.Log (1, 1) string = string(tempname) + ".log"
                opts.Env struct = struct()
                opts.ForceShell (1, 1) logical = false
            end
            p = prismt.run.Process();
            p.Log = opts.Log;
            p.Command = strjoin([exe, args], " ");
            env = prismt.run.Process.cleanEnvironment(exe, opts.Env);
            if usejava('jvm') && ~opts.ForceShell
                cmd = cellstr([exe, args]);
                pb = java.lang.ProcessBuilder(cmd);
                pb.directory(java.io.File(char(opts.Folder)));
                jenv = pb.environment();
                keys = cell(jenv.keySet().toArray());
                for k = 1:numel(keys)
                    if ~isKey(env, keys{k}), jenv.remove(keys{k}); end
                end
                names = env.keys();
                for k = 1:numel(names)
                    jenv.put(names{k}, env(names{k}));
                end
                pb.redirectErrorStream(true);
                pb.redirectOutput(javaMethod('appendTo', 'java.lang.ProcessBuilder$Redirect', java.io.File(char(opts.Log))));
                p.Java = pb.start();
                p.Java.getOutputStream().close();
            else
                p.startShell(exe, args, opts.Folder, env);
            end
        end

        function env = cleanEnvironment(exe, extra)
            %CLEANENVIRONMENT The environment a PRISMT Python child should get.
            env = containers.Map();
            if usejava('jvm')
                m = java.lang.System.getenv();
                keys = cell(m.keySet().toArray());
                for k = 1:numel(keys)
                    env(keys{k}) = char(m.get(keys{k}));
                end
            else
                for name = ["PATH", "HOME", "USER", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "LANG", "USERPROFILE", "APPDATA", "LOCALAPPDATA"]
                    v = getenv(name);
                    if ~isempty(v), env(char(name)) = v; end
                end
            end
            drop = ["MATLAB", "MATLABPATH", "MATLAB_MEM_MGR_VALUE", "TOOLBOX", "ARCH", "OSG_LD_LIBRARY_PATH", ...
                    "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__", "AWT_TOOLKIT"];
            keys = env.keys();
            for k = 1:numel(keys)
                name = string(keys{k});
                if any(name == drop) || startsWith(name, ["KMP_", "GFORTRAN_"])
                    env.remove(keys{k});
                end
            end
            root = string(matlabroot);
            for name = ["PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"]
                if isKey(env, char(name))
                    parts = split(string(env(char(name))), pathsep);
                    parts = parts(~startsWith(parts, root) & strlength(parts) > 0);
                    if isempty(parts)
                        env.remove(char(name));
                    else
                        env(char(name)) = char(strjoin(parts, pathsep));
                    end
                end
            end
            % Put the Python environment's own folders first (needed by conda on Windows).
            bin = fileparts(char(exe));
            front = string(bin);
            if ispc
                front = [front, fullfile(bin, "Library", "bin"), fullfile(bin, "Library", "usr", "bin"), ...
                         fullfile(bin, "Library", "mingw-w64", "bin"), fullfile(bin, "Scripts")];
            end
            current = "";
            if isKey(env, 'PATH'), current = string(env('PATH')); end
            env('PATH') = char(strjoin([front, current], pathsep));
            env('PYTHONPATH') = fullfile(prismt.internal.repoRoot(), 'src');
            env('PYTHONUNBUFFERED') = '1';
            env('PYTHONIOENCODING') = 'utf-8';
            env('PYTHONNOUSERSITE') = '1';
            env('MPLBACKEND') = 'Agg';
            env('PRISMT_LAUNCHED_BY') = 'matlab';
            for f = string(fieldnames(extra))'
                env(char(f)) = char(string(extra.(f)));
            end
        end
    end

    methods
        function tf = isRunning(p)
            if ~isempty(p.Java)
                tf = p.Java.isAlive();
            elseif p.StartedShell
                pid = p.pid();
                tf = ~isempty(pid) && pidAlive(pid);
            else
                tf = false;
            end
        end

        function code = exitCode(p)
            %EXITCODE Exit code once finished ([] while running or unknown).
            code = [];
            if ~isempty(p.Java) && ~p.Java.isAlive()
                code = double(p.Java.exitValue());
            end
        end

        function code = wait(p, timeout)
            %WAIT Block until the process ends or timeout (seconds) passes.
            if nargin < 2, timeout = inf; end
            t0 = tic;
            while p.isRunning() && toc(t0) < timeout
                pause(0.2);
            end
            code = p.exitCode();
        end

        function terminate(p, force)
            %TERMINATE Ask the process to stop (SIGTERM); force = true kills it.
            if nargin < 2, force = false; end
            if ~isempty(p.Java)
                if force, p.Java.destroyForcibly(); else, p.Java.destroy(); end
            else
                pid = p.pid();
                if isempty(pid), return; end
                if ispc
                    flag = '';
                    if force, flag = '/F'; end
                    system(sprintf('taskkill /PID %d /T %s', pid, flag));
                elseif force
                    system(sprintf('kill -9 %d', pid));
                else
                    system(sprintf('kill %d', pid));
                end
            end
        end
    end

    methods (Access = private)
        function startShell(p, exe, args, folder, env)
            q = @prismt.internal.shellQuote;
            p.PidFile = p.Log + ".pid";
            keys = env.keys();
            if ispc
                script = p.Log + ".cmd";
                lines = "@echo off";
                for k = 1:numel(keys)
                    lines(end + 1) = "set """ + keys{k} + "=" + env(keys{k}) + """"; %#ok<AGROW>
                end
                lines(end + 1) = "cd /d " + q(folder, "cmd");
                lines(end + 1) = strjoin(arrayfun(@(a) q(a, "cmd"), [exe, args]), " ") + " >> " + q(p.Log, "cmd") + " 2>&1";
                prismt.internal.atomicWrite(script, strjoin(lines, newline));
                system(sprintf('start "" /B cmd /c %s', q(script, "cmd")));
            else
                script = p.Log + ".sh";
                lines = "#!/bin/sh";
                for k = 1:numel(keys)
                    if ~isempty(regexp(keys{k}, '^[A-Za-z_][A-Za-z0-9_]*$', 'once'))
                        lines(end + 1) = "export " + keys{k} + "=" + q(env(keys{k}), "posix"); %#ok<AGROW>
                    end
                end
                lines(end + 1) = "cd " + q(folder, "posix");
                lines(end + 1) = "echo $$ > " + q(p.PidFile, "posix");
                lines(end + 1) = "exec " + strjoin(arrayfun(@(a) q(a, "posix"), [exe, args]), " ") + ...
                    " >> " + q(p.Log, "posix") + " 2>&1";
                prismt.internal.atomicWrite(script, strjoin(lines, newline) + newline);
                system(sprintf('/bin/sh %s </dev/null >/dev/null 2>&1 &', q(script, "posix")));
            end
            p.StartedShell = true;
        end

        function pid = pid(p)
            pid = [];
            if strlength(p.PidFile) && isfile(p.PidFile)
                pid = str2double(strtrim(fileread(p.PidFile)));
                if isnan(pid), pid = []; end
            end
        end
    end
end

function tf = pidAlive(pid)
if ispc
    [~, out] = system(sprintf('tasklist /FI "PID eq %d" /NH', pid));
    tf = contains(out, string(pid));
else
    tf = system(sprintf('kill -0 %d 2>/dev/null', pid)) == 0;
end
end
