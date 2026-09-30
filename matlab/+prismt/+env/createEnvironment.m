function job = createEnvironment(opts)
%CREATEENVIRONMENT Build the PRISMT Python environment (no admin rights, no terminal).
%
%   job = prismt.env.createEnvironment()          % starts in the background
%   job.Process.wait(); type(job.Log)             % or follow job.Log in the app
%
%   Uses conda if it is installed. Otherwise it downloads micromamba (a single, official
%   program from github.com/mamba-org) into PRISMT's user folder, after asking, and uses it
%   to create the environment from environment.yml there. The environment's Python is
%   job.Python; check it with prismt.env.doctor(job.Python) when the process has finished.
arguments
    opts.Download (1, 1) logical = false   % allow downloading micromamba without asking again
    opts.Conda (1, 1) string = ""          % use this conda executable
end
root = prismt.internal.repoRoot();
yml = fullfile(root, 'environment.yml');
user = prismt.env.userFolder();
if ~isfolder(user), mkdir(user); end
log = fullfile(user, "create_environment.log");
if isfile(log), delete(log); end
conda = opts.Conda;
if strlength(conda) == 0, conda = findConda(); end
if strlength(conda)
    envDir = fullfile(fileparts(fileparts(conda)), "envs", "prismt");
    if endsWith(fileparts(conda), "condabin"), envDir = fullfile(fileparts(fileparts(conda)), "envs", "prismt"); end
    action = "create";
    if isfolder(envDir), action = "update"; end
    args = ["env", action, "--file", yml, "--name", "prismt"];
    if action == "update", args(end + 1) = "--prune"; end
    exe = conda;
else
    if ~opts.Download
        error('prismt:E_ENV_NO_CONDA', ['Neither conda nor micromamba was found. Call ' ...
            'prismt.env.createEnvironment(Download=true) to download micromamba (about 15 MB) into %s, or ' ...
            'install Miniforge from https://conda-forge.org/download/.'], user);
    end
    exe = downloadMicromamba(user);
    envDir = fullfile(user, "env");
    args = ["create", "--yes", "--file", yml, "--prefix", envDir, "--root-prefix", fullfile(user, "mamba")];
end
if ispc, py = fullfile(envDir, "python.exe"); else, py = fullfile(envDir, "bin", "python"); end
proc = prismt.run.Process.start(exe, args, Folder=root, Log=log, Env=struct('CONDA_PKGS_DIRS', fullfile(user, 'pkgs')));
job = struct('Process', proc, 'Log', log, 'Python', py, 'Tool', exe);
end

function c = findConda()
c = "";
list = [string(getenv('CONDA_EXE')); fullfile(string(getenv('HOME')), ["miniforge3", "mambaforge", "miniconda3", "anaconda3"]', "bin", "conda"); ...
        "/opt/anaconda3/bin/conda"; "/opt/miniconda3/bin/conda"; "/opt/miniforge3/bin/conda"; ...
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda"; ...
        fullfile(string(getenv('USERPROFILE')), ["miniforge3", "miniconda3", "anaconda3"]', "Scripts", "conda.exe")];
list = list(strlength(list) > 0);
hit = list(isfile(list));
if ~isempty(hit), c = hit(1); end
end

function exe = downloadMicromamba(user)
if ispc
    plat = "win-64"; exe = fullfile(user, "micromamba.exe");
elseif ismac
    [~, arch] = system('uname -m');
    plat = "osx-" + ternary(contains(arch, "arm64"), "arm64", "64"); exe = fullfile(user, "micromamba");
else
    [~, arch] = system('uname -m');
    plat = "linux-" + ternary(contains(arch, "aarch64"), "aarch64", "64"); exe = fullfile(user, "micromamba");
end
if ~isfile(exe)
    url = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-" + plat;
    if ispc, url = url + ".exe"; end
    websave(exe, url, weboptions('Timeout', 120));
    if ~ispc, system(sprintf('chmod +x %s', prismt.internal.shellQuote(exe, "posix"))); end
end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
