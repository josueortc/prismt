function c = candidates()
%CANDIDATES Python executables that might hold a PRISMT environment, most likely first.
%   Order: the saved setting, PRISMT_PYTHON, conda's list of environments, the usual
%   install locations of miniforge / miniconda / anaconda / micromamba (including the
%   one PRISMT itself creates), and a .venv in the PRISMT folder.
c = strings(0, 1);
saved = prismt.internal.settings("python");
if ~isempty(saved), c(end + 1) = string(saved); end
if ~isempty(getenv('PRISMT_PYTHON')), c(end + 1) = string(getenv('PRISMT_PYTHON')); end
home = string(getenv('HOME'));
if ispc, home = string(getenv('USERPROFILE')); end
exe = fullfile("bin", "python");
if ispc, exe = "python.exe"; end
envs = strings(0, 1);
listFile = fullfile(home, ".conda", "environments.txt");
if isfile(listFile)
    lines = strtrim(splitlines(string(fileread(listFile))));
    envs = [envs; lines(endsWith(lines, filesep + "prismt"))];
end
bases = [fullfile(home, ["miniforge3", "mambaforge", "miniconda3", "anaconda3", "micromamba", ...
                        fullfile("opt", "anaconda3"), fullfile("opt", "miniconda3")]), ...
         "/opt/anaconda3", "/opt/miniconda3", "/opt/miniforge3", "/opt/conda", ...
         "/opt/homebrew/Caskroom/miniforge/base", "/usr/local/Caskroom/miniforge/base", ...
         fullfile(string(getenv('LOCALAPPDATA')), ["miniforge3", "miniconda3", "anaconda3"]), ...
         fullfile("C:", "ProgramData", ["miniforge3", "Miniconda3", "Anaconda3"])];
envs = [envs; fullfile(bases(:), "envs", "prismt"); fullfile(prismt.env.userFolder(), "env")];
for e = envs'
    c(end + 1) = fullfile(e, exe); %#ok<AGROW>
end
c(end + 1) = fullfile(prismt.internal.repoRoot(), ".venv", exe);
c = unique(c(isfile(c)), 'stable');
end
