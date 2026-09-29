function v = version()
%VERSION PRISMT package version and the versions of the files it shares with Python.
%   v = prismt.version() returns a struct with fields Package (e.g. "0.1.0.dev0") and
%   Formats (dataset, config, run, status, results). MATLAB and Python read the same
%   files, so both halves of one PRISMT folder always agree.
res = fullfile(prismt.internal.repoRoot(), 'src', 'prismt', 'resources');
v.Package = strtrim(string(fileread(fullfile(res, 'VERSION'))));
v.Formats = jsondecode(fileread(fullfile(res, 'formats.json')));
end
