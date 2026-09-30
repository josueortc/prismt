function f = projectFolder(newFolder)
%PROJECTFOLDER The folder holding your PRISMT datasets, runs and cluster jobs.
%   Default ~/Documents/PRISMT (with datasets/, runs/, cluster/, scripts/ inside).
%   prismt.internal.projectFolder(folder) changes and remembers it.
if nargin == 1
    prismt.internal.settings("project_folder", char(newFolder));
end
f = prismt.internal.settings("project_folder");
if isempty(f)
    home = getenv('HOME');
    if ispc, home = getenv('USERPROFILE'); end
    f = fullfile(home, 'Documents', 'PRISMT');
end
f = string(f);
for sub = ["datasets", "runs", "cluster", "scripts"]
    if ~isfolder(fullfile(f, sub)), mkdir(fullfile(f, sub)); end
end
end
