function root = repoRoot()
%REPOROOT Folder that holds run_prismt_gui.m, matlab/ and src/ (the PRISMT checkout).
root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
end
