function value = settings(name, value)
%SETTINGS Read or write one PRISMT user setting (kept across MATLAB versions).
%
%   v = prismt.internal.settings("python")          % read ([] if unset)
%   prismt.internal.settings("python", "/path")     % write
%   s = prismt.internal.settings()                  % all settings (struct)
%
%   Settings live in a JSON file in the user's configuration folder (macOS:
%   ~/Library/Application Support/PRISMT, Windows: %APPDATA%\PRISMT, Linux:
%   ~/.config/prismt), or in the folder named by the PRISMT_SETTINGS_DIR environment
%   variable (used by the tests). Unlike setpref, they survive MATLAB upgrades.
file = fullfile(folder(), 'settings.json');
s = struct();
if isfile(file)
    try
        s = jsondecode(fileread(file));
    catch
        s = struct();
    end
end
if nargin == 0
    value = s;
    return
end
key = matlab.lang.makeValidName(char(name));
if nargin == 1
    if isfield(s, key), value = s.(key); else, value = []; end
    return
end
s.(key) = value;
if ~isfolder(fileparts(file)), mkdir(fileparts(file)); end
prismt.internal.atomicWrite(file, jsonencode(s, 'PrettyPrint', true));
end

function f = folder()
f = getenv('PRISMT_SETTINGS_DIR');
if ~isempty(f), return; end
if ispc
    f = fullfile(getenv('APPDATA'), 'PRISMT');
elseif ismac
    f = fullfile(getenv('HOME'), 'Library', 'Application Support', 'PRISMT');
else
    base = getenv('XDG_CONFIG_HOME');
    if isempty(base), base = fullfile(getenv('HOME'), '.config'); end
    f = fullfile(base, 'prismt');
end
end
