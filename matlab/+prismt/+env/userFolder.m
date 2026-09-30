function f = userFolder()
%USERFOLDER Where PRISMT keeps its own tools and environment for this user (no admin rights).
%   macOS: ~/Library/Application Support/PRISMT, Windows: %LOCALAPPDATA%\PRISMT,
%   Linux: ~/.local/share/prismt. Overridden by PRISMT_USER_DIR.
f = string(getenv('PRISMT_USER_DIR'));
if strlength(f), return; end
if ispc
    f = fullfile(string(getenv('LOCALAPPDATA')), "PRISMT");
elseif ismac
    f = fullfile(string(getenv('HOME')), "Library", "Application Support", "PRISMT");
else
    f = fullfile(string(getenv('HOME')), ".local", "share", "prismt");
end
end
