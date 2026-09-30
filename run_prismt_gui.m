%% PRISMT - open the app
% Double-click this file in MATLAB's Current Folder, or type run_prismt_gui.
%
% The app walks through setup (Python), data, the task, training and results.
% Everything it does can also be scripted: see matlab/examples/prismt_tutorial.m.

prismt_release = version('-release');
if str2double(prismt_release(1:4)) < 2021
    error('PRISMT needs MATLAB R2021a or newer (this is R%s).', prismt_release);
end
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab'));
clear prismt_release
prismt.gui();
