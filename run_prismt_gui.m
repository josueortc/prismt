%% PRISMT Training Setup - Quick Launch
% Double-click this file or run: run_prismt_gui
%
% Launches the PRISMT GUI for configuring transformer training on
% widefield calcium imaging data. No need to change directory.
%
% Usage:
%   run_prismt_gui          % from Command Window
%   Double-click run_prismt_gui.m in Current Folder

% Add gui folder to path (allows running from project root)
prismt_root = fileparts(mfilename('fullpath'));
gui_dir = fullfile(prismt_root, 'gui');
if exist(gui_dir, 'dir')
    addpath(gui_dir);
end

% Launch GUI
prismt_training_setup();
