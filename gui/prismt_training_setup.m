function prismt_training_setup()
%PRISMT_TRAINING_SETUP MATLAB GUI for configuring PRISMT training pipeline
%
%   Suite2P-style interface for experimental neuroscientists.
%   Load data → Configure → Run or Generate script
%
%   Launched via: run_prismt_gui  (from project root)
%   Or: prismt_training_setup()   (when gui/ is on path)

    if exist('uifigure', 'file')
        createUIFigure();
    else
        createClassicFigure();
    end
end

%% Main UI (Suite2P-style panel layout)
function createUIFigure()
    fig = uifigure('Name', 'PRISMT - Training Setup', ...
        'Position', [80 50 720 720], ...
        'Color', [0.94 0.94 0.96]);
    
    % === HEADER ===
    uilabel(fig, 'Text', 'PRISMT', 'Position', [20 680 120 28], ...
        'FontSize', 18, 'FontWeight', 'bold', 'FontColor', [0.15 0.35 0.6]);
    uilabel(fig, 'Text', 'Transformer training for trial-based data', 'Position', [145 684 280 22], ...
        'FontSize', 11, 'FontColor', [0.5 0.5 0.5]);
    
    helpBtn = uibutton(fig, 'Text', 'Help', 'Position', [620 672 80 28], ...
        'ButtonPushedFcn', @(src,~) openHelp());
    
    % === PANEL 1: Data ===
    p1 = uipanel(fig, 'Title', '1. Load Dataset', 'Position', [20 500 680 160], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p1, 'Text', 'Dataset (.mat):', 'Position', [15 95 100 22]);
    pathField = uieditfield(p1, 'text', 'Position', [120 92 420 28], ...
        'Value', '');
    uibutton(p1, 'Text', 'Browse', 'Position', [550 90 55 32], ...
        'ButtonPushedFcn', @(src,~) browseForFile(pathField));
    uibutton(p1, 'Text', 'Load', 'Position', [615 90 55 32], ...
        'ButtonPushedFcn', @(src,~) validateDataset(pathField, summaryLabel));
    
    summaryLabel = uilabel(p1, 'Text', 'No data loaded. Click Browse and Load.', ...
        'Position', [15 45 655 35], 'WordWrap', 'on', 'FontColor', [0.4 0.4 0.4]);
    
    % === PANEL 2: Input & Tokenization ===
    p2 = uipanel(fig, 'Title', '2. Input & Tokenization', 'Position', [20 340 680 150], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p2, 'Text', 'Data type:', 'Position', [15 95 80 22]);
    dataTypeDD = uidropdown(p2, 'Position', [100 92 100 28], ...
        'Items', {'dff (ΔF/F)', 'zscore'}, 'Value', 'dff (ΔF/F)');
    
    uilabel(p2, 'Text', 'Normalization:', 'Position', [220 95 90 22]);
    normDD = uidropdown(p2, 'Position', [315 92 200 28], ...
        'Items', {'Scale ×20 (recommended)', 'Robust (median/IQR)', 'Percentile clip', 'None'}, ...
        'Value', 'Scale ×20 (recommended)');
    
    uilabel(p2, 'Text', 'Tokenization:', 'Position', [15 50 90 22]);
    tokenInfoLabel = uilabel(p2, 'Text', 'Spatial: 1 token per region (computed after load)', ...
        'Position', [110 45 565 30], 'WordWrap', 'on', 'FontColor', [0.4 0.4 0.4]);
    
    % === PANEL 3: Conditions ===
    p3 = uipanel(fig, 'Title', '3. Comparison Conditions', 'Position', [20 160 680 170], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p3, 'Text', 'Task:', 'Position', [15 115 50 22]);
    taskTypeDD = uidropdown(p3, 'Position', [70 112 180 28], ...
        'Items', {'Phase (early vs late)', 'Genotype (WT vs mutant)'}, ...
        'Value', 'Phase (early vs late)');
    
    uilabel(p3, 'Text', 'Phases:', 'Position', [270 115 60 22]);
    phase1DD = uidropdown(p3, 'Position', [335 112 80 28], ...
        'Items', {'early', 'mid', 'late'}, 'Value', 'early');
    uilabel(p3, 'Text', 'vs', 'Position', [420 115 20 22]);
    phase2DD = uidropdown(p3, 'Position', [445 112 80 28], ...
        'Items', {'early', 'mid', 'late'}, 'Value', 'late');
    
    uilabel(p3, 'Text', 'Stim:', 'Position', [15 70 50 22]);
    stimEdit = uieditfield(p3, 'text', 'Position', [70 67 60 28], 'Value', '1');
    uilabel(p3, 'Text', 'Response:', 'Position', [150 70 70 22]);
    responseEdit = uieditfield(p3, 'text', 'Position', [225 67 80 28], 'Value', '0, 1');
    
    uilabel(p3, 'Text', 'Seed:', 'Position', [330 70 50 22]);
    seedEdit = uieditfield(p3, 'numeric', 'Position', [385 67 60 28], 'Value', 42);
    
    % === PANEL 4: Training & Output ===
    p4 = uipanel(fig, 'Title', '4. Training & Cluster', 'Position', [20 10 420 140], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p4, 'Text', 'Batch:', 'Position', [15 95 50 22]);
    batchEdit = uieditfield(p4, 'numeric', 'Position', [70 92 50 28], 'Value', 16);
    uilabel(p4, 'Text', 'Epochs:', 'Position', [135 95 50 22]);
    epochsEdit = uieditfield(p4, 'numeric', 'Position', [190 92 50 28], 'Value', 100);
    uilabel(p4, 'Text', 'LR:', 'Position', [255 95 30 22]);
    lrEdit = uieditfield(p4, 'text', 'Position', [290 92 60 28], 'Value', '5e-5');
    
    uilabel(p4, 'Text', 'Output dir:', 'Position', [15 50 75 22]);
    outputDirEdit = uieditfield(p4, 'text', 'Position', [95 47 310 28], ...
        'Value', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'generated_scripts'));
    
    uilabel(p4, 'Text', 'SLURM partition:', 'Position', [15 15 100 22]);
    partitionEdit = uieditfield(p4, 'text', 'Position', [120 12 80 28], 'Value', 'gpu');
    uilabel(p4, 'Text', 'Mem (GB):', 'Position', [210 15 70 22]);
    memEdit = uieditfield(p4, 'numeric', 'Position', [285 12 50 28], 'Value', 32);
    
    % === ACTION PANEL (right side, Suite2P-style) ===
    actionPanel = uipanel(fig, 'Title', 'Run', 'Position', [455 10 245 140], ...
        'BackgroundColor', [0.95 0.97 1], 'FontWeight', 'bold');
    
    runBtn = uibutton(actionPanel, 'Text', 'Run Training Now', ...
        'Position', [15 70 215 40], 'BackgroundColor', [0.2 0.55 0.35], ...
        'FontColor', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(src,~) runTraining(src.Parent.Parent));
    
    genBtn = uibutton(actionPanel, 'Text', 'Generate Cluster Script', ...
        'Position', [15 25 215 38], 'BackgroundColor', [0.25 0.5 0.8], ...
        'FontColor', [1 1 1], 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(src,~) generateScript(src.Parent.Parent));
    
    % === STATUS BAR (bottom, Suite2P-style) ===
    statusLabel = uilabel(fig, 'Text', 'Ready. Load a dataset to begin.', ...
        'Position', [20 5 680 22], 'FontColor', [0.4 0.4 0.4], 'FontSize', 10);
    
    % Store handles
    fig.UserData = struct(...
        'pathField', pathField, 'summaryLabel', summaryLabel, ...
        'dataTypeDD', dataTypeDD, 'normDD', normDD, 'tokenInfoLabel', tokenInfoLabel, ...
        'taskTypeDD', taskTypeDD, 'phase1DD', phase1DD, 'phase2DD', phase2DD, ...
        'stimEdit', stimEdit, 'responseEdit', responseEdit, 'seedEdit', seedEdit, ...
        'batchEdit', batchEdit, 'epochsEdit', epochsEdit, 'lrEdit', lrEdit, ...
        'valSplit', 0.2, ...
        'outputDirEdit', outputDirEdit, 'partitionEdit', partitionEdit, 'memEdit', memEdit, ...
        'statusLabel', statusLabel, 'runBtn', runBtn, 'genBtn', genBtn, ...
        'dataInfo', struct());
    
    pathField.ValueChangedFcn = @(src,~) setStatus(fig.UserData.statusLabel, ...
        iif(exist(src.Value, 'file')==2, 'Path OK. Click Load to validate.', 'Enter dataset path.'));
end

function out = iif(cond, a, b)
    if cond, out = a; else, out = b; end
end

function setStatus(lbl, txt)
    if isvalid(lbl), lbl.Text = txt; end
end

function openHelp()
    docPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'docs', 'PRISMT_GUI_DESIGN.md');
    if exist(docPath, 'file')
        edit(docPath);
    end
    web('https://github.com/josueortc/prismt/wiki', '-browser');
end

%% Callbacks
function browseForFile(pathField)
    [f, p] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, 'Select standardized .mat file');
    if f
        pathField.Value = fullfile(p, f);
    end
end

function validateDataset(pathField, summaryLabel)
    pathStr = strtrim(pathField.Value);
    fig = pathField.Parent.Parent;

    try
        [ok, info, errMsg] = validate_prismt_mat(pathStr);
    catch ME
        summaryLabel.Text = 'Validation error.';
        summaryLabel.FontColor = [0.6 0.2 0.2];
        showAlert(fig, ['Unexpected error: ' ME.message], 'Load failed');
        return;
    end

    if ok
        msg = sprintf('OK: %d datasets | ~%d trials x %d timepoints x %d regions', ...
            info.n_datasets, info.n_trials, info.n_timepoints, info.n_regions);
        summaryLabel.Text = msg;
        summaryLabel.FontColor = [0.1 0.5 0.2];

        fig.UserData.dataInfo = info;
        if isfield(fig.UserData, 'tokenInfoLabel')
            fig.UserData.tokenInfoLabel.Text = sprintf('Spatial: %d regions -> %d tokens (+ CLS)', ...
                info.n_regions, info.n_regions + 1);
        end

        fig.UserData.statusLabel.Text = 'Data loaded. Configure and run.';
        showAlert(fig, msg, 'Data loaded');
    else
        summaryLabel.Text = 'Invalid format. See error for details.';
        summaryLabel.FontColor = [0.6 0.2 0.2];
        showAlert(fig, errMsg, 'Invalid dataset format');
    end
end

function showAlert(parent, msg, title)
    if exist('uialert', 'file')
        uialert(parent, msg, title);
    else
        errordlg(msg, title);
    end
end

function runTraining(fig)
    if ~isvalid(fig), return; end
    ud = fig.UserData;
    
    dataPath = ud.pathField.Value;
    if isempty(dataPath) || exist(dataPath, 'file') ~= 2
        showAlert(fig, 'Load and validate a dataset first.', 'Run Training');
        return;
    end
    
    taskType = strcmp(ud.taskTypeDD.Value, 'Phase (early vs late)');
    phase1 = ud.phase1DD.Value;
    phase2 = ud.phase2DD.Value;
    dataType = strsplit(ud.dataTypeDD.Value, ' ');
    dataType = dataType{1};  % 'dff' or 'zscore'
    
    taskArg = 'phase';
    phaseArgs = sprintf('--phase1 %s --phase2 %s', phase1, phase2);
    if ~taskType
        taskArg = 'genotype';
        phaseArgs = '';
    end
    
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    trainPath = fullfile(projectRoot, 'train.py');
    
    cmd = sprintf('python "%s" --data_path "%s" --data_type %s --task_type %s %s --batch_size %d --epochs %d --learning_rate %s --val_split %.2f --seed %d --save_dir results', ...
        trainPath, dataPath, dataType, taskArg, phaseArgs, ...
        ud.batchEdit.Value, ud.epochsEdit.Value, ud.lrEdit.Value, ...
        ud.valSplit, ud.seedEdit.Value);
    
    ud.statusLabel.Text = 'Starting training...';
    drawnow;
    
    [status, result] = system(sprintf('cd "%s" && %s', projectRoot, cmd));
    
    if status == 0
        ud.statusLabel.Text = 'Training finished.';
        msgbox('Training completed successfully.', 'PRISMT', 'help');
    else
        ud.statusLabel.Text = 'Training failed. See command window.';
        showAlert(fig, sprintf('Training failed:\n%s', result), 'Error');
    end
end

function generateScript(fig)
    if ~isvalid(fig), return; end
    ud = fig.UserData;
    
    dataPath = ud.pathField.Value;
    if isempty(dataPath) || exist(dataPath, 'file') ~= 2
        showAlert(fig, 'Load and validate a dataset first.', 'Generate Script');
        return;
    end
    
    taskType = strcmp(ud.taskTypeDD.Value, 'Phase (early vs late)');
    phase1 = ud.phase1DD.Value;
    phase2 = ud.phase2DD.Value;
    dataType = strsplit(ud.dataTypeDD.Value, ' ');
    dataType = dataType{1};
    
    taskArg = 'phase';
    phaseArgs = sprintf('--phase1 %s --phase2 %s', phase1, phase2);
    if ~taskType
        taskArg = 'genotype';
        phaseArgs = '';
    end
    
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    trainPath = fullfile(projectRoot, 'train.py');
    outDir = ud.outputDirEdit.Value;
    batchSize = ud.batchEdit.Value;
    epochs = ud.epochsEdit.Value;
    lr = ud.lrEdit.Value;
    valSplit = ud.valSplit;
    seed = ud.seedEdit.Value;
    partition = ud.partitionEdit.Value;
    mem = ud.memEdit.Value;
    
    cmd = sprintf(['python "%s" --data_path "%s" --data_type %s --task_type %s %s ' ...
        '--batch_size %d --epochs %d --learning_rate %s --val_split %.2f --seed %d --save_dir results'], ...
        trainPath, dataPath, dataType, taskArg, phaseArgs, ...
        batchSize, epochs, lr, valSplit, seed);
    
    if exist(outDir, 'dir') ~= 7, mkdir(outDir); end
    
    slurmPath = fullfile(outDir, 'run_training.sh');
    fid = fopen(slurmPath, 'w');
    fprintf(fid, '#!/bin/bash\n#SBATCH --job-name=prismt_%s\n#SBATCH --partition=%s\n#SBATCH --gres=gpu:1\n#SBATCH --mem=%dG\n#SBATCH --time=24:00:00\n#SBATCH --output=%%j.out\n#SBATCH --error=%%j.err\n\n', ...
        taskArg, partition, mem);
    fprintf(fid, 'cd "%s"\n%s\n', projectRoot, cmd);
    fclose(fid);
    
    configPath = fullfile(outDir, 'train_config.txt');
    fid = fopen(configPath, 'w');
    fprintf(fid, 'PRISMT Training Config\n======================\n');
    fprintf(fid, 'Data: %s\nTask: %s\n', dataPath, taskArg);
    if taskType
        fprintf(fid, 'Phases: %s vs %s\n', phase1, phase2);
    end
    fprintf(fid, 'Batch: %d Epochs: %d\n', batchSize, epochs);
    fprintf(fid, '\nLocal: %s\nCluster: sbatch %s\n', cmd, slurmPath);
    fclose(fid);
    
    ud.statusLabel.Text = sprintf('Saved: %s', slurmPath);
    showAlert(fig, sprintf('Scripts saved to:\n%s\n\nSubmit: sbatch %s', outDir, slurmPath), 'Done');
end

%% Classic fallback
function createClassicFigure()
    fig = figure('Name', 'PRISMT', 'NumberTitle', 'off', ...
        'Position', [100 100 500 300], 'Menubar', 'none');
    
    uicontrol(fig, 'Style', 'text', 'String', 'PRISMT Training Setup', ...
        'Position', [20 220 460 30], 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlign', 'center');
    uicontrol(fig, 'Style', 'text', 'String', ...
        'Requires MATLAB R2016a+. Run from project root: run_prismt_gui', ...
        'Position', [20 160 460 40], 'FontSize', 10, 'HorizontalAlign', 'center');
    uicontrol(fig, 'Style', 'pushbutton', 'String', 'Open PRISMT wiki', ...
        'Position', [180 100 140 30], 'Callback', @(~,~) web('https://github.com/josueortc/prismt', '-browser'));
end
