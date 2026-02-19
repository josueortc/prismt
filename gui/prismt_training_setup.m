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
    % Layout constants: fit on screen so Panel 1 stays visible (no scroll in uifigure)
    MAR = 20; W = 780;
    try
        ss = get(0, 'ScreenSize');
        scrH = max(600, ss(4) - 120);
        scrW = max(W, ss(3));
    catch
        scrH = 750; scrW = 800;
    end
    H = min(880, max(700, scrH));  % Min 700 to fit all panels
    figX = max(40, (scrW - W) / 2);
    fig = uifigure('Name', 'PRISMT - Training Setup', ...
        'Position', [figX 40 W H], ...
        'Color', [0.96 0.96 0.98]);
    
    % === HEADER (professional bar) ===
    headerPanel = uipanel(fig, 'Title', '', 'Position', [0 H-42 W 42], ...
        'BackgroundColor', [0.12 0.35 0.55], 'BorderType', 'none');
    uilabel(headerPanel, 'Text', 'PRISMT', 'Position', [20 8 140 24], ...
        'FontSize', 16, 'FontWeight', 'bold', 'FontColor', [1 1 1]);
    uilabel(headerPanel, 'Text', 'Transformer training | Optuna HPO', 'Position', [165 10 220 20], ...
        'FontSize', 10, 'FontColor', [0.85 0.9 1]);
    helpBtn = uibutton(headerPanel, 'Text', 'Help', 'Position', [W-120 6 85 28], ...
        'BackgroundColor', [0.2 0.45 0.7], 'FontColor', [1 1 1], ...
        'ButtonPushedFcn', @(src,~) openHelp());
    
    % === PANEL 1: Data ===
    p1w = W - 2*MAR; p1h = 120; p1y = H - 42 - 12 - p1h;
    p1 = uipanel(fig, 'Title', '1. Load Dataset', 'Position', [MAR p1y p1w p1h], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p1, 'Text', 'Dataset (.mat):', 'Position', [15 78 100 22]);
    pathField = uieditfield(p1, 'text', 'Position', [120 75 p1w-280 28], ...
        'Value', '');
    summaryLabel = uilabel(p1, 'Text', 'No data loaded. Click Browse and Load.', ...
        'Position', [15 25 p1w-50 45], 'FontColor', [0.4 0.4 0.4]);
    uibutton(p1, 'Text', 'Browse', 'Position', [p1w-145 73 55 32], ...
        'ButtonPushedFcn', @(src,~) browseForFile(pathField));
    uibutton(p1, 'Text', 'Load', 'Position', [p1w-80 73 55 32], ...
        'ButtonPushedFcn', @(src,~) validateDataset(pathField, summaryLabel));
    
    % === PANEL 2: Input & Tokenization ===
    p2h = 100; p2y = p1y - 10 - p2h;
    p2 = uipanel(fig, 'Title', '2. Input & Tokenization', 'Position', [MAR p2y p1w p2h], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p2, 'Text', 'Data type:', 'Position', [15 58 80 22]);
    dataTypeDD = uidropdown(p2, 'Position', [100 55 100 28], ...
        'Items', {'dff (ΔF/F)', 'zscore'}, 'Value', 'dff (ΔF/F)');
    
    uilabel(p2, 'Text', 'Normalization:', 'Position', [220 58 90 22]);
    normDD = uidropdown(p2, 'Position', [315 55 200 28], ...
        'Items', {'Scale ×20 (recommended)', 'Robust (median/IQR)', 'Percentile clip', 'None'}, ...
        'Value', 'Scale ×20 (recommended)');
    
    uilabel(p2, 'Text', 'Tokenization:', 'Position', [15 18 90 22]);
    tokenInfoLabel = uilabel(p2, 'Text', 'Spatial: 1 token per region (computed after load)', ...
        'Position', [110 15 p1w-130 28], 'FontColor', [0.4 0.4 0.4]);
    
    % === PANEL 3: Conditions ===
    p3h = 110; p3y = p2y - 10 - p3h;
    p3 = uipanel(fig, 'Title', '3. Comparison Conditions', 'Position', [MAR p3y p1w p3h], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p3, 'Text', 'Task:', 'Position', [15 72 50 22]);
    taskTypeDD = uidropdown(p3, 'Position', [70 69 180 28], ...
        'Items', {'Phase (early vs late)', 'Genotype (WT vs mutant)'}, ...
        'Value', 'Phase (early vs late)');
    
    uilabel(p3, 'Text', 'Phases:', 'Position', [270 72 60 22]);
    phase1DD = uidropdown(p3, 'Position', [335 69 80 28], ...
        'Items', {'early', 'mid', 'late'}, 'Value', 'early');
    uilabel(p3, 'Text', 'vs', 'Position', [420 72 20 22]);
    phase2DD = uidropdown(p3, 'Position', [445 69 80 28], ...
        'Items', {'early', 'mid', 'late'}, 'Value', 'late');
    
    uilabel(p3, 'Text', 'Stim:', 'Position', [15 32 50 22]);
    stimEdit = uieditfield(p3, 'text', 'Position', [70 29 60 28], 'Value', '1');
    uilabel(p3, 'Text', 'Response:', 'Position', [150 32 70 22]);
    responseEdit = uieditfield(p3, 'text', 'Position', [225 29 80 28], 'Value', '0, 1');
    
    uilabel(p3, 'Text', 'Seed:', 'Position', [330 32 50 22]);
    seedEdit = uieditfield(p3, 'numeric', 'Position', [385 29 60 28], 'Value', 42);
    
    % === TRAINING MODE ===
    modeY = p3y - 32;
    uilabel(fig, 'Text', 'Mode:', 'Position', [MAR modeY+3 50 22]);
    modeDD = uidropdown(fig, 'Position', [MAR+55 modeY 180 28], ...
        'Items', {'Standard training', 'HPO (Optuna)'}, 'Value', 'Standard training');
    uilabel(fig, 'Text', 'HPO trials:', 'Position', [MAR+250 modeY+3 75 22]);
    hpoTrialsEdit = uieditfield(fig, 'numeric', 'Position', [MAR+330 modeY 55 28], 'Value', 30);
    uilabel(fig, 'Text', 'Epochs/trial:', 'Position', [MAR+400 modeY+3 85 22]);
    hpoEpochsEdit = uieditfield(fig, 'numeric', 'Position', [MAR+490 modeY 55 28], 'Value', 30);

    % === PANEL 4: Training ===
    p4w = 420; p4h = 115; p4y = modeY - 10 - p4h;
    p4 = uipanel(fig, 'Title', '4. Training', 'Position', [MAR p4y p4w p4h], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p4, 'Text', 'Batch:', 'Position', [15 78 45 22]);
    batchEdit = uieditfield(p4, 'numeric', 'Position', [65 75 42 28], 'Value', 16);
    uilabel(p4, 'Text', 'Epochs:', 'Position', [112 78 48 22]);
    epochsEdit = uieditfield(p4, 'numeric', 'Position', [165 75 42 28], 'Value', 100);
    uilabel(p4, 'Text', 'LR:', 'Position', [212 78 25 22]);
    lrEdit = uieditfield(p4, 'text', 'Position', [242 75 52 28], 'Value', '5e-5');
    uilabel(p4, 'Text', 'Weight decay:', 'Position', [299 78 72 22]);
    weightDecayEdit = uieditfield(p4, 'text', 'Position', [376 75 40 28], 'Value', '1e-3');
    
    uilabel(p4, 'Text', 'Val split:', 'Position', [15 42 55 22]);
    valSplitEdit = uieditfield(p4, 'numeric', 'Position', [75 39 45 28], 'Value', 0.2, 'Limits', [0 1]);
    uilabel(p4, 'Text', 'Save dir:', 'Position', [125 42 58 22]);
    saveDirEdit = uieditfield(p4, 'text', 'Position', [188 39 p4w-268 28], 'Value', 'results');
    
    uilabel(p4, 'Text', 'Output dir:', 'Position', [15 8 70 22]);
    outputDirEdit = uieditfield(p4, 'text', 'Position', [90 5 p4w-105 28], ...
        'Value', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'generated_scripts'));
    
    % === PANEL 5: Model (hyperparameters) ===
    p5w = 280; p5 = uipanel(fig, 'Title', '5. Model', 'Position', [MAR+p4w+15 p4y p5w p4h], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p5, 'Text', 'Hidden:', 'Position', [15 78 48 22]);
    hiddenEdit = uieditfield(p5, 'numeric', 'Position', [68 75 42 28], 'Value', 128);
    uilabel(p5, 'Text', 'Heads:', 'Position', [115 78 45 22]);
    numHeadsEdit = uieditfield(p5, 'numeric', 'Position', [165 75 38 28], 'Value', 4);
    uilabel(p5, 'Text', 'Layers:', 'Position', [208 78 45 22]);
    numLayersEdit = uieditfield(p5, 'numeric', 'Position', [258 75 22 28], 'Value', 3);
    
    uilabel(p5, 'Text', 'FF dim:', 'Position', [15 42 50 22]);
    ffDimEdit = uieditfield(p5, 'numeric', 'Position', [70 39 48 28], 'Value', 256);
    uilabel(p5, 'Text', 'Dropout:', 'Position', [123 42 52 22]);
    dropoutEdit = uieditfield(p5, 'numeric', 'Position', [180 39 45 28], 'Value', 0.3, 'Limits', [0 1]);
    
    uilabel(p5, 'Text', 'Scheduler:', 'Position', [15 8 62 22]);
    schedulerDD = uidropdown(p5, 'Position', [82 5 110 28], ...
        'Items', {'cosine_warmup', 'cosine', 'reduce_on_plateau', 'step'}, 'Value', 'cosine_warmup');
    uilabel(p5, 'Text', 'Warmup:', 'Position', [197 8 50 22]);
    warmupEdit = uieditfield(p5, 'numeric', 'Position', [252 5 28 28], 'Value', 5);
    
    % === PANEL 6: Cluster (SLURM) ===
    p6h = 105; p6y = max(MAR, p4y - 10 - p6h);
    p6 = uipanel(fig, 'Title', '6. Cluster (SLURM)', 'Position', [MAR p6y p4w p6h], ...
        'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    
    uilabel(p6, 'Text', 'Partition:', 'Position', [15 72 48 22]);
    partitionEdit = uieditfield(p6, 'text', 'Position', [66 69 50 28], 'Value', 'gpu');
    uilabel(p6, 'Text', 'GPUs:', 'Position', [121 72 32 22]);
    gpusEdit = uieditfield(p6, 'numeric', 'Position', [156 69 32 28], 'Value', 1);
    uilabel(p6, 'Text', 'CPUs:', 'Position', [193 72 32 22]);
    cpusEdit = uieditfield(p6, 'numeric', 'Position', [228 69 32 28], 'Value', 8);
    uilabel(p6, 'Text', 'Mem:', 'Position', [265 72 30 22]);
    memEdit = uieditfield(p6, 'numeric', 'Position', [298 69 32 28], 'Value', 32);
    uilabel(p6, 'Text', 'Time(hr):', 'Position', [335 72 42 22]);
    timeEdit = uieditfield(p6, 'numeric', 'Position', [380 69 36 28], 'Value', 24);
    
    uilabel(p6, 'Text', 'Data path on cluster:', 'Position', [15 42 120 22]);
    clusterDataEdit = uieditfield(p6, 'text', 'Position', [140 39 p4w-155 28], 'Value', '');
    uilabel(p6, 'Text', 'HPO out dir (cluster):', 'Position', [15 12 115 22]);
    clusterOutEdit = uieditfield(p6, 'text', 'Position', [135 9 p4w-150 28], ...
        'Value', '');
    
    uilabel(p6, 'Text', 'Setup (conda activate, etc.):', 'Position', [15 2 155 22]);
    setupEdit = uieditfield(p6, 'text', 'Position', [170 2 p4w-185 22], 'Value', '');
    
    % === ACTION PANEL (right side) ===
    actionPanel = uipanel(fig, 'Title', 'Run', 'Position', [MAR+p4w+15 p6y p5w p6h], ...
        'BackgroundColor', [0.95 0.97 1], 'FontWeight', 'bold');
    
    runBtn = uibutton(actionPanel, 'Text', 'Run Training Now', ...
        'Position', [15 85 p5w-30 40], 'BackgroundColor', [0.2 0.55 0.35], ...
        'FontColor', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(src,~) runTraining(src.Parent.Parent));
    
    genBtn = uibutton(actionPanel, 'Text', 'Generate Cluster Script', ...
        'Position', [15 35 p5w-30 38], 'BackgroundColor', [0.25 0.5 0.8], ...
        'FontColor', [1 1 1], 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(src,~) generateScript(src.Parent.Parent));
    
    % === STATUS BAR (bottom, Suite2P-style) ===
    statusLabel = uilabel(fig, 'Text', 'Ready. Load a dataset to begin.', ...
        'Position', [MAR 0 W-2*MAR 20], 'FontColor', [0.4 0.4 0.4], 'FontSize', 10);
    
    % Store handles
    fig.UserData = struct(...
        'pathField', pathField, 'summaryLabel', summaryLabel, ...
        'dataTypeDD', dataTypeDD, 'normDD', normDD, 'tokenInfoLabel', tokenInfoLabel, ...
        'taskTypeDD', taskTypeDD, 'phase1DD', phase1DD, 'phase2DD', phase2DD, ...
        'stimEdit', stimEdit, 'responseEdit', responseEdit, 'seedEdit', seedEdit, ...
        'batchEdit', batchEdit, 'epochsEdit', epochsEdit, 'lrEdit', lrEdit, ...
        'weightDecayEdit', weightDecayEdit, 'valSplitEdit', valSplitEdit, 'saveDirEdit', saveDirEdit, ...
        'hiddenEdit', hiddenEdit, 'numHeadsEdit', numHeadsEdit, 'numLayersEdit', numLayersEdit, ...
        'ffDimEdit', ffDimEdit, 'dropoutEdit', dropoutEdit, 'schedulerDD', schedulerDD, 'warmupEdit', warmupEdit, ...
        'modeDD', modeDD, 'hpoTrialsEdit', hpoTrialsEdit, 'hpoEpochsEdit', hpoEpochsEdit, ...
        'outputDirEdit', outputDirEdit, 'partitionEdit', partitionEdit, 'memEdit', memEdit, ...
        'gpusEdit', gpusEdit, 'cpusEdit', cpusEdit, 'timeEdit', timeEdit, ...
        'clusterDataEdit', clusterDataEdit, 'clusterOutEdit', clusterOutEdit, 'setupEdit', setupEdit, ...
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

        % Populate condition dropdowns from data
        if isfield(fig.UserData, 'phase1DD') && isfield(info, 'phases') && ~isempty(info.phases)
            phases = info.phases;
            if ischar(phases), phases = {phases}; end
            if ~iscell(phases), phases = cellstr(string(phases)); end
            fig.UserData.phase1DD.Items = phases;
            fig.UserData.phase2DD.Items = phases;
            if numel(phases) >= 2
                try
                    fig.UserData.phase1DD.Value = phases{1};
                    fig.UserData.phase2DD.Value = phases{2};
                catch
                end
            end
        end
        if isfield(fig.UserData, 'stimEdit') && isfield(info, 'stim_values') && ~isempty(info.stim_values)
            fig.UserData.stimEdit.Value = strjoin(arrayfun(@num2str, info.stim_values, 'UniformOutput', false), ', ');
        end
        if isfield(fig.UserData, 'responseEdit') && isfield(info, 'response_values') && ~isempty(info.response_values)
            fig.UserData.responseEdit.Value = strjoin(arrayfun(@num2str, info.response_values, 'UniformOutput', false), ', ');
        end

        fig.UserData.statusLabel.Text = 'Data loaded. Select conditions and run.';
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
    useHpo = strcmp(ud.modeDD.Value, 'HPO (Optuna)');
    scriptName = 'hpo_optuna.py';
    if ~useHpo
        scriptName = 'train.py';
    end
    trainPath = fullfile(projectRoot, scriptName);
    
    if useHpo
        cmd = sprintf(['python "%s" --data_path "%s" --data_type %s --task_type %s %s ' ...
            '--n_trials %d --max_epochs %d --val_split %.2f --seed %d --out_dir %s'], ...
            trainPath, dataPath, dataType, taskArg, phaseArgs, ...
            ud.hpoTrialsEdit.Value, ud.hpoEpochsEdit.Value, ...
            ud.valSplitEdit.Value, ud.seedEdit.Value, ud.saveDirEdit.Value);
    else
        cmd = sprintf(['python "%s" --data_path "%s" --data_type %s --task_type %s %s ' ...
        '--batch_size %d --epochs %d --learning_rate %s --weight_decay %s --val_split %.2f --seed %d ' ...
        '--hidden_dim %d --num_heads %d --num_layers %d --ff_dim %d --dropout %.2f ' ...
        '--scheduler_type %s --warmup_epochs %d --save_dir %s'], ...
        trainPath, dataPath, dataType, taskArg, phaseArgs, ...
        ud.batchEdit.Value, ud.epochsEdit.Value, ud.lrEdit.Value, ud.weightDecayEdit.Value, ...
        ud.valSplitEdit.Value, ud.seedEdit.Value, ...
        ud.hiddenEdit.Value, ud.numHeadsEdit.Value, ud.numLayersEdit.Value, ud.ffDimEdit.Value, ud.dropoutEdit.Value, ...
        ud.schedulerDD.Value, ud.warmupEdit.Value, ud.saveDirEdit.Value);
    end

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
    
    clusterDataPath = strtrim(ud.clusterDataEdit.Value);
    if isempty(clusterDataPath)
        clusterDataPath = dataPath;
    end
    
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    outDir = ud.outputDirEdit.Value;
    batchSize = ud.batchEdit.Value;
    epochs = ud.epochsEdit.Value;
    lr = ud.lrEdit.Value;
    weightDecay = ud.weightDecayEdit.Value;
    valSplit = ud.valSplitEdit.Value;
    seed = ud.seedEdit.Value;
    saveDir = ud.saveDirEdit.Value;
    hiddenDim = ud.hiddenEdit.Value;
    numHeads = ud.numHeadsEdit.Value;
    numLayers = ud.numLayersEdit.Value;
    ffDim = ud.ffDimEdit.Value;
    dropout = ud.dropoutEdit.Value;
    scheduler = ud.schedulerDD.Value;
    warmup = ud.warmupEdit.Value;
    partition = ud.partitionEdit.Value;
    gpus = ud.gpusEdit.Value;
    cpus = ud.cpusEdit.Value;
    mem = ud.memEdit.Value;
    timeHrs = ud.timeEdit.Value;
    setupCmd = strtrim(ud.setupEdit.Value);
    useHpo = strcmp(ud.modeDD.Value, 'HPO (Optuna)');
    clusterOutDir = strtrim(ud.clusterOutEdit.Value);

    if useHpo
        nTrials = ud.hpoTrialsEdit.Value;
        maxEpochs = ud.hpoEpochsEdit.Value;
        outDirHpo = clusterOutDir;
        if isempty(outDirHpo)
            outDirHpo = saveDir;
        end
        % Use $OUTDIR in cmd; script will define OUTDIR for cluster resume
        cmd = sprintf(['OUTDIR="%s"\nmkdir -p "$OUTDIR" logs\n' ...
            'python hpo_optuna.py --data_path "%s" --data_type %s --task_type %s %s ' ...
            '--n_trials %d --max_epochs %d --val_split %.2f --seed %d --out_dir "$OUTDIR" ' ...
            '--storage "sqlite:///$OUTDIR/study.db"'], ...
            outDirHpo, clusterDataPath, dataType, taskArg, phaseArgs, ...
            nTrials, maxEpochs, valSplit, seed);
    else
        cmd = sprintf(['python train.py --data_path "%s" --data_type %s --task_type %s %s ' ...
            '--batch_size %d --epochs %d --learning_rate %s --weight_decay %s --val_split %.2f --seed %d ' ...
            '--hidden_dim %d --num_heads %d --num_layers %d --ff_dim %d --dropout %.2f ' ...
            '--scheduler_type %s --warmup_epochs %d --save_dir %s'], ...
            clusterDataPath, dataType, taskArg, phaseArgs, ...
            batchSize, epochs, lr, weightDecay, valSplit, seed, ...
            hiddenDim, numHeads, numLayers, ffDim, dropout, ...
            scheduler, warmup, saveDir);
    end
    
    if exist(outDir, 'dir') ~= 7, mkdir(outDir); end
    
    slurmPath = fullfile(outDir, 'run_training.sh');
    fid = fopen(slurmPath, 'w');
    fprintf(fid, '#!/bin/bash\n');
    fprintf(fid, '#SBATCH --job-name=prismt_%s\n', taskArg);
    fprintf(fid, '#SBATCH --partition=%s\n', partition);
    fprintf(fid, '#SBATCH --gres=gpu:%d\n', gpus);
    fprintf(fid, '#SBATCH --cpus-per-task=%d\n', cpus);
    fprintf(fid, '#SBATCH --mem=%dG\n', mem);
    fprintf(fid, '#SBATCH --time=%d:00:00\n', timeHrs);
    if useHpo
        fprintf(fid, '#SBATCH --output=logs/%%x_%%j.out\n');
        fprintf(fid, '#SBATCH --error=logs/%%x_%%j.err\n\n');
        fprintf(fid, 'set -euo pipefail\n\n');
    else
        fprintf(fid, '#SBATCH --output=%%j.out\n');
        fprintf(fid, '#SBATCH --error=%%j.err\n\n');
    end
    fprintf(fid, '# Project root: resolved from script location (works on cluster)\n');
    fprintf(fid, 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n');
    fprintf(fid, 'PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"\n');
    fprintf(fid, 'cd "$PROJECT_ROOT"\n\n');
    if useHpo
        fprintf(fid, 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n');
        fprintf(fid, 'export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n');
        fprintf(fid, 'export CUDA_VISIBLE_DEVICES=0\n\n');
    end
    if ~isempty(setupCmd)
        fprintf(fid, '# Optional setup (conda, modules)\n');
        fprintf(fid, '%s\n\n', setupCmd);
    end
    fprintf(fid, '%s\n', cmd);
    fclose(fid);
    
    configPath = fullfile(outDir, 'train_config.txt');
    fid = fopen(configPath, 'w');
    fprintf(fid, 'PRISMT Training Config\n======================\n');
    fprintf(fid, 'Mode: %s\n', ud.modeDD.Value);
    fprintf(fid, 'Data: %s\nCluster data: %s\nTask: %s\n', dataPath, clusterDataPath, taskArg);
    if taskType
        fprintf(fid, 'Phases: %s vs %s\n', phase1, phase2);
    end
    if useHpo
        fprintf(fid, 'HPO: n_trials=%d max_epochs/trial=%d out_dir=%s (resume via sqlite)\n', ...
            ud.hpoTrialsEdit.Value, ud.hpoEpochsEdit.Value, outDirHpo);
    else
        fprintf(fid, 'Batch: %d Epochs: %d LR: %s Weight decay: %s\n', batchSize, epochs, lr, weightDecay);
        fprintf(fid, 'Model: hidden=%d heads=%d layers=%d ff=%d dropout=%.2f\n', hiddenDim, numHeads, numLayers, ffDim, dropout);
        fprintf(fid, 'Scheduler: %s warmup=%d\n', scheduler, warmup);
    end
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
