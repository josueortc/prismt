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

%% Main UI (responsive grid layout, requires R2019b+ for uigridlayout)
function createUIFigure()
    if ~exist('uigridlayout', 'file')
        createUIFigureLegacy();
        return;
    end
    MAR = 16; PAD = 8;
    try
        ss = get(0, 'ScreenSize');
        scrH = max(600, ss(4) - 100);
        scrW = max(900, min(1200, ss(3) - 80));
    catch
        scrH = 800; scrW = 1000;
    end
    figX = max(20, (ss(3) - scrW) / 2);
    fig = uifigure('Name', 'PRISMT - Training Setup', ...
        'Position', [figX 20 scrW scrH], ...
        'Color', [0.97 0.97 0.98], ...
        'Resize', 'on');

    % Main grid: responsive layout that grows with window
    mainGrid = uigridlayout(fig, [8 2]);
    mainGrid.RowHeight = {42, 115, 130, 170, 40, '1x', '1x', 28};
    mainGrid.ColumnWidth = {'1x', 300};
    mainGrid.Padding = [MAR MAR MAR MAR];
    mainGrid.RowSpacing = PAD;
    mainGrid.ColumnSpacing = PAD;

    % === HEADER (spans both columns) ===
    headerPanel = uipanel(mainGrid, 'Title', '', 'BackgroundColor', [0.11 0.33 0.52], 'BorderType', 'none');
    headerPanel.Layout.Row = 1;
    headerPanel.Layout.Column = [1 2];
    headerGl = uigridlayout(headerPanel, [1 3]);
    headerGl.ColumnWidth = {'fit', '1x', 'fit'};
    headerGl.Padding = [0 0 0 0];
    uilabel(headerGl, 'Text', 'PRISMT', 'FontSize', 17, 'FontWeight', 'bold', 'FontColor', [1 1 1]);
    uilabel(headerGl, 'Text', 'Transformer training | Optuna HPO', 'FontSize', 10, 'FontColor', [0.9 0.93 1]);
    uibutton(headerGl, 'Text', 'Help', 'BackgroundColor', [0.25 0.5 0.75], 'FontColor', [1 1 1], ...
        'ButtonPushedFcn', @(src,~) openHelp());

    % === PANEL 1: Data (spans both columns) ===
    p1 = uipanel(mainGrid, 'Title', '1. Load Dataset', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p1.Layout.Row = 2;
    p1.Layout.Column = [1 2];
    gl1 = uigridlayout(p1, [2 4]);
    gl1.ColumnWidth = {'fit', '1x', 'fit', 'fit'};
    gl1.RowHeight = {32, '1x'};
    gl1.Padding = [12 12 12 12];
    gl1.RowSpacing = 6;
    uilabel(gl1, 'Text', 'Dataset (.mat):');
    pathField = uieditfield(gl1, 'text', 'Value', '');
    uibutton(gl1, 'Text', 'Browse', 'ButtonPushedFcn', @(src,~) browseForFile(pathField));
    uibutton(gl1, 'Text', 'Load', 'ButtonPushedFcn', @(src,~) validateDataset(pathField, summaryLabel));
    summaryLabel = uilabel(gl1, 'Text', 'No data loaded. Click Browse and Load.', 'FontColor', [0.45 0.45 0.45]);
    summaryLabel.Layout.Row = 2;
    summaryLabel.Layout.Column = [1 4];

    % === PANEL 2: Input & Tokenization ===
    p2 = uipanel(mainGrid, 'Title', '2. Input & Tokenization', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p2.Layout.Row = 3;
    p2.Layout.Column = [1 2];
    gl2 = uigridlayout(p2, [3 6]);
    gl2.ColumnWidth = {'fit', 100, 'fit', 200, 'fit', '1x'};
    gl2.RowHeight = {32, 32, 32};
    gl2.Padding = [12 12 12 12];
    gl2.RowSpacing = 6;
    uilabel(gl2, 'Text', 'Data type:');
    dataTypeDD = uidropdown(gl2, 'Items', {'dff (ΔF/F)', 'zscore'}, 'Value', 'dff (ΔF/F)');
    uilabel(gl2, 'Text', 'Normalization:');
    normDD = uidropdown(gl2, 'Items', {'Scale ×20 (recommended)', 'Robust (median/IQR)', 'Percentile clip', 'None'}, ...
        'Value', 'Scale ×20 (recommended)');
    uilabel(gl2, 'Text', 'Region pool:');
    regionPoolDD = uidropdown(gl2, 'Items', {'None (1)', '2 (avg pairs)', '4', '8'}, 'Value', 'None (1)');
    uilabel(gl2, 'Text', 'Time pool:');
    timePoolDD = uidropdown(gl2, 'Items', {'None (1)', '2 (avg pairs)', '4', '8'}, 'Value', 'None (1)');
    tokLbl = uilabel(gl2, 'Text', 'Tokenization:');
    tokLbl.Layout.Row = 3;
    tokLbl.Layout.Column = 1;
    tokenInfoLabel = uilabel(gl2, 'Text', 'Load data to see tokenization.', 'FontColor', [0.45 0.45 0.45]);
    tokenInfoLabel.Layout.Row = 3;
    tokenInfoLabel.Layout.Column = [2 6];

    % === PANEL 3: Conditions ===
    p3 = uipanel(mainGrid, 'Title', '3. Comparison Conditions', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p3.Layout.Row = 4;
    p3.Layout.Column = [1 2];
    gl3 = uigridlayout(p3, [4 10]);
    gl3.ColumnWidth = {'fit', 100, 'fit', 95, 'fit', 130, 'fit', 85, 'fit', 85};
    gl3.RowHeight = {32, 32, 32, 32};
    gl3.Padding = [12 12 12 12];
    gl3.RowSpacing = 6;
    uilabel(gl3, 'Text', 'Mode:');
    taskModeDD = uidropdown(gl3, 'Items', {'Classification', 'Regression'}, 'Value', 'Classification');
    uilabel(gl3, 'Text', 'Target column:');
    targetColDD = uidropdown(gl3, 'Items', {'phase', 'mouse', 'stim', 'response'}, 'Value', 'phase');
    classTypeLabel = uilabel(gl3, 'Text', 'Class type:');
    classTypeDD = uidropdown(gl3, 'Items', {'Multiclass (all values)', 'Binary (select 2)'}, 'Value', 'Binary (select 2)');
    class1Label = uilabel(gl3, 'Text', 'Class 1:');
    class1DD = uidropdown(gl3, 'Items', {'early', 'mid', 'late'}, 'Value', 'early');
    class2Label = uilabel(gl3, 'Text', 'vs Class 2:');
    class2DD = uidropdown(gl3, 'Items', {'early', 'mid', 'late'}, 'Value', 'late');
    uilabel(gl3, 'Text', 'Filter stim:');
    stimEdit = uieditfield(gl3, 'text', 'Value', '1');
    uilabel(gl3, 'Text', 'Filter response:');
    responseEdit = uieditfield(gl3, 'text', 'Value', '0, 1');
    phaseFilterLabel = uilabel(gl3, 'Text', 'Filter phase:');
    phaseFilterEdit = uieditfield(gl3, 'text', 'Value', 'early, mid, late');
    uilabel(gl3, 'Text', 'Seed:');
    seedEdit = uieditfield(gl3, 'numeric', 'Value', 42);

    % === MODE ROW ===
    modePanel = uipanel(mainGrid, 'Title', '', 'BackgroundColor', [0.98 0.98 1], 'BorderType', 'none');
    modePanel.Layout.Row = 5;
    modePanel.Layout.Column = [1 2];
    modeGl = uigridlayout(modePanel, [1 6]);
    modeGl.ColumnWidth = {'fit', 180, 'fit', 55, 'fit', 55};
    modeGl.Padding = [4 4 4 4];
    uilabel(modeGl, 'Text', 'Mode:');
    modeDD = uidropdown(modeGl, 'Items', {'Standard training', 'HPO (Optuna)'}, 'Value', 'Standard training');
    hpoTrialsLabel = uilabel(modeGl, 'Text', 'HPO trials:');
    hpoTrialsEdit = uieditfield(modeGl, 'numeric', 'Value', 30);
    hpoEpochsLabel = uilabel(modeGl, 'Text', 'Epochs/trial:');
    hpoEpochsEdit = uieditfield(modeGl, 'numeric', 'Value', 30);

    % === PANEL 4: Training (left column) ===
    p4 = uipanel(mainGrid, 'Title', '4. Training', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p4.Layout.Row = 6;
    p4.Layout.Column = 1;
    gl4 = uigridlayout(p4, [4 4]);
    gl4.ColumnWidth = {'fit', '1x', 'fit', '1x'};
    gl4.RowHeight = {28, 28, 28, 28};
    gl4.Padding = [12 12 12 12];
    gl4.RowSpacing = 6;
    uilabel(gl4, 'Text', 'Batch:'); batchEdit = uieditfield(gl4, 'numeric', 'Value', 16);
    uilabel(gl4, 'Text', 'Epochs:'); epochsEdit = uieditfield(gl4, 'numeric', 'Value', 100);
    uilabel(gl4, 'Text', 'LR:'); lrEdit = uieditfield(gl4, 'text', 'Value', '5e-5');
    uilabel(gl4, 'Text', 'Weight decay:'); weightDecayEdit = uieditfield(gl4, 'text', 'Value', '1e-3');
    uilabel(gl4, 'Text', 'Val split:'); valSplitEdit = uieditfield(gl4, 'numeric', 'Value', 0.2, 'Limits', [0 1]);
    uilabel(gl4, 'Text', 'Save dir:'); saveDirEdit = uieditfield(gl4, 'text', 'Value', 'results');
    uilabel(gl4, 'Text', 'Output dir:');
    outputDirEdit = uieditfield(gl4, 'text', 'Value', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'generated_scripts'));
    outputDirEdit.Layout.Column = [2 4];

    % === PANEL 5: Model (right column) ===
    p5 = uipanel(mainGrid, 'Title', '5. Model', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p5.Layout.Row = 6;
    p5.Layout.Column = 2;
    gl5 = uigridlayout(p5, [4 4]);
    gl5.ColumnWidth = {'fit', '1x', 'fit', '1x'};
    gl5.RowHeight = {28, 28, 28, 28};
    gl5.Padding = [12 12 12 12];
    gl5.RowSpacing = 6;
    uilabel(gl5, 'Text', 'Hidden:'); hiddenEdit = uieditfield(gl5, 'numeric', 'Value', 128);
    uilabel(gl5, 'Text', 'Heads:'); numHeadsEdit = uieditfield(gl5, 'numeric', 'Value', 4);
    uilabel(gl5, 'Text', 'Layers:'); numLayersEdit = uieditfield(gl5, 'numeric', 'Value', 3);
    uilabel(gl5, 'Text', 'FF dim:'); ffDimEdit = uieditfield(gl5, 'numeric', 'Value', 256);
    uilabel(gl5, 'Text', 'Dropout:'); dropoutEdit = uieditfield(gl5, 'numeric', 'Value', 0.3, 'Limits', [0 1]);
    uilabel(gl5, 'Text', 'Scheduler:');
    schedulerDD = uidropdown(gl5, 'Items', {'cosine_warmup', 'cosine', 'reduce_on_plateau', 'step'}, 'Value', 'cosine_warmup');
    uilabel(gl5, 'Text', 'Warmup:'); warmupEdit = uieditfield(gl5, 'numeric', 'Value', 5);

    % === PANEL 6: Cluster (left column) ===
    p6 = uipanel(mainGrid, 'Title', '6. Cluster (SLURM)', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p6.Layout.Row = 7;
    p6.Layout.Column = 1;
    gl6 = uigridlayout(p6, [4 6]);
    gl6.ColumnWidth = {'fit', 'fit', 'fit', 'fit', 'fit', '1x'};
    gl6.RowHeight = {28, 28, 28, 28};
    gl6.Padding = [12 12 12 12];
    gl6.RowSpacing = 6;
    uilabel(gl6, 'Text', 'Partition:'); partitionEdit = uieditfield(gl6, 'text', 'Value', 'gpu');
    uilabel(gl6, 'Text', 'GPUs:'); gpusEdit = uieditfield(gl6, 'numeric', 'Value', 1);
    uilabel(gl6, 'Text', 'CPUs:'); cpusEdit = uieditfield(gl6, 'numeric', 'Value', 8);
    uilabel(gl6, 'Text', 'Mem:'); memEdit = uieditfield(gl6, 'numeric', 'Value', 32);
    uilabel(gl6, 'Text', 'Time(hr):'); timeEdit = uieditfield(gl6, 'numeric', 'Value', 24);
    uilabel(gl6, 'Text', 'Data path on cluster:', 'Layout', struct('Row', 2, 'Column', [1 2]));
    clusterDataEdit = uieditfield(gl6, 'text', 'Value', '');
    clusterDataEdit.Layout.Row = 2;
    clusterDataEdit.Layout.Column = [3 6];
    clusterOutLabel = uilabel(gl6, 'Text', 'HPO out dir (cluster):');
    clusterOutLabel.Layout.Row = 3;
    clusterOutLabel.Layout.Column = [1 2];
    clusterOutEdit = uieditfield(gl6, 'text', 'Value', '');
    clusterOutEdit.Layout.Row = 3;
    clusterOutEdit.Layout.Column = [3 6];
    uilabel(gl6, 'Text', 'Setup (conda activate, etc.):', 'Layout', struct('Row', 4, 'Column', [1 2]));
    setupEdit = uieditfield(gl6, 'text', 'Value', '');
    setupEdit.Layout.Row = 4;
    setupEdit.Layout.Column = [3 6];

    % === ACTION PANEL (right column) ===
    actionPanel = uipanel(mainGrid, 'Title', 'Run', 'BackgroundColor', [0.96 0.97 1], 'FontWeight', 'bold');
    actionPanel.Layout.Row = 7;
    actionPanel.Layout.Column = 2;
    actionGl = uigridlayout(actionPanel, [2 1]);
    actionGl.RowHeight = {'1x', '1x'};
    actionGl.Padding = [12 12 12 12];
    actionGl.RowSpacing = 10;
    runBtn = uibutton(actionGl, 'Text', 'Run Training Now', 'BackgroundColor', [0.25 0.55 0.35], ...
        'FontColor', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(src,~) runTraining(src.Parent.Parent.Parent));
    genBtn = uibutton(actionGl, 'Text', 'Generate Cluster Script', 'BackgroundColor', [0.28 0.52 0.8], ...
        'FontColor', [1 1 1], 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(src,~) generateScript(src.Parent.Parent.Parent));

    % === STATUS BAR ===
    statusPanel = uipanel(mainGrid, 'Title', '', 'BackgroundColor', [0.94 0.94 0.96], 'BorderType', 'none');
    statusPanel.Layout.Row = 8;
    statusPanel.Layout.Column = [1 2];
    statusGl = uigridlayout(statusPanel, [1 1]);
    statusGl.Padding = [6 4 6 4];
    statusLabel = uilabel(statusGl, 'Text', 'Ready. Load a dataset to begin.', ...
        'FontColor', [0.45 0.45 0.45], 'FontSize', 10);
    
    % Fix runTraining/genBtn parent: fig is mainGrid.Parent
    runBtn.ButtonPushedFcn = @(src,~) runTraining(mainGrid.Parent);
    genBtn.ButtonPushedFcn = @(src,~) generateScript(mainGrid.Parent);
    
    % Store handles
    fig.UserData = struct(...
        'pathField', pathField, 'summaryLabel', summaryLabel, ...
        'dataTypeDD', dataTypeDD, 'normDD', normDD, 'tokenInfoLabel', tokenInfoLabel, ...
        'regionPoolDD', regionPoolDD, 'timePoolDD', timePoolDD, ...
        'taskModeDD', taskModeDD, 'targetColDD', targetColDD, ...
        'classTypeDD', classTypeDD, 'class1Label', class1Label, 'class1DD', class1DD, ...
        'class2Label', class2Label, 'class2DD', class2DD, ...
        'phaseFilterLabel', phaseFilterLabel, 'phaseFilterEdit', phaseFilterEdit, ...
        'stimEdit', stimEdit, 'responseEdit', responseEdit, 'seedEdit', seedEdit, ...
        'batchEdit', batchEdit, 'epochsEdit', epochsEdit, 'lrEdit', lrEdit, ...
        'weightDecayEdit', weightDecayEdit, 'valSplitEdit', valSplitEdit, 'saveDirEdit', saveDirEdit, ...
        'hiddenEdit', hiddenEdit, 'numHeadsEdit', numHeadsEdit, 'numLayersEdit', numLayersEdit, ...
        'ffDimEdit', ffDimEdit, 'dropoutEdit', dropoutEdit, 'schedulerDD', schedulerDD, 'warmupEdit', warmupEdit, ...
        'modeDD', modeDD, 'hpoTrialsLabel', hpoTrialsLabel, 'hpoTrialsEdit', hpoTrialsEdit, ...
        'hpoEpochsLabel', hpoEpochsLabel, 'hpoEpochsEdit', hpoEpochsEdit, ...
        'clusterOutLabel', clusterOutLabel, 'clusterOutEdit', clusterOutEdit, ...
        'outputDirEdit', outputDirEdit, 'partitionEdit', partitionEdit, 'memEdit', memEdit, ...
        'gpusEdit', gpusEdit, 'cpusEdit', cpusEdit, 'timeEdit', timeEdit, ...
        'clusterDataEdit', clusterDataEdit, 'setupEdit', setupEdit, ...
        'statusLabel', statusLabel, 'runBtn', runBtn, 'genBtn', genBtn, ...
        'dataInfo', struct());
    
    % Decision tree: show HPO options only when HPO mode selected
    fig = mainGrid.Parent;
    modeDD.ValueChangedFcn = @(src,~) updateModeDependentUI(fig);
    updateModeDependentUI(fig);
    
    pathField.ValueChangedFcn = @(src,~) setStatus(fig.UserData.statusLabel, ...
        iif(exist(src.Value, 'file')==2, 'Path OK. Click Load to validate.', 'Enter dataset path.'));
    regionPoolDD.ValueChangedFcn = @(src,~) refreshTokenInfo(fig);
    timePoolDD.ValueChangedFcn = @(src,~) refreshTokenInfo(fig);
    targetColDD.ValueChangedFcn = @(src,~) updateTargetDependentUI(fig);
    classTypeDD.ValueChangedFcn = @(src,~) updateTargetDependentUI(fig);
end

function updateTargetDependentUI(fig)
    if ~isfield(fig.UserData, 'dataInfo') || isempty(fig.UserData.dataInfo), return; end
    ud = fig.UserData;
    info = ud.dataInfo;
    targetCol = ud.targetColDD.Value;
    isMulticlass = strcmp(ud.classTypeDD.Value, 'Multiclass (all values)');
    % Show/hide binary class dropdowns and labels
    vis = iif(isMulticlass, 'off', 'on');
    for f = {'class1Label', 'class1DD', 'class2Label', 'class2DD'}
        if isfield(ud, f{1}) && isvalid(ud.(f{1}))
            ud.(f{1}).Visible = vis;
        end
    end
    % Populate class dropdowns from target column values
    if isfield(info, 'column_values') && isfield(info.column_values, targetCol)
        vals = info.column_values.(targetCol);
        if isnumeric(vals), vals = arrayfun(@num2str, vals, 'UniformOutput', false); end
        if ischar(vals), vals = {vals}; end
        if ~iscell(vals), vals = cellstr(string(vals)); end
        if isfield(ud, 'class1DD') && isvalid(ud.class1DD)
            ud.class1DD.Items = vals;
            if ~isempty(vals), ud.class1DD.Value = vals{1}; end
        end
        if isfield(ud, 'class2DD') && isvalid(ud.class2DD)
            ud.class2DD.Items = vals;
            if numel(vals) >= 2, ud.class2DD.Value = vals{2}; else, ud.class2DD.Value = vals{1}; end
        end
    end
    % Show/hide phase filter when target is phase
    vis = iif(strcmp(targetCol, 'phase'), 'off', 'on');
    if isfield(ud, 'phaseFilterEdit') && isvalid(ud.phaseFilterEdit)
        ud.phaseFilterEdit.Visible = vis;
    end
    if isfield(ud, 'phaseFilterLabel') && isvalid(ud.phaseFilterLabel)
        ud.phaseFilterLabel.Visible = vis;
    end
end

function refreshTokenInfo(fig)
    if ~isfield(fig.UserData, 'dataInfo') || isempty(fig.UserData.dataInfo), return; end
    info = fig.UserData.dataInfo;
    rp = parsePoolValue(fig.UserData.regionPoolDD.Value);
    tp = parsePoolValue(fig.UserData.timePoolDD.Value);
    R = floor(info.n_regions / rp);
    T = floor(info.n_timepoints / tp);
    fig.UserData.tokenInfoLabel.Text = sprintf(...
        'Spatial: %d×%d → %d regions → %d tokens (+ CLS)', T, R, R, R + 1);
end

function out = iif(cond, a, b)
    if cond, out = a; else, out = b; end
end

function setStatus(lbl, txt)
    if isvalid(lbl), lbl.Text = txt; end
end

function updateModeDependentUI(fig)
    % Decision tree: show/hide options based on Mode selection
    if ~isvalid(fig) || ~isfield(fig.UserData, 'modeDD'), return; end
    ud = fig.UserData;
    isHpo = strcmp(ud.modeDD.Value, 'HPO (Optuna)');
    vis = iif(isHpo, 'on', 'off');
    if isfield(ud, 'hpoTrialsLabel') && isvalid(ud.hpoTrialsLabel)
        ud.hpoTrialsLabel.Visible = vis;
    end
    if isfield(ud, 'hpoTrialsEdit') && isvalid(ud.hpoTrialsEdit)
        ud.hpoTrialsEdit.Visible = vis;
    end
    if isfield(ud, 'hpoEpochsLabel') && isvalid(ud.hpoEpochsLabel)
        ud.hpoEpochsLabel.Visible = vis;
    end
    if isfield(ud, 'hpoEpochsEdit') && isvalid(ud.hpoEpochsEdit)
        ud.hpoEpochsEdit.Visible = vis;
    end
    if isfield(ud, 'clusterOutLabel') && isvalid(ud.clusterOutLabel)
        ud.clusterOutLabel.Visible = vis;
    end
    if isfield(ud, 'clusterOutEdit') && isvalid(ud.clusterOutEdit)
        ud.clusterOutEdit.Visible = vis;
    end
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

function rp = parsePoolValue(str)
    % Parse region/time pool from dropdown (e.g. 'None (1)' -> 1, '2 (avg pairs)' -> 2)
    if contains(str, 'None') || strcmp(str, '1'), rp = 1; return; end
    tok = regexp(str, '\d+', 'match');
    if ~isempty(tok), rp = str2double(tok{1}); else, rp = 1; end
end

function validateDataset(pathField, summaryLabel)
    pathStr = strtrim(pathField.Value);
    fig = ancestor(pathField, 'figure');

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
            rp = 1; tp = 1;
            if isfield(fig.UserData, 'regionPoolDD') && isvalid(fig.UserData.regionPoolDD)
                rp = parsePoolValue(fig.UserData.regionPoolDD.Value);
            end
            if isfield(fig.UserData, 'timePoolDD') && isvalid(fig.UserData.timePoolDD)
                tp = parsePoolValue(fig.UserData.timePoolDD.Value);
            end
            R = floor(info.n_regions / rp);
            T = floor(info.n_timepoints / tp);
            fig.UserData.tokenInfoLabel.Text = sprintf(...
                'Spatial: %d×%d → %d regions → %d tokens (+ CLS)', ...
                T, R, R, R + 1);
        end

        % Populate condition dropdowns from dataset columns (dataset-agnostic)
        if isfield(fig.UserData, 'targetColDD') && isfield(info, 'column_names') && ~isempty(info.column_names)
            fig.UserData.targetColDD.Items = info.column_names;
            if ismember('phase', info.column_names)
                fig.UserData.targetColDD.Value = 'phase';
            else
                fig.UserData.targetColDD.Value = info.column_names{1};
            end
        end
        if isfield(fig.UserData, 'class1DD') && isfield(info, 'column_values')
            targetCol = fig.UserData.targetColDD.Value;
            if isfield(info.column_values, targetCol)
                vals = info.column_values.(targetCol);
                if isnumeric(vals), vals = arrayfun(@num2str, vals, 'UniformOutput', false); end
                if ischar(vals), vals = {vals}; end
                if ~iscell(vals), vals = cellstr(string(vals)); end
                fig.UserData.class1DD.Items = vals;
                fig.UserData.class2DD.Items = vals;
                if ~isempty(vals)
                    fig.UserData.class1DD.Value = vals{1};
                    fig.UserData.class2DD.Value = vals{min(2,end)};
                end
            end
        end
        updateTargetDependentUI(fig);
        if isfield(fig.UserData, 'phaseFilterEdit') && isfield(info, 'column_values') && isfield(info.column_values, 'phase')
            pv = info.column_values.phase;
            if iscell(pv), pv = strjoin(pv, ', '); elseif isnumeric(pv), pv = strjoin(arrayfun(@num2str, pv, 'UniformOutput', false), ', '); end
            fig.UserData.phaseFilterEdit.Value = pv;
        end
        if isfield(fig.UserData, 'stimEdit') && isfield(info, 'column_values') && isfield(info.column_values, 'stim')
            fig.UserData.stimEdit.Value = strjoin(arrayfun(@num2str, info.column_values.stim, 'UniformOutput', false), ', ');
        elseif isfield(fig.UserData, 'stimEdit') && isfield(info, 'stim_values') && ~isempty(info.stim_values)
            fig.UserData.stimEdit.Value = strjoin(arrayfun(@num2str, info.stim_values, 'UniformOutput', false), ', ');
        end
        if isfield(fig.UserData, 'responseEdit') && isfield(info, 'column_values') && isfield(info.column_values, 'response')
            fig.UserData.responseEdit.Value = strjoin(arrayfun(@num2str, info.column_values.response, 'UniformOutput', false), ', ');
        elseif isfield(fig.UserData, 'responseEdit') && isfield(info, 'response_values') && ~isempty(info.response_values)
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
    
    taskMode = strcmp(ud.taskModeDD.Value, 'Classification');
    taskModeArg = iif(taskMode, 'classification', 'regression');
    targetCol = ud.targetColDD.Value;
    dataType = strsplit(ud.dataTypeDD.Value, ' ');
    dataType = dataType{1};
    stimStr = strrep(strtrim(ud.stimEdit.Value), ' ', '');
    respStr = strrep(strtrim(ud.responseEdit.Value), ' ', '');
    condArgs = sprintf('--task_mode %s --target_column %s', taskModeArg, targetCol);
    if taskMode
        isMulticlass = strcmp(ud.classTypeDD.Value, 'Multiclass (all values)');
        if ~isMulticlass
            c1 = char(ud.class1DD.Value);
            c2 = char(ud.class2DD.Value);
            condArgs = [condArgs sprintf(' --target_values %s,%s', c1, c2)];
        end
    end
    filtersCell = {sprintf('"stim":[%s]', stimStr), sprintf('"response":[%s]', respStr)};
    if ~strcmp(targetCol, 'phase') && isfield(ud, 'phaseFilterEdit') && strcmp(ud.phaseFilterEdit.Visible, 'on')
        phaseStr = strtrim(ud.phaseFilterEdit.Value);
        if ~isempty(phaseStr)
            phaseParts = strsplit(phaseStr, ',');
            phaseParts = cellfun(@(p) sprintf('"%s"', strtrim(p)), phaseParts, 'UniformOutput', false);
            filtersCell{end+1} = sprintf('"phase":[%s]', strjoin(phaseParts, ','));
        end
    end
    filtersJson = ['{' strjoin(filtersCell, ',') '}'];
    condArgs = [condArgs ' --filters ''' filtersJson ''''];
    
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    useHpo = strcmp(ud.modeDD.Value, 'HPO (Optuna)');
    scriptName = 'hpo_optuna.py';
    if ~useHpo
        scriptName = 'train.py';
    end
    trainPath = fullfile(projectRoot, scriptName);
    
    rp = parsePoolValue(ud.regionPoolDD.Value);
    tp = parsePoolValue(ud.timePoolDD.Value);
    poolArgs = sprintf('--region_pool %d --time_pool %d', rp, tp);
    if useHpo
        cmd = sprintf(['python "%s" --data_path "%s" --data_type %s %s %s ' ...
            '--n_trials %d --max_epochs %d --val_split %.2f --seed %d --out_dir %s'], ...
            trainPath, dataPath, dataType, condArgs, poolArgs, ...
            ud.hpoTrialsEdit.Value, ud.hpoEpochsEdit.Value, ...
            ud.valSplitEdit.Value, ud.seedEdit.Value, ud.saveDirEdit.Value);
    else
        cmd = sprintf(['python "%s" --data_path "%s" --data_type %s %s %s ' ...
        '--batch_size %d --epochs %d --learning_rate %s --weight_decay %s --val_split %.2f --seed %d ' ...
        '--hidden_dim %d --num_heads %d --num_layers %d --ff_dim %d --dropout %.2f ' ...
        '--scheduler_type %s --warmup_epochs %d --save_dir %s'], ...
        trainPath, dataPath, dataType, condArgs, poolArgs, ...
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
    
    if ~isfield(ud, 'taskModeDD'), ud.taskModeDD = []; ud.targetColDD = []; end
    taskMode = isempty(ud.taskModeDD) || strcmp(ud.taskModeDD.Value, 'Classification');
    taskModeArg = iif(taskMode, 'classification', 'regression');
    targetCol = iif(isfield(ud, 'targetColDD') && isvalid(ud.targetColDD), ud.targetColDD.Value, 'phase');
    dataType = strsplit(ud.dataTypeDD.Value, ' ');
    dataType = dataType{1};
    stimStr = strrep(strtrim(ud.stimEdit.Value), ' ', '');
    respStr = strrep(strtrim(ud.responseEdit.Value), ' ', '');
    condArgs = sprintf('--task_mode %s --target_column %s', taskModeArg, targetCol);
    if taskMode
        isMulticlass = strcmp(ud.classTypeDD.Value, 'Multiclass (all values)');
        if ~isMulticlass
            c1 = char(ud.class1DD.Value);
            c2 = char(ud.class2DD.Value);
            condArgs = [condArgs sprintf(' --target_values %s,%s', c1, c2)];
        end
    end
    filtersCell = {sprintf('"stim":[%s]', stimStr), sprintf('"response":[%s]', respStr)};
    if ~strcmp(targetCol, 'phase') && isfield(ud, 'phaseFilterEdit') && strcmp(ud.phaseFilterEdit.Visible, 'on')
        phaseStr = strtrim(ud.phaseFilterEdit.Value);
        if ~isempty(phaseStr)
            phaseParts = strsplit(phaseStr, ',');
            phaseParts = cellfun(@(p) sprintf('"%s"', strtrim(p)), phaseParts, 'UniformOutput', false);
            filtersCell{end+1} = sprintf('"phase":[%s]', strjoin(phaseParts, ','));
        end
    end
    filtersJson = ['{' strjoin(filtersCell, ',') '}'];
    condArgs = [condArgs ' --filters ''' filtersJson ''''];
    
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
    rp = parsePoolValue(ud.regionPoolDD.Value);
    tp = parsePoolValue(ud.timePoolDD.Value);
    poolArgs = sprintf('--region_pool %d --time_pool %d', rp, tp);

    if useHpo
        nTrials = ud.hpoTrialsEdit.Value;
        maxEpochs = ud.hpoEpochsEdit.Value;
        outDirHpo = clusterOutDir;
        if isempty(outDirHpo)
            outDirHpo = saveDir;
        end
        % Use $OUTDIR in cmd; script will define OUTDIR for cluster resume
        cmd = sprintf(['OUTDIR="%s"\nmkdir -p "$OUTDIR" logs\n' ...
            'python hpo_optuna.py --data_path "%s" --data_type %s %s %s ' ...
            '--n_trials %d --max_epochs %d --val_split %.2f --seed %d --out_dir "$OUTDIR" ' ...
            '--storage "sqlite:///$OUTDIR/study.db"'], ...
            outDirHpo, clusterDataPath, dataType, condArgs, poolArgs, ...
            nTrials, maxEpochs, valSplit, seed);
    else
        cmd = sprintf(['python train.py --data_path "%s" --data_type %s %s %s ' ...
            '--batch_size %d --epochs %d --learning_rate %s --weight_decay %s --val_split %.2f --seed %d ' ...
            '--hidden_dim %d --num_heads %d --num_layers %d --ff_dim %d --dropout %.2f ' ...
            '--scheduler_type %s --warmup_epochs %d --save_dir %s'], ...
            clusterDataPath, dataType, condArgs, poolArgs, ...
            batchSize, epochs, lr, weightDecay, valSplit, seed, ...
            hiddenDim, numHeads, numLayers, ffDim, dropout, ...
            scheduler, warmup, saveDir);
    end
    
    if exist(outDir, 'dir') ~= 7, mkdir(outDir); end
    
    slurmPath = fullfile(outDir, 'run_training.sh');
    fid = fopen(slurmPath, 'w');
    fprintf(fid, '#!/bin/bash\n');
    fprintf(fid, '#SBATCH --job-name=prismt_%s\n', targetCol);
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
    fprintf(fid, 'Data: %s\nCluster data: %s\nTask: %s target=%s\n', dataPath, clusterDataPath, taskModeArg, targetCol);
    if taskMode && ~strcmp(ud.classTypeDD.Value, 'Multiclass (all values)')
        fprintf(fid, 'Binary classes: %s vs %s\n', char(ud.class1DD.Value), char(ud.class2DD.Value));
    elseif taskMode
        fprintf(fid, 'Multiclass: all values in target column\n');
    end
    if useHpo
        fprintf(fid, 'HPO: n_trials=%d max_epochs/trial=%d out_dir=%s (resume via sqlite)\n', ...
            ud.hpoTrialsEdit.Value, ud.hpoEpochsEdit.Value, outDirHpo);
    else
        fprintf(fid, 'Batch: %d Epochs: %d LR: %s Weight decay: %s\n', batchSize, epochs, lr, weightDecay);
        fprintf(fid, 'Model: hidden=%d heads=%d layers=%d ff=%d dropout=%.2f\n', hiddenDim, numHeads, numLayers, ffDim, dropout);
        fprintf(fid, 'Scheduler: %s warmup=%d\n', scheduler, warmup);
    end
    fprintf(fid, 'Tokenization: region_pool=%d time_pool=%d\n', rp, tp);
    fprintf(fid, '\nLocal: %s\nCluster: sbatch %s\n', cmd, slurmPath);
    fclose(fid);
    
    ud.statusLabel.Text = sprintf('Saved: %s', slurmPath);
    showAlert(fig, sprintf('Scripts saved to:\n%s\n\nSubmit: sbatch %s', outDir, slurmPath), 'Done');
end

%% Legacy layout (R2016a–R2019a: no uigridlayout, uses SizeChangedFcn for resize)
function createUIFigureLegacy()
    MAR = 20; W = 900; H = 800;
    try
        ss = get(0, 'ScreenSize');
        H = min(900, max(700, ss(4) - 100));
        W = max(W, min(1100, ss(3) - 80));
    catch
    end
    fig = uifigure('Name', 'PRISMT - Training Setup', 'Position', [max(20,(get(0,'ScreenSize',3)-W)/2) 20 W H], ...
        'Color', [0.97 0.97 0.98], 'Resize', 'on');
    fig.SizeChangedFcn = @(src,~) layoutResize(src, MAR);

    % Panels with initial positions (layoutResize will update)
    headerPanel = uipanel(fig, 'Title', '', 'BackgroundColor', [0.11 0.33 0.52], 'BorderType', 'none');
    p1 = uipanel(fig, 'Title', '1. Load Dataset', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p2 = uipanel(fig, 'Title', '2. Input & Tokenization', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p3 = uipanel(fig, 'Title', '3. Comparison Conditions', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    modePanel = uipanel(fig, 'Title', '', 'BackgroundColor', [0.98 0.98 1], 'BorderType', 'none');
    p4 = uipanel(fig, 'Title', '4. Training', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p5 = uipanel(fig, 'Title', '5. Model', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    p6 = uipanel(fig, 'Title', '6. Cluster (SLURM)', 'BackgroundColor', [1 1 1], 'FontWeight', 'bold');
    actionPanel = uipanel(fig, 'Title', 'Run', 'BackgroundColor', [0.96 0.97 1], 'FontWeight', 'bold');
    statusPanel = uipanel(fig, 'Title', '', 'BackgroundColor', [0.94 0.94 0.96], 'BorderType', 'none');

    % Store panel handles for resize
    fig.UserData.layoutPanels = struct('header', headerPanel, 'p1', p1, 'p2', p2, 'p3', p3, ...
        'mode', modePanel, 'p4', p4, 'p5', p5, 'p6', p6, 'action', actionPanel, 'status', statusPanel);

    % Panel 1
    uilabel(p1, 'Text', 'Dataset (.mat):', 'Position', [15 78 100 22]);
    pathField = uieditfield(p1, 'text', 'Position', [120 75 400 28], 'Value', '');
    summaryLabel = uilabel(p1, 'Text', 'No data loaded. Click Browse and Load.', 'Position', [15 25 500 45], 'FontColor', [0.45 0.45 0.45]);
    uibutton(p1, 'Text', 'Browse', 'Position', [530 73 55 32], 'ButtonPushedFcn', @(src,~) browseForFile(pathField));
    uibutton(p1, 'Text', 'Load', 'Position', [595 73 55 32], 'ButtonPushedFcn', @(src,~) validateDataset(pathField, summaryLabel));

    % Panel 2
    uilabel(p2, 'Text', 'Data type:', 'Position', [15 98 80 22]);
    dataTypeDD = uidropdown(p2, 'Position', [100 95 100 28], 'Items', {'dff (ΔF/F)', 'zscore'}, 'Value', 'dff (ΔF/F)');
    uilabel(p2, 'Text', 'Normalization:', 'Position', [220 98 90 22]);
    normDD = uidropdown(p2, 'Position', [315 95 200 28], 'Items', {'Scale ×20 (recommended)', 'Robust (median/IQR)', 'Percentile clip', 'None'}, 'Value', 'Scale ×20 (recommended)');
    uilabel(p2, 'Text', 'Region pool:', 'Position', [15 58 80 22]);
    regionPoolDD = uidropdown(p2, 'Position', [100 55 80 28], 'Items', {'None (1)', '2 (avg pairs)', '4', '8'}, 'Value', 'None (1)');
    uilabel(p2, 'Text', 'Time pool:', 'Position', [195 58 70 22]);
    timePoolDD = uidropdown(p2, 'Position', [270 55 80 28], 'Items', {'None (1)', '2 (avg pairs)', '4', '8'}, 'Value', 'None (1)');
    uilabel(p2, 'Text', 'Tokenization:', 'Position', [15 18 90 22]);
    tokenInfoLabel = uilabel(p2, 'Text', 'Load data to see tokenization.', 'Position', [110 15 450 28], 'FontColor', [0.45 0.45 0.45]);

    % Panel 3
    uilabel(p3, 'Text', 'Mode:', 'Position', [15 128 45 22]);
    taskModeDD = uidropdown(p3, 'Position', [65 125 100 28], 'Items', {'Classification', 'Regression'}, 'Value', 'Classification');
    uilabel(p3, 'Text', 'Target column:', 'Position', [180 128 75 22]);
    targetColDD = uidropdown(p3, 'Position', [260 125 95 28], 'Items', {'phase', 'mouse', 'stim', 'response'}, 'Value', 'phase');
    classTypeLabel = uilabel(p3, 'Text', 'Class type:', 'Position', [15 88 65 22]);
    classTypeDD = uidropdown(p3, 'Position', [85 85 130 28], 'Items', {'Multiclass (all values)', 'Binary (select 2)'}, 'Value', 'Binary (select 2)');
    class1Label = uilabel(p3, 'Text', 'Class 1:', 'Position', [230 88 55 22]);
    class1DD = uidropdown(p3, 'Position', [290 85 85 28], 'Items', {'early', 'mid', 'late'}, 'Value', 'early');
    class2Label = uilabel(p3, 'Text', 'vs Class 2:', 'Position', [385 88 65 22]);
    class2DD = uidropdown(p3, 'Position', [455 85 85 28], 'Items', {'early', 'mid', 'late'}, 'Value', 'late');
    uilabel(p3, 'Text', 'Filter stim:', 'Position', [15 48 70 22]);
    stimEdit = uieditfield(p3, 'text', 'Position', [90 45 55 28], 'Value', '1');
    uilabel(p3, 'Text', 'Filter response:', 'Position', [155 48 85 22]);
    responseEdit = uieditfield(p3, 'text', 'Position', [245 45 65 28], 'Value', '0, 1');
    phaseFilterLabel = uilabel(p3, 'Text', 'Filter phase:', 'Position', [325 48 75 22]);
    phaseFilterEdit = uieditfield(p3, 'text', 'Position', [405 45 80 28], 'Value', 'early, mid, late');
    uilabel(p3, 'Text', 'Seed:', 'Position', [500 48 40 22]);
    seedEdit = uieditfield(p3, 'numeric', 'Position', [545 45 55 28], 'Value', 42);

    % Mode row
    uilabel(modePanel, 'Text', 'Mode:', 'Position', [15 5 50 22]);
    modeDD = uidropdown(modePanel, 'Position', [70 2 180 28], 'Items', {'Standard training', 'HPO (Optuna)'}, 'Value', 'Standard training');
    hpoTrialsLabel = uilabel(modePanel, 'Text', 'HPO trials:', 'Position', [265 5 75 22]);
    hpoTrialsEdit = uieditfield(modePanel, 'numeric', 'Position', [345 2 55 28], 'Value', 30);
    hpoEpochsLabel = uilabel(modePanel, 'Text', 'Epochs/trial:', 'Position', [415 5 85 22]);
    hpoEpochsEdit = uieditfield(modePanel, 'numeric', 'Position', [505 2 55 28], 'Value', 30);

    % Header
    uilabel(headerPanel, 'Text', 'PRISMT', 'Position', [20 8 140 24], 'FontSize', 16, 'FontWeight', 'bold', 'FontColor', [1 1 1]);
    uilabel(headerPanel, 'Text', 'Transformer training | Optuna HPO', 'Position', [165 10 220 20], 'FontSize', 10, 'FontColor', [0.9 0.93 1]);
    uibutton(headerPanel, 'Text', 'Help', 'Position', [W-120 6 85 28], 'BackgroundColor', [0.25 0.5 0.75], 'FontColor', [1 1 1], 'ButtonPushedFcn', @(src,~) openHelp());

    % Panels 4, 5, 6, action (same structure as before)
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
    saveDirEdit = uieditfield(p4, 'text', 'Position', [188 39 220 28], 'Value', 'results');
    uilabel(p4, 'Text', 'Output dir:', 'Position', [15 8 70 22]);
    outputDirEdit = uieditfield(p4, 'text', 'Position', [90 5 318 28], 'Value', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'generated_scripts'));

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
    schedulerDD = uidropdown(p5, 'Position', [82 5 110 28], 'Items', {'cosine_warmup', 'cosine', 'reduce_on_plateau', 'step'}, 'Value', 'cosine_warmup');
    uilabel(p5, 'Text', 'Warmup:', 'Position', [197 8 50 22]);
    warmupEdit = uieditfield(p5, 'numeric', 'Position', [252 5 28 28], 'Value', 5);

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
    clusterDataEdit = uieditfield(p6, 'text', 'Position', [140 39 270 28], 'Value', '');
    clusterOutLabel = uilabel(p6, 'Text', 'HPO out dir (cluster):', 'Position', [15 12 115 22]);
    clusterOutEdit = uieditfield(p6, 'text', 'Position', [135 9 275 28], 'Value', '');
    uilabel(p6, 'Text', 'Setup (conda activate, etc.):', 'Position', [15 2 155 22]);
    setupEdit = uieditfield(p6, 'text', 'Position', [170 2 240 22], 'Value', '');
    runBtn = uibutton(actionPanel, 'Text', 'Run Training Now', 'Position', [15 65 250 40], 'BackgroundColor', [0.25 0.55 0.35], ...
        'FontColor', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold', 'ButtonPushedFcn', @(src,~) runTraining(fig));
    genBtn = uibutton(actionPanel, 'Text', 'Generate Cluster Script', 'Position', [15 15 250 38], 'BackgroundColor', [0.28 0.52 0.8], ...
        'FontColor', [1 1 1], 'FontWeight', 'bold', 'ButtonPushedFcn', @(src,~) generateScript(fig));
    statusLabel = uilabel(statusPanel, 'Text', 'Ready. Load a dataset to begin.', 'Position', [10 4 600 20], 'FontColor', [0.45 0.45 0.45], 'FontSize', 10);

    lp = fig.UserData.layoutPanels;
    fig.UserData = struct('pathField', pathField, 'summaryLabel', summaryLabel, 'layoutPanels', lp, ...
        'dataTypeDD', dataTypeDD, 'normDD', normDD, 'tokenInfoLabel', tokenInfoLabel, 'regionPoolDD', regionPoolDD, 'timePoolDD', timePoolDD, ...
        'taskModeDD', taskModeDD, 'targetColDD', targetColDD, 'classTypeDD', classTypeDD, 'class1Label', class1Label, 'class1DD', class1DD, ...
        'class2Label', class2Label, 'class2DD', class2DD, 'phaseFilterLabel', phaseFilterLabel, 'phaseFilterEdit', phaseFilterEdit, ...
        'stimEdit', stimEdit, 'responseEdit', responseEdit, 'seedEdit', seedEdit, 'batchEdit', batchEdit, 'epochsEdit', epochsEdit, ...
        'lrEdit', lrEdit, 'weightDecayEdit', weightDecayEdit, 'valSplitEdit', valSplitEdit, 'saveDirEdit', saveDirEdit, ...
        'hiddenEdit', hiddenEdit, 'numHeadsEdit', numHeadsEdit, 'numLayersEdit', numLayersEdit, 'ffDimEdit', ffDimEdit, ...
        'dropoutEdit', dropoutEdit, 'schedulerDD', schedulerDD, 'warmupEdit', warmupEdit, 'outputDirEdit', outputDirEdit, ...
        'modeDD', modeDD, 'hpoTrialsLabel', hpoTrialsLabel, 'hpoTrialsEdit', hpoTrialsEdit, 'hpoEpochsLabel', hpoEpochsLabel, 'hpoEpochsEdit', hpoEpochsEdit, ...
        'clusterOutLabel', clusterOutLabel, 'clusterOutEdit', clusterOutEdit, 'partitionEdit', partitionEdit, 'memEdit', memEdit, ...
        'gpusEdit', gpusEdit, 'cpusEdit', cpusEdit, 'timeEdit', timeEdit, 'clusterDataEdit', clusterDataEdit, 'setupEdit', setupEdit, ...
        'statusLabel', statusLabel, 'runBtn', runBtn, 'genBtn', genBtn, 'dataInfo', struct());

    modeDD.ValueChangedFcn = @(src,~) updateModeDependentUI(fig);
    updateModeDependentUI(fig);
    pathField.ValueChangedFcn = @(src,~) setStatus(fig.UserData.statusLabel, iif(exist(src.Value, 'file')==2, 'Path OK. Click Load to validate.', 'Enter dataset path.'));
    regionPoolDD.ValueChangedFcn = @(src,~) refreshTokenInfo(fig);
    timePoolDD.ValueChangedFcn = @(src,~) refreshTokenInfo(fig);
    targetColDD.ValueChangedFcn = @(src,~) updateTargetDependentUI(fig);
    classTypeDD.ValueChangedFcn = @(src,~) updateTargetDependentUI(fig);
    layoutResize(fig, MAR);
end

function layoutResize(fig, MAR)
    if ~isvalid(fig) || ~isfield(fig.UserData, 'layoutPanels'), return; end
    lp = fig.UserData.layoutPanels;
    if ~isfield(lp, 'header'), return; end
    pos = fig.Position;
    W = pos(3); H = pos(4);
    if W < 600 || H < 500, return; end
    rightColW = min(300, floor(W*0.35));
    leftW = W - 2*MAR - 15 - rightColW;
    hdr = 42; p1h = 115; p2h = 130; p3h = 170; modeH = 40; p4h = 115; p6h = 105; statusH = 28;
    y = H - hdr;
    lp.header.Position = [0 y W hdr];
    y = y - 12 - p1h;
    lp.p1.Position = [MAR y leftW+15+rightColW p1h];
    y = y - 10 - p2h;
    lp.p2.Position = [MAR y leftW+15+rightColW p2h];
    y = y - 10 - p3h;
    lp.p3.Position = [MAR y leftW+15+rightColW p3h];
    y = y - 10 - modeH;
    lp.mode.Position = [MAR y leftW+15+rightColW modeH];
    y = y - 10 - p4h;
    lp.p4.Position = [MAR y leftW p4h];
    lp.p5.Position = [MAR+leftW+15 y rightColW p4h];
    y = y - 10 - p6h;
    lp.p6.Position = [MAR y leftW p6h];
    lp.action.Position = [MAR+leftW+15 y rightColW p6h];
    lp.status.Position = [MAR 0 W-2*MAR statusH];
    % Update Help button in header to stay right-aligned
    hdrKids = lp.header.Children;
    for k = 1:numel(hdrKids)
        if isa(hdrKids(k), 'matlab.ui.control.Button')
            hdrKids(k).Position(1) = lp.header.Position(3) - 95;
            break;
        end
    end
    % Resize path field and summary in p1
    if isfield(fig.UserData, 'pathField') && isvalid(fig.UserData.pathField)
        pw = lp.p1.Position(3);
        fig.UserData.pathField.Position(3) = max(200, pw - 155);
        if isfield(fig.UserData, 'summaryLabel') && isvalid(fig.UserData.summaryLabel)
            fig.UserData.summaryLabel.Position(3) = max(150, pw - 30);
        end
    end
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
