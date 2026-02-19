function [ok, info, errMsg] = validate_prismt_mat(pathStr)
%VALIDATE_PRISMT_MAT Validate .mat file for PRISMT training pipeline
%
%   [ok, info, errMsg] = validate_prismt_mat(pathStr)
%
%   Returns:
%     ok     - true if valid, false otherwise
%     info   - struct with n_datasets, n_trials, n_timepoints, n_regions
%     errMsg - clear error description if ok is false
%
%   Supported formats:
%     1. processed_data with dataset_001, dataset_002, ... (each with dff/zscore)
%     2. Table T with columns dff, zscore, stim, response, phase, mouse

    ok = false;
    info = struct('n_datasets', 0, 'n_trials', 0, 'n_timepoints', 0, 'n_regions', 0, ...
        'phases', {{}}, 'stim_values', [], 'response_values', []);
    errMsg = '';

    % 1. Path checks
    if isempty(pathStr)
        errMsg = ['No file selected.' newline newline ...
            'Click Browse to select a .mat file.'];
        return;
    end

    if exist(pathStr, 'file') ~= 2
        errMsg = ['File not found: ' pathStr newline newline ...
            'Check that the path is correct and the file exists.'];
        return;
    end

    [~, ~, ext] = fileparts(pathStr);
    if ~strcmpi(ext, '.mat')
        errMsg = ['File must be a .mat file: ' pathStr newline newline ...
            'PRISMT requires MATLAB .mat format (standardized structure).'];
        return;
    end

    % 2. Load file
    try
        data = load(pathStr);
    catch ME
        errMsg = ['Could not load file. MATLAB error:' newline ME.message newline newline ...
            'The file may be corrupted, in an unsupported format (e.g. v7.3 HDF5), or not a valid .mat file.'];
        return;
    end

    % 3. Check for required top-level structure
    if ~isfield(data, 'processed_data') && ~isfield(data, 'T')
        fn = fieldnames(data);
        fnStr = strjoin(fn(1:min(10, numel(fn))), ', ');
        if numel(fn) > 10
            fnStr = [fnStr ' ...'];
        end
        errMsg = ['Invalid format: file must contain either processed_data or table T.' newline newline ...
            'Found variables: ' fnStr newline newline ...
            'Expected: processed_data (with dataset_001, dataset_002, ...) OR table T (with dff, zscore, stim, response, phase, mouse columns).' newline newline ...
            'Use standardize_data(input.mat, output.mat, ''widefield'') or ''cdkl5'' to create the correct format.'];
        return;
    end

    % 4. Validate processed_data format
    if isfield(data, 'processed_data')
        pd = data.processed_data;

        if ~isstruct(pd)
            errMsg = ['processed_data must be a struct.' newline newline ...
                'Received: ' class(pd) '.'];
            return;
        end

        fn = fieldnames(pd);
        dsFn = fn(~cellfun(@isempty, strfind(fn, 'dataset_')));

        if isempty(dsFn)
            errMsg = ['processed_data contains no dataset_XXX fields.' newline newline ...
                'Expected: processed_data.dataset_001, dataset_002, ...' newline newline ...
                'Use standardize_data() to create the correct structure.'];
            return;
        end

        nDs = length(dsFn);
        ds1Name = sprintf('dataset_%03d', 1);

        if ~isfield(pd, ds1Name)
            ds1Name = dsFn{1};
        end

        ds1 = pd.(ds1Name);

        if ~isfield(ds1, 'dff')
            errMsg = ['Dataset is missing ''dff'' field.' newline newline ...
                'Each dataset_XXX must have: dff (required), zscore (optional), stim, response, phase, mouse.' newline newline ...
                'dff must be 3D: (trials × timepoints × regions).'];
            return;
        end

        dff = ds1.dff;
        if ~isnumeric(dff)
            errMsg = ['dff must be a numeric array.' newline newline ...
                'Received: ' class(dff)];
            return;
        end

        if ndims(dff) ~= 3
            errMsg = ['dff must be 3-dimensional (trials × timepoints × regions).' newline newline ...
                'Received shape: [' num2str(size(dff)) ']' newline newline ...
                'Expected: (N_trials, N_timepoints, N_regions) where regions = brain areas, ROIs, or features.'];
            return;
        end

        n1 = size(dff, 1);
        n2 = size(dff, 2);
        n3 = size(dff, 3);

        if n1 == 0 || n2 == 0 || n3 == 0
            errMsg = ['dff has zero dimension. Data appears empty.' newline newline ...
                'Shape: [' num2str([n1 n2 n3]) ']'];
            return;
        end

        if any(isnan(dff(:))) && all(isnan(dff(:)))
            errMsg = 'dff contains only NaN values. No valid neural data.';
            return;
        end

        % Infer layout: typically (trials, timepoints, regions)
        if n2 > n3 && n3 > 1 && n2 > 20
            info.n_trials = n1;
            info.n_timepoints = n3;
            info.n_regions = n2;
        else
            info.n_trials = n1;
            info.n_timepoints = n2;
            info.n_regions = n3;
        end

        info.n_datasets = nDs;

        info.phases = extract_unique_values(pd, 'phase');
        info.stim_values = extract_unique_numeric(pd, 'stim');
        info.response_values = extract_unique_numeric(pd, 'response');
        ok = true;
        return;
    end

    % 5. Validate table T format
    T = data.T;

    if ~istable(T)
        errMsg = ['T must be a MATLAB table.' newline newline ...
            'Received: ' class(T) newline newline ...
            'Use standardize_data() or ensure your .mat contains a table with columns: dff, zscore, stim, response, phase, mouse.'];
        return;
    end

    reqCols = {'dff', 'stim', 'response', 'phase', 'mouse'};
    haveCols = T.Properties.VariableNames;
    missingCols = setdiff(reqCols, haveCols);

    if ~isempty(missingCols)
        errMsg = ['Table T is missing required columns: ' strjoin(missingCols, ', ') newline newline ...
            'Required: dff, stim, response, phase, mouse' newline ...
            'Found: ' strjoin(haveCols, ', ') newline newline ...
            'Use standardize_data() to create the correct format.'];
        return;
    end

    nDs = height(T);
    if nDs == 0
        errMsg = 'Table T has no rows. No datasets to load.';
        return;
    end

    try
        dffCell = T.dff{1};
    catch
        errMsg = ['Could not read dff from first row of T.' newline newline ...
            'dff column should contain cell arrays with 3D numeric data (trials × timepoints × regions).'];
        return;
    end

    if iscell(dffCell)
        dffCell = dffCell{1};
    end

    if ~isnumeric(dffCell)
        errMsg = ['dff must contain numeric arrays.' newline newline ...
            'First row dff type: ' class(dffCell)];
        return;
    end

    if ndims(dffCell) ~= 3
        errMsg = ['dff in table must be 3D (trials × timepoints × regions).' newline newline ...
            'First row dff shape: [' num2str(size(dffCell)) ']'];
        return;
    end

    n1 = size(dffCell, 1);
    n2 = size(dffCell, 2);
    n3 = size(dffCell, 3);

    if n1 == 0 || n2 == 0 || n3 == 0
        errMsg = ['First dataset dff is empty.' newline 'Shape: [' num2str([n1 n2 n3]) ']'];
        return;
    end

    if n2 > n3 && n3 > 1 && n2 > 20
        info.n_trials = n1;
        info.n_timepoints = n3;
        info.n_regions = n2;
    else
        info.n_trials = n1;
        info.n_timepoints = n2;
        info.n_regions = n3;
    end

    info.n_datasets = nDs;
    info.phases = extract_unique_values_table(T, 'phase');
    info.stim_values = extract_unique_numeric_table(T, 'stim');
    info.response_values = extract_unique_numeric_table(T, 'response');
    ok = true;
end

function vals = extract_unique_values(pd, fieldName)
    vals = {'early', 'mid', 'late'};
    fn = fieldnames(pd);
    dsFn = fn(~cellfun(@isempty, strfind(fn, 'dataset_')));
    seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    out = {};
    for i = 1:min(20, length(dsFn))
        ds = pd.(dsFn{i});
        if ~isfield(ds, fieldName), continue; end
        v = ds.(fieldName);
        if ischar(v)
            s = strtrim(lower(v));
            if ~isempty(s) && ~isKey(seen, s), out{end+1} = s; seen(s) = true; end
        elseif iscell(v)
            for j = 1:numel(v)
                s = strtrim(lower(char(string(v{j}))));
                if ~isempty(s) && length(s) < 30 && ~isKey(seen, s), out{end+1} = s; seen(s) = true; end
            end
        else
            s = strtrim(lower(char(string(v))));
            if ~isempty(s) && ~isKey(seen, s), out{end+1} = s; seen(s) = true; end
        end
    end
    if ~isempty(out), vals = out; end
end

function vals = extract_unique_numeric(pd, fieldName)
    vals = [];
    fn = fieldnames(pd);
    dsFn = fn(~cellfun(@isempty, strfind(fn, 'dataset_')));
    for i = 1:min(20, length(dsFn))
        ds = pd.(dsFn{i});
        if isfield(ds, fieldName)
            v = ds.(fieldName);
            if isnumeric(v), vals = [vals; v(:)]; end
        end
    end
    vals = unique(vals(~isnan(vals)));
    if isempty(vals), vals = 1; end
end

function vals = extract_unique_values_table(T, fieldName)
    vals = {'early', 'mid', 'late'};
    if ~ismember(fieldName, T.Properties.VariableNames), return; end
    try
        col = T.(fieldName);
        seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
        out = {};
        nRows = min(height(T), 50);
        for i = 1:nRows
            if iscell(col)
                v = col{i};
            else
                v = col(i);
            end
            if iscell(v), v = v{1}; end
            s = strtrim(lower(char(string(v))));
            if ~isempty(s) && length(s) < 50 && ~isKey(seen, s)
                out{end+1} = s;
                seen(s) = true;
            end
        end
        if ~isempty(out), vals = out; end
    catch
    end
end

function vals = extract_unique_numeric_table(T, fieldName)
    vals = 1;
    if ~ismember(fieldName, T.Properties.VariableNames), return; end
    try
        col = T.(fieldName);
        out = [];
        for i = 1:min(height(T), 50)
            if iscell(col), v = col{i}; else, v = col(i); end
            if iscell(v), v = v{1}; end
            if isnumeric(v), out = [out; v(:)]; end
        end
        out = unique(out(~isnan(out)));
        if ~isempty(out), vals = out; end
    catch
    end
end
