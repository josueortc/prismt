% Script to preprocess MATLAB table into a format easier for Python to read
% Usage: matlab -batch "preprocess_matlab_table('input.mat', 'output.mat')"
%
% This script:
% 1. Loads the MATLAB table T
% 2. Extracts each row (dataset) with columns: dff, zscore, stim, response, phase, mouse
% 3. Saves data in a cleaner structure that Python can easily read

function preprocess_matlab_table(input_file, output_file)
    fprintf('Loading MATLAB file: %s\n', input_file);
    
    % Load the table
    data = load(input_file);
    
    if ~isfield(data, 'T')
        error('Table T not found in file');
    end
    
    T = data.T;
    
    fprintf('Table size: %d rows x %d columns\n', size(T, 1), size(T, 2));
    
    if ~istable(T)
        error('T is not a MATLAB table');
    end
    
    % Get column names
    varNames = T.Properties.VariableNames;
    fprintf('Column names: %s\n', strjoin(varNames, ', '));
    
    % Expected columns: dff, zscore, stim, response, phase, mouse
    % Find column indices
    dff_idx = find(strcmp(varNames, 'dff'), 1);
    zscore_idx = find(strcmp(varNames, 'zscore'), 1);
    stim_idx = find(strcmp(varNames, 'stim'), 1);
    response_idx = find(strcmp(varNames, 'response'), 1);
    phase_idx = find(strcmp(varNames, 'phase'), 1);
    mouse_idx = find(strcmp(varNames, 'mouse'), 1);
    
    if isempty(dff_idx) || isempty(zscore_idx) || isempty(stim_idx) || ...
       isempty(response_idx) || isempty(phase_idx) || isempty(mouse_idx)
        error('Required columns not found. Expected: dff, zscore, stim, response, phase, mouse');
    end
    
    n_rows = size(T, 1);
    fprintf('Processing %d datasets (rows)...\n', n_rows);
    
    % Create output structure
    % Each row will be stored as a separate dataset with consistent structure
    processed_data = struct();
    processed_data.n_datasets = n_rows;
    processed_data.column_names = {'dff', 'zscore', 'stim', 'response', 'phase', 'mouse'};
    
    % Process each row (dataset)
    for row_idx = 1:n_rows
        fprintf('Processing dataset %d/%d...\n', row_idx, n_rows);
        
        % Extract dff data
        dff_cell = T{row_idx, dff_idx};
        if iscell(dff_cell)
            dff_data = dff_cell{1};
        else
            dff_data = dff_cell;
        end
        % Ensure shape: [trials, timepoints, brain_areas]
        % MATLAB shows: [225, 82, 41] = [trials, brain_areas, timepoints]
        % Need: [trials, timepoints, brain_areas]
        if ndims(dff_data) == 3
            [n1, n2, n3] = size(dff_data);
            % Determine orientation based on typical values
            % If first dim is largest, it's likely [trials, brain_areas, timepoints]
            if n1 > n3
                % [trials, brain_areas, timepoints] -> [trials, timepoints, brain_areas]
                dff_data = permute(dff_data, [1, 3, 2]);
            end
            % Otherwise assume it's already [trials, timepoints, brain_areas] or [timepoints, brain_areas, trials]
            if n1 < n3 && n1 == 41
                % [timepoints, brain_areas, trials] -> [trials, timepoints, brain_areas]
                dff_data = permute(dff_data, [3, 1, 2]);
            end
        end
        
        % Determine number of trials from dff_data shape
        % After permutation, dff_data should be [trials, timepoints, brain_areas]
        if ndims(dff_data) == 3
            n_trials = size(dff_data, 1);
        elseif ndims(dff_data) == 2
            % If 2D, assume [trials, features]
            n_trials = size(dff_data, 1);
        else
            error('Unexpected dff_data dimensions: %s. Expected 2D or 3D array.', mat2str(size(dff_data)));
        end
        
        % Extract zscore data (same processing)
        zscore_cell = T{row_idx, zscore_idx};
        if iscell(zscore_cell)
            zscore_data = zscore_cell{1};
        else
            zscore_data = zscore_cell;
        end
        if ndims(zscore_data) == 3
            [n1, n2, n3] = size(zscore_data);
            if n1 > n3
                zscore_data = permute(zscore_data, [1, 3, 2]);
            end
            if n1 < n3 && n1 == 41
                zscore_data = permute(zscore_data, [3, 1, 2]);
            end
        end
        
        % Extract stim data
        stim_cell = T{row_idx, stim_idx};
        if iscell(stim_cell)
            stim_data = stim_cell{1};
        else
            stim_data = stim_cell;
        end
        % Ensure it's a column vector [trials, 1] or [trials]
        if size(stim_data, 2) > 1
            stim_data = stim_data(:);
        end
        stim_data = double(stim_data(:));  % Ensure numeric
        
        % Extract response data
        response_cell = T{row_idx, response_idx};
        if iscell(response_cell)
            response_data = response_cell{1};
        else
            response_data = response_cell;
        end
        if size(response_data, 2) > 1
            response_data = response_data(:);
        end
        response_data = double(response_data(:));  % Ensure numeric
        
        % Extract phase (string) - single string per row, not per trial
        % Phase should be alphanumeric string (e.g., 'early', 'mid', 'late', or IDs like '101', '105')
        phase_cell = T{row_idx, phase_idx};
        if iscell(phase_cell)
            phase_str = phase_cell{1};
        else
            phase_str = phase_cell;
        end
        % Convert to string preserving letters and numbers
        % Handle character arrays (shape [1 N]) - these are strings stored as char arrays
        if ischar(phase_str)
            % Character array - convert to string preserving all characters
            phase_str = deblank(phase_str);  % Remove trailing whitespace/null chars
            phase_str = strtrim(phase_str);  % Remove leading/trailing whitespace
            % Convert char array to string (preserves letters and numbers)
            phase_str = string(phase_str);
        elseif isstring(phase_str)
            % Already a string - just trim
            phase_str = strtrim(phase_str);
        elseif isnumeric(phase_str)
            % If numeric, convert to string (preserves numbers)
            phase_str = string(num2str(phase_str));
        else
            % Try to convert to string
            phase_str = string(char(phase_str));
            phase_str = strtrim(phase_str);
        end
        % Ensure it's a scalar string (not array)
        if ~isscalar(phase_str)
            phase_str = phase_str(1);
        end
        % Convert to char for storage (MATLAB strings are stored as char in .mat files)
        % This preserves the string value (e.g., 'late', 'early', 'mid', '101', '105')
        phase_data = char(phase_str);
        
        % Extract mouse (string) - single string per row, not per trial
        % Mouse should be alphanumeric string (e.g., mouse IDs like '72', '101', '105', '108')
        mouse_cell = T{row_idx, mouse_idx};
        if iscell(mouse_cell)
            mouse_str = mouse_cell{1};
        else
            mouse_str = mouse_cell;
        end
        
        % Debug: log raw mouse value for first few rows
        if row_idx <= 3
            fprintf('  Dataset %d: Raw mouse type: %s, value: %s\n', row_idx, class(mouse_str), mat2str(mouse_str));
        end
        
        % Convert to string preserving letters and numbers
        % Handle character arrays (shape [1 N]) - these are strings stored as char arrays
        if ischar(mouse_str)
            % Character array - convert to string preserving all characters
            mouse_str = deblank(mouse_str);  % Remove trailing whitespace/null chars
            mouse_str = strtrim(mouse_str);  % Remove leading/trailing whitespace
            % Check if empty after trimming
            if isempty(mouse_str)
                fprintf('  WARNING: Dataset %d: Mouse char array is empty after trimming\n', row_idx);
                mouse_str = 'unknown';
            else
                % Convert char array to string (preserves letters and numbers)
                mouse_str = string(mouse_str);
            end
        elseif isstring(mouse_str)
            % Already a string - just trim
            mouse_str = strtrim(mouse_str);
            if isempty(mouse_str) || strlength(mouse_str) == 0
                fprintf('  WARNING: Dataset %d: Mouse string is empty\n', row_idx);
                mouse_str = 'unknown';
            end
        elseif isnumeric(mouse_str)
            % If numeric, convert to string (preserves numbers)
            mouse_str = string(num2str(mouse_str));
            mouse_str = strtrim(mouse_str);
        else
            % Try to convert to string
            try
                mouse_str = string(char(mouse_str));
                mouse_str = strtrim(mouse_str);
                if isempty(mouse_str) || strlength(mouse_str) == 0
                    fprintf('  WARNING: Dataset %d: Mouse converted to empty string\n', row_idx);
                    mouse_str = 'unknown';
                end
            catch
                fprintf('  WARNING: Dataset %d: Failed to convert mouse to string, using unknown\n', row_idx);
                mouse_str = 'unknown';
            end
        end
        % Ensure it's a scalar string (not array)
        if ~isscalar(mouse_str)
            mouse_str = mouse_str(1);
        end
        % Ensure non-empty
        if isempty(mouse_str) || strlength(mouse_str) == 0
            mouse_str = 'unknown';
        end
        % Convert to char for storage (MATLAB strings are stored as char in .mat files)
        % This preserves the string value (e.g., 'HB059', '72', '101', '105', '108')
        % Use the same format as phase: convert to char array
        mouse_data = char(mouse_str);
        
        % Store in structure
        dataset_name = sprintf('dataset_%03d', row_idx);
        processed_data.(dataset_name) = struct();
        processed_data.(dataset_name).dff = dff_data;
        processed_data.(dataset_name).zscore = zscore_data;
        processed_data.(dataset_name).stim = stim_data;
        processed_data.(dataset_name).response = response_data;
        processed_data.(dataset_name).phase = phase_data;  % Single char array per row (same format as mouse)
        processed_data.(dataset_name).mouse = mouse_data;  % Single char array per row (same format as phase)
        processed_data.(dataset_name).n_trials = n_trials;
        
        fprintf('  Dataset %d: %d trials, dff shape: %s, stim values: %s, response values: %s, phase: %s, mouse: %s\n', ...
            row_idx, n_trials, mat2str(size(dff_data)), ...
            mat2str(unique(stim_data)'), mat2str(unique(response_data)'), ...
            phase_data, mouse_data);
    end
    
    % Save processed data
    fprintf('\nSaving processed data to: %s\n', output_file);
    save(output_file, 'processed_data', '-v7.3');
    
    fprintf('Preprocessing complete!\n');
    fprintf('Output file: %s\n', output_file);
    fprintf('Structure: processed_data with %d datasets\n', n_rows);
end

