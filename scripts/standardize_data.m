function standardize_data(input_file, output_file, data_type)
%STANDARDIZE_DATA Standardize widefield or CDKL5 data into unified format
%
%   STANDARDIZE_DATA(input_file, output_file, data_type)
%
%   This function standardizes data from either widefield or CDKL5 experiments
%   into a unified format that can be used by the PRISMT training pipeline.
%
%   Inputs:
%       input_file - Path to input .mat file
%       output_file - Path to output standardized .mat file
%       data_type - 'widefield' or 'cdkl5'
%
%   Output:
%       Creates a standardized .mat file with 'standardized_data' structure:
%       - n_datasets: Number of datasets
%       - dataset_XXX: Each dataset contains:
%           - dff: Neural data (trials, timepoints, brain_areas)
%           - zscore: Z-scored data (trials, timepoints, brain_areas)
%           - stim: Stimulus values (trials, 1)
%           - response: Response values (trials, 1)
%           - phase: Phase string or array
%           - mouse: Mouse ID string
%           - label: Classification label (for CDKL5: 0=WT, 1=Mutant)
%           - dataset_type: 'widefield' or 'cdkl5'
%
%   Example:
%       standardize_data('raw_data.mat', 'standardized.mat', 'widefield')
%       standardize_data('cdkl5_raw.mat', 'standardized.mat', 'cdkl5')

    fprintf('=== PRISMT Data Standardization ===\n');
    fprintf('Input file: %s\n', input_file);
    fprintf('Output file: %s\n', output_file);
    fprintf('Data type: %s\n', data_type);
    fprintf('\n');
    
    % Load input data
    fprintf('Loading input data...\n');
    try
        data = load(input_file);
    catch ME
        error('Failed to load input file %s: %s', input_file, ME.message);
    end
    
    % Process based on data type
    if strcmpi(data_type, 'widefield')
        standardized_data = standardize_widefield_data(data);
    elseif strcmpi(data_type, 'cdkl5')
        standardized_data = standardize_cdkl5_data(data);
    else
        error('Unknown data_type: %s. Must be ''widefield'' or ''cdkl5''', data_type);
    end
    
    % Save standardized data
    fprintf('Saving standardized data to %s...\n', output_file);
    save(output_file, 'standardized_data', '-v7.3');
    
    fprintf('Standardization complete!\n');
    fprintf('Total datasets: %d\n', standardized_data.n_datasets);
end

function std_data = standardize_widefield_data(data)
    % Standardize widefield data format
    
    fprintf('Processing widefield data...\n');
    
    % Check for table T or processed_data
    if isfield(data, 'T')
        T = data.T;
        fprintf('Found table T with %d rows\n', height(T));
        
        % Process each row
        n_datasets = height(T);
        std_data.n_datasets = n_datasets;
        
        for i = 1:n_datasets
            dataset_name = sprintf('dataset_%03d', i);
            
            % Extract data from table row
            row = T(i, :);
            
            % Get dff data
            dff_cell = row.dff{1};
            if ndims(dff_cell) == 3
                [n1, n2, n3] = size(dff_cell);
                % Check if (trials, brain_areas, timepoints) -> transpose to (trials, timepoints, brain_areas)
                if n2 > n3 && n2 > 50
                    dff_cell = permute(dff_cell, [1, 3, 2]);
                end
            end
            
            % Get zscore data (same processing)
            zscore_cell = row.zscore{1};
            if ndims(zscore_cell) == 3
                [n1, n2, n3] = size(zscore_cell);
                if n2 > n3 && n2 > 50
                    zscore_cell = permute(zscore_cell, [1, 3, 2]);
                end
            end
            
            % Extract metadata
            stim_data = row.stim{1};
            response_data = row.response{1};
            phase_data = row.phase{1};
            mouse_data = row.mouse{1};
            
            % Ensure correct shapes
            if size(stim_data, 2) > size(stim_data, 1)
                stim_data = stim_data';
            end
            if size(response_data, 2) > size(response_data, 1)
                response_data = response_data';
            end
            
            % Store standardized dataset
            std_data.(dataset_name).dff = dff_cell;
            std_data.(dataset_name).zscore = zscore_cell;
            std_data.(dataset_name).stim = stim_data;
            std_data.(dataset_name).response = response_data;
            std_data.(dataset_name).phase = phase_data;
            std_data.(dataset_name).mouse = mouse_data;
            std_data.(dataset_name).dataset_type = 'widefield';
            std_data.(dataset_name).label = [];  % Will be set based on phase/task
            
            fprintf('  Dataset %d: dff shape = %s, %d trials\n', i, mat2str(size(dff_cell)), size(dff_cell, 1));
        end
        
    elseif isfield(data, 'processed_data')
        % Already preprocessed, just copy structure
        fprintf('Found processed_data structure\n');
        processed = data.processed_data;
        
        if isstruct(processed) && isfield(processed, 'n_datasets')
            n_datasets = processed.n_datasets;
        else
            % Count datasets
            fields = fieldnames(processed);
            dataset_fields = fields(startsWith(fields, 'dataset_'));
            n_datasets = length(dataset_fields);
        end
        
        std_data.n_datasets = n_datasets;
        
        for i = 1:n_datasets
            dataset_name = sprintf('dataset_%03d', i);
            if isfield(processed, dataset_name)
                std_data.(dataset_name) = processed.(dataset_name);
                if ~isfield(std_data.(dataset_name), 'dataset_type')
                    std_data.(dataset_name).dataset_type = 'widefield';
                end
            end
        end
    else
        error('Input file must contain either table T or processed_data structure');
    end
end

function std_data = standardize_cdkl5_data(data)
    % Standardize CDKL5 data format
    
    fprintf('Processing CDKL5 data...\n');
    
    % Check for CDKL5 structures
    if ~isfield(data, 'cdkl5_m_wt_struct') && ~isfield(data, 'cdkl5_m_mut_struct')
        error('CDKL5 data must contain cdkl5_m_wt_struct and/or cdkl5_m_mut_struct');
    end
    
    all_datasets = {};
    dataset_idx = 1;
    
    % Process wild type animals
    if isfield(data, 'cdkl5_m_wt_struct')
        wt_struct = data.cdkl5_m_wt_struct;
        if istable(wt_struct)
            n_wt = height(wt_struct);
        else
            n_wt = length(wt_struct);
        end
        
        fprintf('Processing %d wild type animals...\n', n_wt);
        
        for i = 1:n_wt
            if istable(wt_struct)
                animal = wt_struct(i, :);
                field_names = wt_struct.Properties.VariableNames;
            else
                animal = wt_struct(i);
                field_names = fieldnames(animal);
            end
            
            % Extract allen_parcels data
            allen_parcels_data = [];
            if istable(wt_struct)
                if ismember('allen_parcels', field_names)
                    allen_parcels_data = animal.allen_parcels{1};
                end
            else
                if isfield(animal, 'allen_parcels')
                    allen_parcels_data = animal.allen_parcels;
                end
            end
            
            if ~isempty(allen_parcels_data)
                % Replace NaN with 0
                allen_parcels_data(isnan(allen_parcels_data)) = 0;
                
                % Determine orientation and transpose to (timepoints, brain_areas)
                [d1, d2] = size(allen_parcels_data);
                min_dim = min(d1, d2);
                max_dim = max(d1, d2);
                
                if min_dim >= 50 && min_dim <= 100 && max_dim > min_dim * 5
                    % Smaller dim is brain_areas
                    if d1 == min_dim
                        allen_parcels_data = allen_parcels_data';
                    end
                elseif d1 < d2
                    allen_parcels_data = allen_parcels_data';
                end
                
                [n_timepoints, n_brain_areas] = size(allen_parcels_data);
                
                % Verify brain_areas is 56
                if n_brain_areas ~= 56
                    fprintf('  Warning: Animal %d has %d brain areas (expected 56)\n', i, n_brain_areas);
                end
                
                % Split into trials of 30 timepoints
                trial_length = 30;
                n_trials = floor(n_timepoints / trial_length);
                
                if n_trials > 0
                    n_timepoints_used = n_trials * trial_length;
                    allen_parcels_data = allen_parcels_data(1:n_timepoints_used, :);
                    dff_data = reshape(allen_parcels_data, [n_trials, trial_length, n_brain_areas]);
                    
                    % Get mouse ID
                    if istable(wt_struct)
                        if ismember('mouse', field_names)
                            mouse_id = animal.mouse{1};
                        else
                            mouse_id = sprintf('wt_%03d', i);
                        end
                    else
                        if isfield(animal, 'mouse')
                            mouse_id = animal.mouse;
                        else
                            mouse_id = sprintf('wt_%03d', i);
                        end
                    end
                    
                    if isnumeric(mouse_id)
                        mouse_id = num2str(mouse_id);
                    end
                    mouse_id = char(string(mouse_id));
                    if ~startsWith(mouse_id, 'wt_')
                        mouse_id = ['wt_', mouse_id];
                    end
                    
                    % Create dataset structure
                    dataset.dff = dff_data;
                    dataset.zscore = dff_data;  % Use same data for zscore
                    dataset.stim = ones(n_trials, 1);
                    dataset.response = ones(n_trials, 1);
                    dataset.phase = 'all';
                    dataset.mouse = mouse_id;
                    dataset.label = 0;  % 0 = Wild Type
                    dataset.dataset_type = 'cdkl5';
                    
                    all_datasets{end+1} = dataset;
                    dataset_idx = dataset_idx + 1;
                end
            end
        end
    end
    
    % Process mutant animals
    if isfield(data, 'cdkl5_m_mut_struct')
        mut_struct = data.cdkl5_m_mut_struct;
        if istable(mut_struct)
            n_mut = height(mut_struct);
        else
            n_mut = length(mut_struct);
        end
        
        fprintf('Processing %d mutant animals...\n', n_mut);
        
        for i = 1:n_mut
            if istable(mut_struct)
                animal = mut_struct(i, :);
                field_names = mut_struct.Properties.VariableNames;
            else
                animal = mut_struct(i);
                field_names = fieldnames(animal);
            end
            
            % Extract allen_parcels data
            allen_parcels_data = [];
            if istable(mut_struct)
                if ismember('allen_parcels', field_names)
                    allen_parcels_data = animal.allen_parcels{1};
                end
            else
                if isfield(animal, 'allen_parcels')
                    allen_parcels_data = animal.allen_parcels;
                end
            end
            
            if ~isempty(allen_parcels_data)
                % Replace NaN with 0
                allen_parcels_data(isnan(allen_parcels_data)) = 0;
                
                % Determine orientation and transpose to (timepoints, brain_areas)
                [d1, d2] = size(allen_parcels_data);
                min_dim = min(d1, d2);
                max_dim = max(d1, d2);
                
                if min_dim >= 50 && min_dim <= 100 && max_dim > min_dim * 5
                    if d1 == min_dim
                        allen_parcels_data = allen_parcels_data';
                    end
                elseif d1 < d2
                    allen_parcels_data = allen_parcels_data';
                end
                
                [n_timepoints, n_brain_areas] = size(allen_parcels_data);
                
                % Verify brain_areas is 56
                if n_brain_areas ~= 56
                    fprintf('  Warning: Animal %d has %d brain areas (expected 56)\n', i, n_brain_areas);
                end
                
                % Split into trials of 30 timepoints
                trial_length = 30;
                n_trials = floor(n_timepoints / trial_length);
                
                if n_trials > 0
                    n_timepoints_used = n_trials * trial_length;
                    allen_parcels_data = allen_parcels_data(1:n_timepoints_used, :);
                    dff_data = reshape(allen_parcels_data, [n_trials, trial_length, n_brain_areas]);
                    
                    % Get mouse ID
                    if istable(mut_struct)
                        if ismember('mouse', field_names)
                            mouse_id = animal.mouse{1};
                        else
                            mouse_id = sprintf('mut_%03d', i);
                        end
                    else
                        if isfield(animal, 'mouse')
                            mouse_id = animal.mouse;
                        else
                            mouse_id = sprintf('mut_%03d', i);
                        end
                    end
                    
                    if isnumeric(mouse_id)
                        mouse_id = num2str(mouse_id);
                    end
                    mouse_id = char(string(mouse_id));
                    if ~startsWith(mouse_id, 'mut_')
                        mouse_id = ['mut_', mouse_id];
                    end
                    
                    % Create dataset structure
                    dataset.dff = dff_data;
                    dataset.zscore = dff_data;
                    dataset.stim = ones(n_trials, 1);
                    dataset.response = ones(n_trials, 1);
                    dataset.phase = 'all';
                    dataset.mouse = mouse_id;
                    dataset.label = 1;  % 1 = Mutant
                    dataset.dataset_type = 'cdkl5';
                    
                    all_datasets{end+1} = dataset;
                    dataset_idx = dataset_idx + 1;
                end
            end
        end
    end
    
    % Create standardized structure
    n_datasets = length(all_datasets);
    std_data.n_datasets = n_datasets;
    
    for i = 1:n_datasets
        dataset_name = sprintf('dataset_%03d', i);
        std_data.(dataset_name) = all_datasets{i};
    end
    
    fprintf('Standardized %d datasets from CDKL5 data\n', n_datasets);
end
