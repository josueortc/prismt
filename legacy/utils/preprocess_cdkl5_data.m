% Script to preprocess CDKL5 MATLAB struct data into a format compatible with WidefieldDataset
% Usage: matlab -batch "preprocess_cdkl5_data('input.mat', 'output.mat')"
%
% This script:
% 1. Loads cdkl5_m_wt_struct and cdkl5_m_mut_struct
% 2. Extracts neural data and metadata from each animal
% 3. Creates a table T similar to the widefield format with columns: dff, zscore, stim, response, phase, mouse
% 4. Labels: 0 = wild type, 1 = mutant

function preprocess_cdkl5_data(input_file, output_file)
    fprintf('Loading MATLAB file: %s\n', input_file);
    
    % Load the data
    data = load(input_file);
    
    % Check for required structs
    if ~isfield(data, 'cdkl5_m_wt_struct') && ~isfield(data, 'cdkl5_m_mut_struct')
        error('cdkl5_m_wt_struct or cdkl5_m_mut_struct not found in file');
    end
    
    % Initialize cell arrays to store data
    all_dff = {};
    all_zscore = {};
    all_stim = {};
    all_response = {};
    all_phase = {};
    all_mouse = {};
    all_genotype = {};  % Store genotype label
    
    % Process wild type animals
    if isfield(data, 'cdkl5_m_wt_struct')
        wt_struct = data.cdkl5_m_wt_struct;
        % Handle both struct arrays and tables
        if istable(wt_struct)
            n_wt_animals = height(wt_struct);
        else
            n_wt_animals = length(wt_struct);
        end
        fprintf('Processing %d wild type animals...\n', n_wt_animals);
        
        for i = 1:n_wt_animals
            % Handle both struct arrays and tables
            if istable(wt_struct)
                animal = wt_struct(i, :);
                field_names = wt_struct.Properties.VariableNames;
            else
                animal = wt_struct(i);
                field_names = fieldnames(animal);
            end
            
            % Look for neural data fields
            % Priority: allen_parcels (brain_areas x timepoints), then dff, data, calcium
            allen_parcels_data = [];
            dff_data = [];
            zscore_data = [];
            stim_data = [];
            response_data = [];
            
            % Try to find allen_parcels first (this is the primary data source)
            % Handle both struct and table access
            if istable(wt_struct)
                % Table access: use column names
                if ismember('allen_parcels', field_names)
                    allen_parcels_data = animal.allen_parcels{1};
                    fprintf('  Animal %d: Found allen_parcels, shape: %s\n', i, mat2str(size(allen_parcels_data)));
                elseif ismember('dff', field_names)
                    dff_data = animal.dff{1};
                elseif ismember('data', field_names)
                    dff_data = animal.data{1};
                elseif ismember('calcium', field_names)
                    dff_data = animal.calcium{1};
                end
            else
                % Struct access: use fieldnames
                if isfield(animal, 'allen_parcels')
                    allen_parcels_data = animal.allen_parcels;
                    % Replace NaN values with 0
                    allen_parcels_data(isnan(allen_parcels_data)) = 0;
                    fprintf('  Animal %d: Found allen_parcels, shape: %s\n', i, mat2str(size(allen_parcels_data)));
                elseif isfield(animal, 'dff')
                    dff_data = animal.dff;
                elseif isfield(animal, 'data')
                    dff_data = animal.data;
                elseif isfield(animal, 'calcium')
                    dff_data = animal.calcium;
                end
            end
            
            % If we have allen_parcels, use it as dff_data
            if ~isempty(allen_parcels_data)
                % Replace NaN values with 0
                allen_parcels_data(isnan(allen_parcels_data)) = 0;
                dff_data = allen_parcels_data;
            end
            
            % Handle zscore, stim, response, and mouse_id for both struct and table
            if istable(wt_struct)
                % Table access
                if ismember('zscore', field_names)
                    zscore_data = animal.zscore{1};
                elseif ~isempty(dff_data)
                    zscore_data = dff_data;
                end
                
                if ismember('stim', field_names)
                    stim_data = animal.stim{1};
                elseif ismember('stimulus', field_names)
                    stim_data = animal.stimulus{1};
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        stim_data = ones(n_trials, 1);
                    end
                end
                
                if ismember('response', field_names)
                    response_data = animal.response{1};
                elseif ismember('choice', field_names)
                    response_data = animal.choice{1};
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        response_data = ones(n_trials, 1);
                    end
                end
                
                if ismember('mouse', field_names)
                    mouse_id = animal.mouse{1};
                elseif ismember('mouse_id', field_names)
                    mouse_id = animal.mouse_id{1};
                elseif ismember('animal_id', field_names)
                    mouse_id = animal.animal_id{1};
                else
                    mouse_id = sprintf('wt_%03d', i);
                end
            else
                % Struct access
                if isfield(animal, 'zscore')
                    zscore_data = animal.zscore;
                elseif ~isempty(dff_data)
                    zscore_data = dff_data;
                end
                
                if isfield(animal, 'stim')
                    stim_data = animal.stim;
                elseif isfield(animal, 'stimulus')
                    stim_data = animal.stimulus;
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        stim_data = ones(n_trials, 1);
                    end
                end
                
                if isfield(animal, 'response')
                    response_data = animal.response;
                elseif isfield(animal, 'choice')
                    response_data = animal.choice;
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        response_data = ones(n_trials, 1);
                    end
                end
                
                if isfield(animal, 'mouse')
                    mouse_id = animal.mouse;
                elseif isfield(animal, 'mouse_id')
                    mouse_id = animal.mouse_id;
                elseif isfield(animal, 'animal_id')
                    mouse_id = animal.animal_id;
                else
                    mouse_id = sprintf('wt_%03d', i);
                end
            end
            
            % Convert mouse_id to string if needed
            if isnumeric(mouse_id)
                mouse_id = num2str(mouse_id);
            elseif ischar(mouse_id)
                mouse_id = char(mouse_id);
            else
                mouse_id = char(string(mouse_id));
            end
            
            % Only add if we have neural data
            if ~isempty(dff_data)
                % Handle allen_parcels format: (brain_areas, timepoints)
                % Need to split into trials of 30 timepoints each
                % Then reshape to (trials, timepoints, brain_areas)
                
                original_shape = size(dff_data);
                fprintf('  Animal %d: Processing neural data, original shape: %s\n', i, mat2str(original_shape));
                
                if ndims(dff_data) == 2
                    % 2D array: (brain_areas, timepoints) or (timepoints, brain_areas)
                    [d1, d2] = size(dff_data);
                    
                    % Determine orientation based on typical values
                    % For CDKL5: brain_areas should be ~56, timepoints could be hundreds/thousands
                    % Rule: If smaller dim is ~50-100, it's likely brain_areas
                    %       If larger dim is >> smaller dim, larger is likely timepoints
                    
                    % Check if smaller dimension is in brain_areas range (50-100)
                    min_dim = min(d1, d2);
                    max_dim = max(d1, d2);
                    
                    if min_dim >= 50 && min_dim <= 100 && max_dim > min_dim * 5
                        % Smaller dim is likely brain_areas, larger is timepoints
                        if d1 == min_dim
                            % (brain_areas, timepoints) - transpose to (timepoints, brain_areas)
                            dff_data = dff_data';
                            fprintf('  Animal %d: Transposed from (brain_areas=%d, timepoints=%d) to (timepoints=%d, brain_areas=%d)\n', ...
                                i, d1, d2, d2, d1);
                            [n_timepoints, n_brain_areas] = size(dff_data);
                        else
                            % (timepoints, brain_areas) - already correct
                            [n_timepoints, n_brain_areas] = size(dff_data);
                            fprintf('  Animal %d: Already correct format (timepoints=%d, brain_areas=%d)\n', ...
                                i, n_timepoints, n_brain_areas);
                        end
                    else
                        % Fallback: assume (brain_areas, timepoints) if first dim is smaller
                        if d1 < d2
                            dff_data = dff_data';
                            fprintf('  Animal %d: Transposed from (brain_areas=%d, timepoints=%d) to (timepoints=%d, brain_areas=%d)\n', ...
                                i, d1, d2, d2, d1);
                            [n_timepoints, n_brain_areas] = size(dff_data);
                        else
                            [n_timepoints, n_brain_areas] = size(dff_data);
                            fprintf('  Animal %d: Using format (timepoints=%d, brain_areas=%d)\n', ...
                                i, n_timepoints, n_brain_areas);
                        end
                    end
                    
                    % Verify brain_areas is reasonable (should be ~56 for CDKL5)
                    if n_brain_areas > 200
                        fprintf('  Animal %d: WARNING - Unexpected brain_areas count: %d (expected ~56). Check data orientation.\n', ...
                            i, n_brain_areas);
                    end
                    
                    % Split into trials of 30 timepoints each
                    trial_length = 30;
                    n_trials = floor(n_timepoints / trial_length);
                    
                    if n_trials == 0
                        fprintf('  Animal %d: WARNING - Not enough timepoints (%d) for even one trial (need %d), skipping\n', ...
                            i, n_timepoints, trial_length);
                        continue;
                    end
                    
                    % Truncate to multiple of trial_length
                    n_timepoints_used = n_trials * trial_length;
                    dff_data = dff_data(1:n_timepoints_used, :);
                    
                    % Replace any remaining NaN values with 0 before reshaping
                    dff_data(isnan(dff_data)) = 0;
                    
                    % Reshape to (n_trials, trial_length, n_brain_areas)
                    dff_data = reshape(dff_data, [n_trials, trial_length, n_brain_areas]);
                    
                    fprintf('  Animal %d: Split into %d trials of %d timepoints each, final shape: %s\n', ...
                        i, n_trials, trial_length, mat2str(size(dff_data)));
                    
                    % Process zscore the same way
                    if ~isempty(zscore_data) && isequal(size(zscore_data), original_shape)
                        if ndims(zscore_data) == 2
                            [z1, z2] = size(zscore_data);
                            if z1 < z2 && z1 > 50 && z1 < 500
                                zscore_data = zscore_data';
                            end
                            zscore_data = zscore_data(1:n_timepoints_used, :);
                            % Replace any remaining NaN values with 0 before reshaping
                            zscore_data(isnan(zscore_data)) = 0;
                            zscore_data = reshape(zscore_data, [n_trials, trial_length, n_brain_areas]);
                        end
                    end
                    
                    % Create stim and response arrays for each trial
                    % Use dummy values (all ones) since we don't have trial-level metadata
                    stim_data = ones(n_trials, 1);
                    response_data = ones(n_trials, 1);
                    
                elseif ndims(dff_data) == 3
                    % 3D array: check if already in (trials, timepoints, brain_areas) format
                    [d1, d2, d3] = size(dff_data);
                    
                    % Replace NaN values with 0 before permuting
                    dff_data(isnan(dff_data)) = 0;
                    if ~isempty(zscore_data)
                        zscore_data(isnan(zscore_data)) = 0;
                    end
                    
                    % Heuristic: if first dimension is large (likely brain areas), transpose
                    if d1 > 50 && d3 < 1000
                        % (brain_areas, timepoints, trials) -> (trials, timepoints, brain_areas)
                        dff_data = permute(dff_data, [3, 2, 1]);
                        if ~isempty(zscore_data)
                            zscore_data = permute(zscore_data, [3, 2, 1]);
                        end
                    elseif d1 < 50 && d3 > 50
                        % (timepoints, brain_areas, trials) -> (trials, timepoints, brain_areas)
                        dff_data = permute(dff_data, [3, 1, 2]);
                        if ~isempty(zscore_data)
                            zscore_data = permute(zscore_data, [3, 1, 2]);
                        end
                    end
                    
                    % If trials are longer than 30 timepoints, we could split them further
                    % But for now, assume they're already in the right format
                    [n_trials, n_timepoints_per_trial, n_brain_areas] = size(dff_data);
                    
                    % Create stim and response arrays
                    stim_data = ones(n_trials, 1);
                    response_data = ones(n_trials, 1);
                else
                    fprintf('  Animal %d: WARNING - Unexpected dff_data dimensions: %s, skipping\n', ...
                        i, mat2str(size(dff_data)));
                    continue;
                end
                
                % Final check: replace any NaN values with 0 before storing
                dff_data(isnan(dff_data)) = 0;
                if ~isempty(zscore_data)
                    zscore_data(isnan(zscore_data)) = 0;
                end
                
                % Final check: replace any NaN values with 0 before storing
                dff_data(isnan(dff_data)) = 0;
                if ~isempty(zscore_data)
                    zscore_data(isnan(zscore_data)) = 0;
                end
                
                all_dff{end+1} = dff_data;
                all_zscore{end+1} = zscore_data;
                all_stim{end+1} = stim_data;
                all_response{end+1} = response_data;
                all_phase{end+1} = 'all';  % Use 'all' phase for CDKL5 data
                all_mouse{end+1} = mouse_id;
                all_genotype{end+1} = 0;  % 0 = wild type
                
                fprintf('  Animal %d: Final dff shape = %s, %d trials, mouse = %s\n', ...
                    i, mat2str(size(dff_data)), size(dff_data, 1), mouse_id);
            else
                fprintf('  Animal %d: No neural data found, skipping\n', i);
            end
        end
    end
    
    % Process mutant animals
    if isfield(data, 'cdkl5_m_mut_struct')
        mut_struct = data.cdkl5_m_mut_struct;
        % Handle both struct arrays and tables
        if istable(mut_struct)
            n_mut_animals = height(mut_struct);
        else
            n_mut_animals = length(mut_struct);
        end
        fprintf('Processing %d mutant animals...\n', n_mut_animals);
        
        for i = 1:n_mut_animals
            % Handle both struct arrays and tables
            if istable(mut_struct)
                animal = mut_struct(i, :);
                field_names = mut_struct.Properties.VariableNames;
            else
                animal = mut_struct(i);
                field_names = fieldnames(animal);
            end
            
            % Look for neural data fields
            % Priority: allen_parcels (brain_areas x timepoints), then dff, data, calcium
            allen_parcels_data = [];
            dff_data = [];
            zscore_data = [];
            stim_data = [];
            response_data = [];
            
            % Try to find allen_parcels first (this is the primary data source)
            % Handle both struct and table access
            if istable(mut_struct)
                % Table access: use column names
                if ismember('allen_parcels', field_names)
                    allen_parcels_data = animal.allen_parcels{1};
                    fprintf('  Animal %d: Found allen_parcels, shape: %s\n', i, mat2str(size(allen_parcels_data)));
                elseif ismember('dff', field_names)
                    dff_data = animal.dff{1};
                elseif ismember('data', field_names)
                    dff_data = animal.data{1};
                elseif ismember('calcium', field_names)
                    dff_data = animal.calcium{1};
                end
            else
                % Struct access: use fieldnames
                if isfield(animal, 'allen_parcels')
                    allen_parcels_data = animal.allen_parcels;
                    % Replace NaN values with 0
                    allen_parcels_data(isnan(allen_parcels_data)) = 0;
                    fprintf('  Animal %d: Found allen_parcels, shape: %s\n', i, mat2str(size(allen_parcels_data)));
                elseif isfield(animal, 'dff')
                    dff_data = animal.dff;
                elseif isfield(animal, 'data')
                    dff_data = animal.data;
                elseif isfield(animal, 'calcium')
                    dff_data = animal.calcium;
                end
            end
            
            % If we have allen_parcels, use it as dff_data
            if ~isempty(allen_parcels_data)
                dff_data = allen_parcels_data;
            end
            
            % Handle zscore, stim, response, and mouse_id for both struct and table
            if istable(mut_struct)
                % Table access
                if ismember('zscore', field_names)
                    zscore_data = animal.zscore{1};
                elseif ~isempty(dff_data)
                    zscore_data = dff_data;
                end
                
                if ismember('stim', field_names)
                    stim_data = animal.stim{1};
                elseif ismember('stimulus', field_names)
                    stim_data = animal.stimulus{1};
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        stim_data = ones(n_trials, 1);
                    end
                end
                
                if ismember('response', field_names)
                    response_data = animal.response{1};
                elseif ismember('choice', field_names)
                    response_data = animal.choice{1};
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        response_data = ones(n_trials, 1);
                    end
                end
                
                if ismember('mouse', field_names)
                    mouse_id = animal.mouse{1};
                elseif ismember('mouse_id', field_names)
                    mouse_id = animal.mouse_id{1};
                elseif ismember('animal_id', field_names)
                    mouse_id = animal.animal_id{1};
                else
                    mouse_id = sprintf('mut_%03d', i);
                end
            else
                % Struct access
                if isfield(animal, 'zscore')
                    zscore_data = animal.zscore;
                elseif ~isempty(dff_data)
                    zscore_data = dff_data;
                end
                
                if isfield(animal, 'stim')
                    stim_data = animal.stim;
                elseif isfield(animal, 'stimulus')
                    stim_data = animal.stimulus;
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        stim_data = ones(n_trials, 1);
                    end
                end
                
                if isfield(animal, 'response')
                    response_data = animal.response;
                elseif isfield(animal, 'choice')
                    response_data = animal.choice;
                else
                    if ~isempty(dff_data)
                        n_trials = size(dff_data, 1);
                        response_data = ones(n_trials, 1);
                    end
                end
                
                if isfield(animal, 'mouse')
                    mouse_id = animal.mouse;
                elseif isfield(animal, 'mouse_id')
                    mouse_id = animal.mouse_id;
                elseif isfield(animal, 'animal_id')
                    mouse_id = animal.animal_id;
                else
                    mouse_id = sprintf('mut_%03d', i);
                end
            end
            
            if isnumeric(mouse_id)
                mouse_id = num2str(mouse_id);
            elseif ischar(mouse_id)
                mouse_id = char(mouse_id);
            else
                mouse_id = char(string(mouse_id));
            end
            
            if ~isempty(dff_data)
                % Handle allen_parcels format: (brain_areas, timepoints)
                % Need to split into trials of 30 timepoints each
                % Then reshape to (trials, timepoints, brain_areas)
                
                original_shape = size(dff_data);
                fprintf('  Animal %d: Processing neural data, original shape: %s\n', i, mat2str(original_shape));
                
                if ndims(dff_data) == 2
                    % 2D array: (brain_areas, timepoints) or (timepoints, brain_areas)
                    [d1, d2] = size(dff_data);
                    
                    % Determine orientation based on typical values
                    % For CDKL5: brain_areas should be ~56, timepoints could be hundreds/thousands
                    % Rule: If smaller dim is ~50-100, it's likely brain_areas
                    %       If larger dim is >> smaller dim, larger is likely timepoints
                    
                    % Check if smaller dimension is in brain_areas range (50-100)
                    min_dim = min(d1, d2);
                    max_dim = max(d1, d2);
                    
                    if min_dim >= 50 && min_dim <= 100 && max_dim > min_dim * 5
                        % Smaller dim is likely brain_areas, larger is timepoints
                        if d1 == min_dim
                            % (brain_areas, timepoints) - transpose to (timepoints, brain_areas)
                            dff_data = dff_data';
                            fprintf('  Animal %d: Transposed from (brain_areas=%d, timepoints=%d) to (timepoints=%d, brain_areas=%d)\n', ...
                                i, d1, d2, d2, d1);
                            [n_timepoints, n_brain_areas] = size(dff_data);
                        else
                            % (timepoints, brain_areas) - already correct
                            [n_timepoints, n_brain_areas] = size(dff_data);
                            fprintf('  Animal %d: Already correct format (timepoints=%d, brain_areas=%d)\n', ...
                                i, n_timepoints, n_brain_areas);
                        end
                    else
                        % Fallback: assume (brain_areas, timepoints) if first dim is smaller
                        if d1 < d2
                            dff_data = dff_data';
                            fprintf('  Animal %d: Transposed from (brain_areas=%d, timepoints=%d) to (timepoints=%d, brain_areas=%d)\n', ...
                                i, d1, d2, d2, d1);
                            [n_timepoints, n_brain_areas] = size(dff_data);
                        else
                            [n_timepoints, n_brain_areas] = size(dff_data);
                            fprintf('  Animal %d: Using format (timepoints=%d, brain_areas=%d)\n', ...
                                i, n_timepoints, n_brain_areas);
                        end
                    end
                    
                    % Verify brain_areas is reasonable (should be ~56 for CDKL5)
                    if n_brain_areas > 200
                        fprintf('  Animal %d: WARNING - Unexpected brain_areas count: %d (expected ~56). Check data orientation.\n', ...
                            i, n_brain_areas);
                    end
                    
                    % Split into trials of 30 timepoints each
                    trial_length = 30;
                    n_trials = floor(n_timepoints / trial_length);
                    
                    if n_trials == 0
                        fprintf('  Animal %d: WARNING - Not enough timepoints (%d) for even one trial (need %d), skipping\n', ...
                            i, n_timepoints, trial_length);
                        continue;
                    end
                    
                    % Truncate to multiple of trial_length
                    n_timepoints_used = n_trials * trial_length;
                    dff_data = dff_data(1:n_timepoints_used, :);
                    
                    % Replace any remaining NaN values with 0 before reshaping
                    dff_data(isnan(dff_data)) = 0;
                    
                    % Reshape to (n_trials, trial_length, n_brain_areas)
                    dff_data = reshape(dff_data, [n_trials, trial_length, n_brain_areas]);
                    
                    fprintf('  Animal %d: Split into %d trials of %d timepoints each, final shape: %s\n', ...
                        i, n_trials, trial_length, mat2str(size(dff_data)));
                    
                    % Process zscore the same way
                    if ~isempty(zscore_data) && isequal(size(zscore_data), original_shape)
                        if ndims(zscore_data) == 2
                            [z1, z2] = size(zscore_data);
                            if z1 < z2 && z1 > 50 && z1 < 500
                                zscore_data = zscore_data';
                            end
                            zscore_data = zscore_data(1:n_timepoints_used, :);
                            % Replace any remaining NaN values with 0 before reshaping
                            zscore_data(isnan(zscore_data)) = 0;
                            zscore_data = reshape(zscore_data, [n_trials, trial_length, n_brain_areas]);
                        end
                    end
                    
                    % Create stim and response arrays for each trial
                    % Use dummy values (all ones) since we don't have trial-level metadata
                    stim_data = ones(n_trials, 1);
                    response_data = ones(n_trials, 1);
                    
                elseif ndims(dff_data) == 3
                    % 3D array: check if already in (trials, timepoints, brain_areas) format
                    [d1, d2, d3] = size(dff_data);
                    
                    % Replace NaN values with 0 before permuting
                    dff_data(isnan(dff_data)) = 0;
                    if ~isempty(zscore_data)
                        zscore_data(isnan(zscore_data)) = 0;
                    end
                    
                    % Heuristic: if first dimension is large (likely brain areas), transpose
                    if d1 > 50 && d3 < 1000
                        % (brain_areas, timepoints, trials) -> (trials, timepoints, brain_areas)
                        dff_data = permute(dff_data, [3, 2, 1]);
                        if ~isempty(zscore_data)
                            zscore_data = permute(zscore_data, [3, 2, 1]);
                        end
                    elseif d1 < 50 && d3 > 50
                        % (timepoints, brain_areas, trials) -> (trials, timepoints, brain_areas)
                        dff_data = permute(dff_data, [3, 1, 2]);
                        if ~isempty(zscore_data)
                            zscore_data = permute(zscore_data, [3, 1, 2]);
                        end
                    end
                    
                    % If trials are longer than 30 timepoints, we could split them further
                    % But for now, assume they're already in the right format
                    [n_trials, n_timepoints_per_trial, n_brain_areas] = size(dff_data);
                    
                    % Create stim and response arrays
                    stim_data = ones(n_trials, 1);
                    response_data = ones(n_trials, 1);
                else
                    fprintf('  Animal %d: WARNING - Unexpected dff_data dimensions: %s, skipping\n', ...
                        i, mat2str(size(dff_data)));
                    continue;
                end
                
                % Final check: replace any NaN values with 0 before storing
                dff_data(isnan(dff_data)) = 0;
                if ~isempty(zscore_data)
                    zscore_data(isnan(zscore_data)) = 0;
                end
                
                % Final check: replace any NaN values with 0 before storing
                dff_data(isnan(dff_data)) = 0;
                if ~isempty(zscore_data)
                    zscore_data(isnan(zscore_data)) = 0;
                end
                
                all_dff{end+1} = dff_data;
                all_zscore{end+1} = zscore_data;
                all_stim{end+1} = stim_data;
                all_response{end+1} = response_data;
                all_phase{end+1} = 'all';
                all_mouse{end+1} = mouse_id;
                all_genotype{end+1} = 1;  % 1 = mutant
                
                fprintf('  Animal %d: Final dff shape = %s, %d trials, mouse = %s\n', ...
                    i, mat2str(size(dff_data)), size(dff_data, 1), mouse_id);
            else
                fprintf('  Animal %d: No neural data found, skipping\n', i);
            end
        end
    end
    
    % Create preprocessed data structure (compatible with Python loader)
    n_datasets = length(all_dff);
    fprintf('\nCreating preprocessed data structure with %d datasets...\n', n_datasets);
    
    if n_datasets == 0
        error('No valid datasets found');
    end
    
    % Create structure similar to preprocess_matlab_table.m format
    processed_data = struct();
    processed_data.n_datasets = n_datasets;
    processed_data.column_names = {'dff', 'zscore', 'stim', 'response', 'phase', 'mouse'};
    
    % Store each dataset
    for i = 1:n_datasets
        dataset_name = sprintf('dataset_%03d', i);
        processed_data.(dataset_name) = struct();
        processed_data.(dataset_name).dff = all_dff{i};
        processed_data.(dataset_name).zscore = all_zscore{i};
        processed_data.(dataset_name).stim = all_stim{i};
        processed_data.(dataset_name).response = all_response{i};
        processed_data.(dataset_name).phase = all_phase{i};
        processed_data.(dataset_name).mouse = all_mouse{i};
        
        % Calculate number of trials
        if ndims(all_dff{i}) >= 2
            processed_data.(dataset_name).n_trials = size(all_dff{i}, 1);
        else
            processed_data.(dataset_name).n_trials = 1;
        end
    end
    
    % Also create table T for compatibility
    T = table(all_dff', all_zscore', all_stim', all_response', all_phase', all_mouse', ...
        'VariableNames', {'dff', 'zscore', 'stim', 'response', 'phase', 'mouse'});
    
    % Save preprocessed data (save both structures)
    fprintf('Saving preprocessed data to: %s\n', output_file);
    save(output_file, 'processed_data', 'T', '-v7.3');
    
    fprintf('Preprocessing complete!\n');
    fprintf('Summary:\n');
    fprintf('  Total datasets: %d\n', n_datasets);
    fprintf('  Wild type: %d\n', sum([all_genotype{:}] == 0));
    fprintf('  Mutant: %d\n', sum([all_genotype{:}] == 1));
end
