% Script to inspect MATLAB table structure
% Usage: matlab -batch "inspect_matlab_table('path/to/tableForModeling.mat')"

function inspect_matlab_table(mat_file_path)
    fprintf('Loading MATLAB file: %s\n', mat_file_path);
    
    % Load the table
    data = load(mat_file_path);
    
    if ~isfield(data, 'T')
        error('Table T not found in file');
    end
    
    T = data.T;
    
    fprintf('\n=== Table Structure ===\n');
    fprintf('Table class: %s\n', class(T));
    fprintf('Table size: %d rows x %d columns\n', size(T, 1), size(T, 2));
    
    if istable(T)
        fprintf('\n=== Column Names ===\n');
        varNames = T.Properties.VariableNames;
        for i = 1:length(varNames)
            fprintf('Column %d: %s\n', i, varNames{i});
        end
        
        fprintf('\n=== Column Data Types ===\n');
        for i = 1:length(varNames)
            col_data = T.(varNames{i});
            if iscell(col_data)
                % Check first element
                if ~isempty(col_data) && size(col_data{1}, 1) > 0
                    fprintf('Column %d (%s): cell array, first element shape: %s\n', ...
                        i, varNames{i}, mat2str(size(col_data{1})));
                else
                    fprintf('Column %d (%s): cell array (empty or scalar)\n', i, varNames{i});
                end
            else
                fprintf('Column %d (%s): %s, shape: %s\n', ...
                    i, varNames{i}, class(col_data), mat2str(size(col_data)));
            end
        end
        
        fprintf('\n=== Sample Values (first row) ===\n');
        for i = 1:length(varNames)
            col_data = T.(varNames{i});
            if iscell(col_data) && ~isempty(col_data)
                first_val = col_data{1};
                if isnumeric(first_val) && numel(first_val) <= 20
                    fprintf('Column %d (%s): %s\n', i, varNames{i}, mat2str(first_val));
                elseif isnumeric(first_val)
                    fprintf('Column %d (%s): numeric array, shape: %s, sample: %s\n', ...
                        i, varNames{i}, mat2str(size(first_val)), mat2str(first_val(1:min(5, numel(first_val)))));
                else
                    fprintf('Column %d (%s): %s\n', i, varNames{i}, class(first_val));
                end
            elseif isnumeric(col_data) && numel(col_data) <= 20
                fprintf('Column %d (%s): %s\n', i, varNames{i}, mat2str(col_data(1, :)));
            elseif isnumeric(col_data)
                fprintf('Column %d (%s): numeric, shape: %s, sample: %s\n', ...
                    i, varNames{i}, mat2str(size(col_data)), mat2str(col_data(1, 1:min(5, size(col_data, 2)))));
            else
                fprintf('Column %d (%s): %s\n', i, varNames{i}, class(col_data));
            end
        end
        
        % Check for stim/response values
        if ismember('stim', varNames)
            stim_col = T.stim;
            if iscell(stim_col)
                all_stim = [];
                for j = 1:min(10, length(stim_col))
                    if ~isempty(stim_col{j})
                        all_stim = [all_stim; stim_col{j}(:)];
                    end
                end
                fprintf('\n=== Stim Values (sample) ===\n');
                fprintf('Unique values: %s\n', mat2str(unique(all_stim)));
            else
                fprintf('\n=== Stim Values ===\n');
                fprintf('Unique values: %s\n', mat2str(unique(stim_col(:))));
            end
        end
        
        if ismember('response', varNames)
            response_col = T.response;
            if iscell(response_col)
                all_response = [];
                for j = 1:min(10, length(response_col))
                    if ~isempty(response_col{j})
                        all_response = [all_response; response_col{j}(:)];
                    end
                end
                fprintf('\n=== Response Values (sample) ===\n');
                fprintf('Unique values: %s\n', mat2str(unique(all_response)));
            else
                fprintf('\n=== Response Values ===\n');
                fprintf('Unique values: %s\n', mat2str(unique(response_col(:))));
            end
        end
        
    else
        fprintf('Warning: T is not a MATLAB table, it is a %s\n', class(T));
        fprintf('Structure:\n');
        disp(T);
    end
    
    fprintf('\n=== Inspection Complete ===\n');
end

