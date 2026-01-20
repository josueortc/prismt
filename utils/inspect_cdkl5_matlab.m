% Script to inspect CDKL5 MATLAB data structure
% Usage: matlab -batch "inspect_cdkl5_matlab('/path/to/cdkl5_data_for_josue_w_states.mat')"

function inspect_cdkl5_matlab(mat_file_path)
    fprintf('Loading MATLAB file: %s\n', mat_file_path);
    
    % Load the data
    data = load(mat_file_path);
    
    fprintf('\n=== Variables in file ===\n');
    var_names = fieldnames(data);
    for i = 1:length(var_names)
        fprintf('  %s\n', var_names{i});
    end
    
    % Inspect wild type struct
    if isfield(data, 'cdkl5_m_wt_struct')
        fprintf('\n=== cdkl5_m_wt_struct (Wild Type Males) ===\n');
        wt_struct = data.cdkl5_m_wt_struct;
        fprintf('Type: %s\n', class(wt_struct));
        fprintf('Size: %s\n', mat2str(size(wt_struct)));
        
        % Handle both tables and struct arrays
        if istable(wt_struct)
            n_animals = height(wt_struct);
            fprintf('Number of animals: %d\n', n_animals);
        else
            n_animals = length(wt_struct);
            fprintf('Number of animals: %d\n', n_animals);
        end
        
        if n_animals > 0
            fprintf('\nFirst animal structure:\n');
            if istable(wt_struct)
                first_animal = wt_struct(1, :);
                field_names = wt_struct.Properties.VariableNames;
            else
                first_animal = wt_struct(1);
                field_names = fieldnames(first_animal);
            end
            fprintf('Number of fields: %d\n', length(field_names));
            fprintf('Field names:\n');
            for i = 1:length(field_names)
                field_name = field_names{i};
                field_data = first_animal.(field_name);
                fprintf('  %s: %s', field_name, class(field_data));
                if isnumeric(field_data) || islogical(field_data)
                    fprintf(', size: %s', mat2str(size(field_data)));
                    if numel(field_data) <= 10
                        fprintf(', value: %s', mat2str(field_data));
                    end
                elseif iscell(field_data)
                    fprintf(', cell array, size: %s', mat2str(size(field_data)));
                    if length(field_data) > 0 && isnumeric(field_data{1})
                        fprintf(', first element size: %s', mat2str(size(field_data{1})));
                    end
                elseif ischar(field_data) || isstring(field_data)
                    fprintf(', value: %s', mat2str(field_data));
                end
                fprintf('\n');
            end
        end
    end
    
    % Inspect mutant struct
    if isfield(data, 'cdkl5_m_mut_struct')
        fprintf('\n=== cdkl5_m_mut_struct (Mutant Males) ===\n');
        mut_struct = data.cdkl5_m_mut_struct;
        fprintf('Type: %s\n', class(mut_struct));
        fprintf('Size: %s\n', mat2str(size(mut_struct)));
        
        % Handle both tables and struct arrays
        if istable(mut_struct)
            n_animals = height(mut_struct);
            fprintf('Number of animals: %d\n', n_animals);
        else
            n_animals = length(mut_struct);
            fprintf('Number of animals: %d\n', n_animals);
        end
        
        if n_animals > 0
            fprintf('\nFirst animal structure:\n');
            if istable(mut_struct)
                first_animal = mut_struct(1, :);
                field_names = mut_struct.Properties.VariableNames;
            else
                first_animal = mut_struct(1);
                field_names = fieldnames(first_animal);
            end
            fprintf('Number of fields: %d\n', length(field_names));
            fprintf('Field names:\n');
            for i = 1:length(field_names)
                field_name = field_names{i};
                field_data = first_animal.(field_name);
                fprintf('  %s: %s', field_name, class(field_data));
                if isnumeric(field_data) || islogical(field_data)
                    fprintf(', size: %s', mat2str(size(field_data)));
                    if numel(field_data) <= 10
                        fprintf(', value: %s', mat2str(field_data));
                    end
                elseif iscell(field_data)
                    fprintf(', cell array, size: %s', mat2str(size(field_data)));
                    if length(field_data) > 0 && isnumeric(field_data{1})
                        fprintf(', first element size: %s', mat2str(size(field_data{1})));
                    end
                elseif ischar(field_data) || isstring(field_data)
                    fprintf(', value: %s', mat2str(field_data));
                end
                fprintf('\n');
            end
        end
    end
    
    fprintf('\n=== Inspection Complete ===\n');
end
