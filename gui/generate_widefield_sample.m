function generate_widefield_sample()
%GENERATE_WIDEFIELD_SAMPLE Generate sample training script for widefield data
%
%   Creates a pre-filled configuration and run script suitable for
%   widefield phase classification (early vs late).
%
%   Edit DATA_PATH below to point to your standardized .mat file.
%
%   Usage: generate_widefield_sample()

    % === EDIT THIS PATH to your standardized .mat file ===
    DATA_PATH = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'standardized_widefield.mat');
    % To create standardized data, run in MATLAB:
    %   cd(fullfile(fileparts(fileparts(which('generate_widefield_sample'))), 'scripts'))
    %   standardize_data('your_widefield.mat', 'standardized_widefield.mat', 'widefield')

    OUT_DIR = fullfile(pwd, 'generated_widefield_sample');
    mkdir(OUT_DIR);

    % Widefield-typical parameters
    config = struct();
    config.data_path = DATA_PATH;
    config.data_type = 'dff';
    config.task_type = 'phase';
    config.phase1 = 'early';
    config.phase2 = 'late';
    config.batch_size = 16;
    config.epochs = 100;
    config.learning_rate = '5e-5';
    config.val_split = 0.2;
    config.seed = 42;

    project_root = fileparts(fileparts(mfilename('fullpath')));
    train_script = fullfile(project_root, 'train.py');

    cmd = sprintf(['python %s --data_path %s --data_type %s --task_type %s ' ...
        '--phase1 %s --phase2 %s --batch_size %d --epochs %d --learning_rate %s ' ...
        '--val_split %.2f --seed %d --save_dir results'], ...
        train_script, config.data_path, config.data_type, config.task_type, ...
        config.phase1, config.phase2, config.batch_size, config.epochs, ...
        config.learning_rate, config.val_split, config.seed);

    % Write SLURM script
    slurm_path = fullfile(OUT_DIR, 'run_widefield_training.sh');
    fid = fopen(slurm_path, 'w');
    fprintf(fid, '#!/bin/bash\n');
    fprintf(fid, '#SBATCH --job-name=prismt_widefield\n');
    fprintf(fid, '#SBATCH --partition=gpu\n');
    fprintf(fid, '#SBATCH --gres=gpu:1\n');
    fprintf(fid, '#SBATCH --mem=32G\n');
    fprintf(fid, '#SBATCH --time=24:00:00\n');
    fprintf(fid, '#SBATCH --output=%%j.out\n');
    fprintf(fid, '#SBATCH --error=%%j.err\n\n');
    fprintf(fid, 'cd %s\n\n', project_root);
    fprintf(fid, '%s\n', cmd);
    fclose(fid);

    % Write config summary
    config_path = fullfile(OUT_DIR, 'widefield_config.txt');
    fid = fopen(config_path, 'w');
    fprintf(fid, 'PRISMT Widefield Sample Configuration\n');
    fprintf(fid, '=====================================\n');
    fprintf(fid, 'Data path: %s\n', config.data_path);
    fprintf(fid, 'Data type: %s (ΔF/F)\n', config.data_type);
    fprintf(fid, 'Task: Phase classification (%s vs %s)\n', config.phase1, config.phase2);
    fprintf(fid, 'Normalization: scale_20 (default)\n');
    fprintf(fid, 'Tokenization: Spatial (brain areas as tokens)\n');
    fprintf(fid, 'Batch size: %d\n', config.batch_size);
    fprintf(fid, 'Epochs: %d\n', config.epochs);
    fprintf(fid, 'Learning rate: %s\n', config.learning_rate);
    fprintf(fid, 'Val split: %.2f\n', config.val_split);
    fprintf(fid, '\nTo run locally:\n%s\n', cmd);
    fprintf(fid, '\nTo submit:\nsbatch %s\n', slurm_path);
    fclose(fid);

    fprintf('Sample widefield configuration generated.\n');
    fprintf('  SLURM script: %s\n', slurm_path);
    fprintf('  Config: %s\n', config_path);
    fprintf('\nEdit DATA_PATH in this script if needed, then:\n');
    fprintf('  sbatch %s\n', slurm_path);
end
