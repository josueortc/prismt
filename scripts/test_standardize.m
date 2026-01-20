% Test script for standardize_data.m
% Tests both widefield and CDKL5 data standardization

fprintf('=== Testing PRISMT Standardization Script ===\n\n');

% Test CDKL5 data
cdkl5_input = '/Users/josueortegacaro/Documents/rachel_cdkl5/cdkl5_data_for_josue_w_states.mat';
cdkl5_output = '/Users/josueortegacaro/repos/prismt/test_cdkl5_standardized.mat';

if exist(cdkl5_input, 'file')
    fprintf('Testing CDKL5 standardization...\n');
    fprintf('Input: %s\n', cdkl5_input);
    fprintf('Output: %s\n', cdkl5_output);
    
    try
        standardize_data(cdkl5_input, cdkl5_output, 'cdkl5');
        fprintf('✓ CDKL5 standardization successful!\n\n');
    catch ME
        fprintf('✗ CDKL5 standardization failed: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
else
    fprintf('⚠ CDKL5 input file not found: %s\n', cdkl5_input);
end

fprintf('\n=== Test Complete ===\n');
