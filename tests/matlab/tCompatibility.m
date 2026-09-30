classdef tCompatibility < matlab.unittest.TestCase
    %TCOMPATIBILITY PRISMT's MATLAB code runs on R2021a and needs no toolboxes.
    %   The source scan catches functions and properties added after R2021a; CI also runs the
    %   tests on R2021a itself.

    properties (Constant)
        % pattern -> why it is not allowed (introduced after R2021a, or not in base MATLAB)
        Banned = { ...
            '\<dictionary\(', 'dictionary is R2022b'; ...
            '\<writelines\(', 'writelines is R2022a'; ...
            '\<readlines\(', 'readlines is R2020b but behaves differently; use fileread + splitlines'; ...
            '\<fontsize\(', 'fontsize is R2022a'; ...
            '\<clim\(', 'clim is R2022a; use prismt.plot.setLimits'; ...
            '\<uiaccordion\>', 'uiaccordion is R2025a'; ...
            '\<isMATLABReleaseOlderThan\(', 'isMATLABReleaseOlderThan is R2020b and not in all installs'; ...
            'ClickedFcn', 'ClickedFcn is R2022b; use CellSelectionCallback'; ...
            'DoubleClickedFcn', 'DoubleClickedFcn is R2021b'; ...
            'ValueChangingFcn', 'ValueChangingFcn on text areas is R2022b'; ...
            '''Selection''', 'uitable Selection is R2021b'; ...
            'SelectionChangedFcn.*uitable', 'uitable SelectionChangedFcn is R2021b'; ...
            '\<pyrun\(|\<pyenv\(', 'PRISMT never runs Python inside MATLAB'; ...
            '\<sgtitle\(|\<nnz\(isstring', ''; ...
            '\.Theme\s*=', ''; ...
            '\.Placeholder\s*=', ''}
        % files allowed to use a pattern, because they check the release (isprop / try) first
        Allowed = struct('Theme', ["+app/App.m", "+plot/prepareAxes.m"], 'Placeholder', "+app/ui.m", ...
            'clim', "+plot/setLimits.m")
    end

    methods (Test)
        function noApisNewerThanR2021a(tc)
            files = tc.sources();
            problems = strings(0, 1);
            for f = files'
                lines = splitlines(string(fileread(f)));
                code = regexprep(lines, '%.*$', '');         % ignore comments
                for k = 1:size(tc.Banned, 1)
                    hit = find(~cellfun(@isempty, regexp(code, tc.Banned{k, 1}, 'once')));
                    if isempty(hit), continue; end
                    if tc.guarded(tc.Banned{k, 1}, f), continue; end
                    why = string(tc.Banned{k, 2});
                    if strlength(why) == 0, why = "only allowed behind an isprop check in one place"; end
                    problems = [problems; relative(f) + ":" + hit + "  " + why]; %#ok<AGROW>
                end
            end
            tc.verifyEmpty(problems, strjoin(problems, newline));
        end

        function needsNoToolboxes(tc)
            files = tc.sources();
            [~, products] = matlab.codetools.requiredFilesAndProducts(cellstr(files));
            names = string({products.Name});
            tc.verifyEqual(names, "MATLAB", "PRISMT must run with base MATLAB only: " + strjoin(names, ", "));
        end
    end

    methods
        function files = sources(~)
            root = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'matlab');
            d = dir(fullfile(root, '**', '*.m'));
            files = string(fullfile({d.folder}, {d.name}))';
        end

        function tf = guarded(tc, pattern, file)
            tf = false;
            for name = string(fieldnames(tc.Allowed))'
                if contains(pattern, name) && any(endsWith(file, tc.Allowed.(name)))
                    text = string(fileread(file));
                    tf = contains(text, "isprop(") || contains(text, "exist(");
                end
            end
        end
    end
end

function r = relative(f)
r = extractAfter(f, "matlab" + filesep);
end
