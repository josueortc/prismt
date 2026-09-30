function results = runPrismtTests(opts)
%RUNPRISMTTESTS Run PRISMT's MATLAB tests.
%
%   results = runPrismtTests()                         % everything that can run here
%   results = runPrismtTests(Exclude=["UI","Python"])  % skip tagged tests
%   results = runPrismtTests(Only="Python")            % only tagged tests
%   results = runPrismtTests(Name="tDataset")          % one test class
%
%   Tags: Python (needs PRISMT_PYTHON = path of a Python with prismt installed),
%   UI (builds the app), Slow. From a terminal:
%     matlab -batch "addpath('tests/matlab'); assertSuccess(runPrismtTests())" > test.log 2>&1
arguments
    opts.Exclude (1, :) string = strings(1, 0)
    opts.Only (1, :) string = strings(1, 0)
    opts.Name (1, :) string = strings(1, 0)
end
import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.selectors.HasTag

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
addpath(fullfile(root, 'matlab'), here);

suite = TestSuite.fromFolder(here);
if ~isempty(opts.Only)
    selector = HasTag(opts.Only(1));
    for k = 2:numel(opts.Only)
        selector = selector | HasTag(opts.Only(k));
    end
    suite = suite.selectIf(selector);
end
for tag = opts.Exclude
    suite = suite.selectIf(~HasTag(tag));
end
if ~isempty(opts.Name)
    keep = false(size(suite));
    for k = 1:numel(opts.Name)
        keep = keep | contains({suite.Name}, opts.Name(k));
    end
    suite = suite(keep);
end
runner = TestRunner.withTextOutput();
recorder = matlab.unittest.plugins.DiagnosticsRecordingPlugin;
runner.addPlugin(recorder);
results = runner.run(suite);
if strcmp(getenv('GITHUB_ACTIONS'), 'true')
    % Failures as GitHub annotations, readable without opening the log.
    for k = find([results.Failed])
        text = results(k).Name;
        try
            rec = results(k).Details.DiagnosticRecord;
            if ~isempty(rec), text = text + ": " + string(rec(1).Report); end
        catch
        end
        text = replace(extractBefore(text + " ", min(strlength(text) + 1, 1500)), newline, "%0A");
        fprintf('::error::%s\n', text);
    end
end
fprintf('\n%d passed, %d failed, %d incomplete (skipped), %.1f s\n', nnz([results.Passed]), ...
    nnz([results.Failed]), nnz([results.Incomplete]), sum([results.Duration]));
end
