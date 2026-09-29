classdef (TestTags = {'Python'}) tCrossLanguage < matlab.unittest.TestCase
    %TCROSSLANGUAGE MATLAB and Python read each other's files exactly.
    %   Needs PRISMT_PYTHON = a Python executable with prismt installed.

    properties
        Python
        Dir
    end

    methods (TestClassSetup)
        function findPython(tc)
            tc.Python = string(getenv('PRISMT_PYTHON'));
            tc.assumeTrue(strlength(tc.Python) > 0 && isfile(tc.Python), ...
                'Set PRISMT_PYTHON to run cross-language tests.');
        end
    end

    methods (TestMethodSetup)
        function tempFolder(tc)
            tc.Dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
        end
    end

    methods (Test)
        function pythonValidatesAMatlabDataset(tc)
            [ds, ~] = prismt.demo.makeSyntheticDataset(Profile="fast");
            f = prismt.writeDataset(ds, fullfile(tc.Dir, "from_matlab.mat"));
            [status, out] = tc.runPython("validate", f, "--json");
            tc.verifyEqual(status, 0, out);
            report = jsondecode(out);
            tc.verifyTrue(report.ok);
            s = report.summary;
            tc.verifyEqual([s.n_trials s.n_channels s.n_time s.n_modalities], [480 12 10 2]);
            tc.verifyEqual(s.n_subjects, 8);
            tc.verifyEqual(s.n_sessions, 16);
            tc.verifyEqual(string(s.channel_names(1)), "ch01");
        end

        function matlabReadsAPythonDataset(tc)
            f = fullfile(tc.Dir, "from_python.mat");
            [status, out] = tc.runPython("synth", "--profile", "tiny", "--out", f);
            tc.verifyEqual(status, 0, out);
            ds = prismt.loadDataset(f);
            tc.verifyEqual([ds.N ds.R ds.T ds.M], [128 6 6 2]);
            tc.verifyEqual(ds.ModalityNames, ["calcium"; "ach"]);
            tc.verifyEqual(categories(ds.Trials.phase), {'early'; 'late'});
            tc.verifyEqual(ds.ValueLabels.Label(ds.ValueLabels.Column == "stim"), ["CS-"; "CS+"]);
            tc.verifyEmpty(ds.validate());
        end

        function pythonReportsAnInvalidFileAsAnError(tc)
            f = fullfile(tc.Dir, "junk.mat");
            fid = fopen(f, 'w'); fwrite(fid, 'not a dataset'); fclose(fid);
            [status, out] = tc.runPython("validate", f, "--json");
            tc.verifyEqual(status, 2);
            report = jsondecode(out);
            tc.verifyFalse(report.ok);
        end
    end

    methods
        function [status, out] = runPython(tc, varargin)
            root = prismt.internal.repoRoot();
            args = strjoin(compose('"%s"', string(varargin)), ' ');
            cmd = sprintf('PYTHONPATH="%s" "%s" -m prismt %s 2>/dev/null', fullfile(root, 'src'), tc.Python, args);
            [status, out] = system(cmd);
            out = strtrim(out);
        end
    end
end
