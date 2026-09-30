classdef (TestTags = {'Python', 'RealData'}) tRealData < matlab.unittest.TestCase
    %TREALDATA Import a real lab file and check a classification on it (no training).
    %   Opt-in: set PRISMT_REAL_DATA to a tableForModeling_v2-style .mat file (one row per
    %   session, with a meta struct). Real data is never committed.

    methods (Test)
        function importsAndPlansAClassification(tc)
            file = string(getenv('PRISMT_REAL_DATA'));
            tc.assumeTrue(isfile(file), 'Set PRISMT_REAL_DATA to a tableForModeling_v2 file to run this test.');
            tc.assumeTrue(isfile(getenv('PRISMT_PYTHON')), 'Set PRISMT_PYTHON.');
            dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
            [ds, rep] = prismt.importData(file, Atlas="grid82", Behavior=["runSpeed", "faceMotion"]);
            tc.verifyNotEmpty(rep.notes);
            tc.verifyEqual(ds.ModalityNames, ["calcium"; "behavior"]);
            tc.verifyEqual(ds.T, 41, "tableForModeling trials are 41 frames");
            tc.verifyEqual(ds.ValueLabels.Label(ds.ValueLabels.Column == "stim")', ["CS-", "CS+"]);
            issues = ds.validate();
            tc.verifyFalse(any(string({issues.Level}) == "error"), strjoin(string({issues.Message}), " | "));
            f = prismt.writeDataset(ds, fullfile(dir, "real_prismt.mat"));
            cfg = prismt.defaultConfig("classify", f, Label="phase");
            cfg.labels.classes = {struct('name', 'early', 'values', {{'early'}}), struct('name', 'late', 'values', {{'late'}})};
            cfg.selection.max_trials_per_session = 20;
            cfg.preprocess = struct('time_window_s', [0 3], 'bin_width_s', 0.3);
            r = prismt.check(cfg);
            tc.verifyTrue(r.ok, "check failed");
            tc.verifyEqual(string(r.split.test_on), "subject", "real data has enough animals to test on new ones");
            tc.verifyEqual(r.model.tokens_per_trial, (82 + 2) * 10 + 1);
        end
    end
end
