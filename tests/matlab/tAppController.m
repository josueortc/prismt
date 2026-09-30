classdef tAppController < matlab.unittest.TestCase
    %TAPPCONTROLLER The app's logic without a window or Python: settings, classes, filters,
    %checks and script export.

    properties
        Dir
        C
    end

    methods (TestMethodSetup)
        function setup(tc)
            tc.Dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
            old = getenv('PRISMT_SETTINGS_DIR');
            setenv('PRISMT_SETTINGS_DIR', fullfile(tc.Dir, 'settings'));
            tc.addTeardown(@() setenv('PRISMT_SETTINGS_DIR', old));
            prismt.internal.projectFolder(fullfile(tc.Dir, 'project'));
            tc.C = prismt.app.AppController();
            tc.C.useDemo("tiny");
        end
    end

    methods (Test)
        function demoDataIsSavedAndLabelled(tc)
            c = tc.C;
            tc.verifyTrue(isfile(c.DatasetFile));
            tc.verifyTrue(startsWith(c.DatasetFile, fullfile(tc.Dir, 'project', 'datasets')));
            tc.verifyEqual(string(c.value("labels.column", "")), "phase", "a likely label is preselected");
            T = c.classCounts();
            tc.verifyEqual(T.Value, ["early"; "late"]);
            tc.verifyEqual(sum(T.Trials), c.Dataset.N);
            tc.verifyEqual(T.Subjects, [4; 4]);
        end

        function tasksSwitchCleanly(tc)
            c = tc.C;
            c.setTask("finetune");
            tc.verifyEqual(c.task(), "finetune");
            tc.verifyEqual(string(c.Config.task), "classify");
            tc.verifyTrue(isfield(c.Config.model, 'init_from'));
            c.setTask("mae");
            tc.verifyEqual(c.task(), "mae");
            tc.verifyFalse(isfield(c.Config.model, 'init_from'), "no stale init_from on an autoencoder run");
            c.setTask("classify");
            tc.verifyEqual(c.task(), "classify");
        end

        function classesMergeAndFiltersCount(tc)
            c = tc.C;
            c.setLabel("stim");
            c.setClasses(["CS+", "CS-"], ["any", "any"]);
            tc.verifyNumElements(c.Config.labels.classes, 1);
            tc.verifyEqual(string(c.Config.labels.classes{1}.values), ["CS+", "CS-"]);
            issues = prismt.app.listIssues(c);
            tc.verifyTrue(any(contains([issues.message], "At least two classes")));
            c.setClasses(["CS+", "CS-"]);
            c.setFilter("mouse", ["M01", "M02"]);
            tc.verifyEqual(nnz(c.keptTrials()), c.Dataset.N / 2);
            tc.verifyEqual(sum(c.classCounts().Trials), c.Dataset.N / 2);
            c.setFilter("mouse", strings(0, 1));
            tc.verifyEqual(nnz(c.keptTrials()), c.Dataset.N);
            c.setFilter("mouse", "nobody");
            issues = prismt.app.listIssues(c);
            tc.verifyTrue(any(contains([issues.message], "remove every trial")));
        end

        function settingsSetAndClear(tc)
            c = tc.C;
            c.setValue("train.lr", 3e-4);
            tc.verifyEqual(c.value("train.lr", []), 3e-4);
            c.setValue("train.lr", []);
            tc.verifyFalse(isfield(c.Config, 'train') && isfield(c.Config.train, 'lr'), "empty means default");
            c.setValue("hpo.n_trials", 5);
            c.setValue("hpo", []);
            tc.verifyFalse(isfield(c.Config, 'hpo'));
        end

        function checksWithoutPythonExplainWhatToDo(tc)
            issues = prismt.app.listIssues(tc.C);
            first = issues(1);
            tc.verifyEqual(first.level, "error");
            tc.verifySubstring(first.hint, "Setup");
            tc.verifyEqual(first.field, "setup");
        end

        function exportedScriptRebuildsTheSettings(tc)
            c = tc.C;
            c.setLabel("stim");
            c.setClasses(["CS+", "CS-"], ["plus", "minus"]);
            c.setFilter("mouse", ["M01", "M02", "M03"]);
            c.setValue("preprocess.time_window_s", [0 0.3]);
            c.setValue("train.epochs", 7);
            file = fullfile(tc.Dir, "exported.m");
            c.exportScript(file);
            text = string(fileread(file));
            tc.verifySubstring(text, "prismt.train(cfg, Wait=true)");
            % run only the part that builds cfg, and compare with what the app would send
            lines = splitlines(text);
            stop = find(startsWith(lines, "report ="), 1);
            eval(strjoin(lines(1:stop - 1), newline));
            want = c.runConfig();
            tc.verifyEqual(jsondecode(jsonencode(cfg)), jsondecode(jsonencode(want))); %#ok<NODEF>
        end

        function signalsGroupsAndCombining(tc)
            c = tc.C;
            ds = c.Dataset;
            groups = repmat("front", ds.R, 1); groups(end - 1:end) = "back";
            c.setChannelGroups(groups);
            tc.verifyEqual(c.Dataset.ChannelGroups, groups, "saved in the dataset file");
            tc.verifyEqual(prismt.loadDataset(c.DatasetFile).ChannelGroups, groups);
            c.setValue("selection.channel_groups", {'back'});
            c.setValue("selection.modalities", {'ach'});
            cfg = c.runConfig();
            tc.verifyEqual(string(cfg.selection.channel_groups), "back");
            % another recording with other channels, joined by name
            X = rand(10, 2, ds.T, 1);
            other = prismt.makeDataset(X, table(repmat("M99", 10, 1), repmat("early", 10, 1), 'VariableNames', {'mouse', 'phase'}), ...
                Times=ds.Times, ChannelNames=["extra1", "extra2"], ModalityNames="emg", ModalityKinds="physiology", Subject="mouse");
            f = prismt.writeDataset(other, fullfile(tc.Dir, "other.mat"));
            notes = c.addDatasets(f);
            tc.verifyEqual(c.Dataset.N, ds.N + 10);
            tc.verifyEqual(c.Dataset.ModalityNames, [ds.ModalityNames; "emg"]);
            tc.verifyEqual(c.Dataset.R, ds.R + 2);
            tc.verifyTrue(any(contains(notes, "lacks")));
            tc.verifyTrue(ismember("source_dataset", string(c.Dataset.Trials.Properties.VariableNames)));
            tc.verifyFalse(isfield(c.Config.selection, 'modalities'), "a new dataset resets the signal choice");
        end

        function toCodeRoundTrips(tc)
            s = struct('a', 'text', 'b', 3.25, 'c', [1 2 3], 'd', true, 'e', {{'x', 'y'}}, ...
                'f', struct('g', "str", 'h', {{struct('name', 'n', 'values', {{'v1', 'v2'}})}}), 'i', []);
            lines = prismt.internal.toCode(s, "out");
            eval(strjoin(lines, newline));
            tc.verifyEqual(out, s); %#ok<NODEF>
        end
    end
end
