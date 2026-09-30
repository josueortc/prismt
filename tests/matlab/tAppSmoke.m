classdef (TestTags = {'UI', 'Python'}) tAppSmoke < matlab.unittest.TestCase
    %TAPPSMOKE The app, without a window, driven the way a user would: demo data, the three
    %tasks, a tiny run on each, results plots, a cluster job folder and an exported script.
    %   Screenshots of every step go to PRISMT_TEST_PNG_DIR when it is set (to look at them).

    properties
        Dir
        Png
        App
        C
    end

    methods (TestClassSetup)
        function setup(tc)
            tc.assumeTrue(isfile(getenv('PRISMT_PYTHON')), 'Set PRISMT_PYTHON to run the app tests.');
            tc.Dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
            tc.Png = string(getenv('PRISMT_TEST_PNG_DIR'));
            if strlength(tc.Png) == 0, tc.Png = fullfile(tc.Dir, "png"); end
            if ~isfolder(tc.Png), mkdir(tc.Png); end
            old = getenv('PRISMT_SETTINGS_DIR');
            setenv('PRISMT_SETTINGS_DIR', fullfile(tc.Dir, 'settings'));
            tc.addTeardown(@() setenv('PRISMT_SETTINGS_DIR', old));
            prismt.internal.projectFolder(fullfile(tc.Dir, 'project'));
            tc.C = prismt.app.AppController();
            tc.C.setPython(string(getenv('PRISMT_PYTHON')));
            tc.App = prismt.gui(Visible="off", Controller=tc.C);
            tc.addTeardown(@() tc.App.close(true));
            tc.shot("01_setup");
            tc.App.show("Data");
            tc.C.useDemo("tiny");
            tc.C.setValue("train", struct('epochs', 4, 'min_epochs', 0, 'device', 'cpu'));
        end
    end

    methods (Test, TestTags = {'Slow'})
        function a_dataPreviews(tc)
            d = tc.App.show("Data");
            tc.verifyTrue(contains(d.Summary.Text, "128 trials"));
            tc.verifyTrue(contains(string(d.Issues.Text), "None"));
            tc.shot("02_data_average");
            d.Controls.avgBy.Value = "stim";
            d.Previews.SelectedTab = d.Previews.Children(1);
            tc.App.safely(@() d.Controls.avgBy.ValueChangedFcn(d.Controls.avgBy, []));
            tc.shot("02_data_difference");
            names = ["conditions", "trial", "map", "crosstab"];
            for k = 2:5
                d.Previews.SelectedTab = d.Previews.Children(k);
                tc.shot("02_data_" + names(k - 1));
            end
        end

        function b_classifyRunsAndShowsResults(tc)
            t = tc.App.show("Task");
            tc.C.setTask("classify");
            t.Classes.column.Value = "stim";
            t.Classes.column.ValueChangedFcn(t.Classes.column, []);
            tc.verifyEqual(string(t.Classes.table.Data(:, 2))', ["CS+", "CS-"]);
            t.Checks.runCheck();
            levels = string({t.Checks.Items.level});
            tc.verifyFalse(any(levels == "error"), strjoin([t.Checks.Items.message], " | "));
            tc.verifyTrue(any(levels == "ok"));
            tc.verifySubstring(string(t.Testing.split.Text), "fold");
            tc.shot("03_task_classify");
            tc.App.show("Training");
            tc.shot("04_training");
            tr = tc.App.Tabs.Training;
            tr.ShowAdvanced.Value = true; tr.refresh();
            tc.shot("04_training_all_settings");
            tr.ShowAdvanced.Value = false; tr.refresh();

            r = tc.App.show("Run");
            tc.verifyEqual(string(r.Local.start.Enable), "on");
            r.start();
            tc.shot("05_run_started");
            s = tc.waitForRun();
            tc.assertEqual(string(s.state), "finished", tc.runLog());
            r.tick();
            tc.verifySubstring(string(r.Local.message.Text), "Done");
            tc.shot("05_run_finished");

            res = tc.App.show("Results");
            res.openRun(tc.C.Run.RunDir);
            tc.verifyGreaterThanOrEqual(height(res.RunTable), 1);
            for p = res.ClassifyPlots
                res.show(p);
                tc.shot("06_results_" + lower(regexprep(p, '\W+', '_')));
            end
            f = res.popOut(false);
            tc.verifyTrue(isgraphics(f));
            close(f);
        end

        function c_autoencoderThenFinetune(tc)
            t = tc.App.show("Task");
            tc.C.setTask("mae");
            tc.verifyEqual(string(t.Mae.panel.Visible), "on");
            for strat = ["channel", "forecast", "modality", "random"]
                tc.C.setValue("mae.mask.strategy", char(strat));
                tc.shot("03_task_mae_" + strat);
            end
            r = tc.App.show("Run");
            r.start();
            s = tc.waitForRun();
            tc.assertEqual(string(s.state), "finished", tc.runLog());
            maeRun = tc.C.Run.RunDir;
            res = tc.App.show("Results");
            res.openRun(maeRun);
            for p = res.MaePlots
                res.show(p);
                tc.shot("06_results_mae_" + lower(regexprep(p, '\W+', '_')));
            end

            t = tc.App.show("Task");
            tc.C.setTask("finetune");
            tc.C.setLabel("stim");
            items = string(t.Classes.initFrom.ItemsData);
            tc.verifyTrue(ismember(string(maeRun), items), "the finished autoencoder run is offered");
            t.Classes.initFrom.Value = string(maeRun);
            t.Classes.initFrom.ValueChangedFcn(t.Classes.initFrom, []);
            tc.shot("03_task_finetune");
            t.Checks.runCheck();
            tc.verifyFalse(any(string({t.Checks.Items.level}) == "error"), strjoin([t.Checks.Items.message], " | "));
            r = tc.App.show("Run");
            r.start();
            s = tc.waitForRun();
            tc.assertEqual(string(s.state), "finished", tc.runLog());
            tc.C.setTask("classify");
        end

        function d_clusterJobFolder(tc)
            s = tc.App.show("Setup");
            s.Cluster.host.Value = "cluster.example.edu";
            s.Cluster.user.Value = "me";
            s.Cluster.gpus.Value = "0";
            s.saveCluster();
            r = tc.App.show("Run");
            r.Where.Value = "cluster";
            r.refresh();
            job = r.createJob();
            tc.verifyTrue(isfile(fullfile(job.Folder, "submit.sh")));
            tc.verifyTrue(isfolder(fullfile(job.Folder, "data")), "the dataset travels with the job");
            tc.verifyTrue(any(contains(string(r.Cluster.readme.Value), "me@cluster.example.edu")));
            tc.shot("05_run_cluster");
            r.Where.Value = "local";
            r.refresh();
        end

        function e_exportedScriptRuns(tc)
            tc.App.Dialogs.Answers = {char(fullfile(tc.Dir, "exported_run.m"))};
            file = tc.App.Tabs.Run.exportScript();
            tc.verifyTrue(isfile(file));
            out = evalc("run('" + file + "')");
            tc.verifySubstring(lower(string(out)), "balanced accuracy", "the script trains and prints the summary");
        end

        function f_settingsAreValidated(tc)
            tr = tc.App.show("Training");
            err = captureError(@() tr.setField("train.lr", "fast"));
            tc.verifyEqual(string(err.identifier), "prismt:app");
            tc.verifySubstring(string(err.message), "must be a number");
            tr.setField("preprocess.time_window_s", "0, 0.3");
            tc.verifyEqual(tc.C.value("preprocess.time_window_s", []), [0 0.3]);
            tr.setField("preprocess.time_window_s", "");
            tc.verifyFalse(isfield(tc.C.Config, 'preprocess') && isfield(tc.C.Config.preprocess, 'time_window_s'));
            tr.setField("model.time_patch", "all");
            tc.verifyEqual(tc.C.value("model.time_patch", []), 'all');
            tc.verifySubstring(string(tr.PresetNote.Text), "Custom");
            tc.C.setValue("model", []);
        end

        function g_importWithOptionsChannelsAndSignals(tc)
            % A lab table: sessions with different channel counts, channels in 2 blocks
            % (two signals), and a behavior time series.
            T = 6;
            sess = {rand(8, 4, T); rand(6, 4, T); rand(8, 6, T); rand(7, 6, T); rand(8, 4, T); rand(6, 6, T)};
            n = cellfun(@(x) size(x, 1), sess);
            cond = arrayfun(@(k) double(rand(k, 1) > 0.5), n, 'UniformOutput', false);
            speed = arrayfun(@(k) rand(k, T), n, 'UniformOutput', false);
            who = {'a'; 'a'; 'b'; 'b'; 'c'; 'c'};
            tbl = table(sess, cond, speed, who, 'VariableNames', {'lfp', 'go', 'speed', 'rat'});
            f = fullfile(tc.Dir, "lab_table.mat");
            T = tbl; save(f, 'T');
            d = tc.App.show("Data");
            d.startImport(f);
            tc.verifyEqual(string(d.Import.panel.Visible), "on");
            tc.verifySubstring(string(d.Import.contains.Text), "4-6 channels");
            tc.verifyEqual(string(d.Import.subject.Value), "rat", "the subject column is recognized");
            d.Import.behavior.Value = {'speed'};
            d.Import.kind.Value = "neural";
            d.Import.fs.Value = 20;
            d.Import.event.Value = "lever press";
            d.previewImport();
            tc.verifySubstring(string(d.Import.result.Text), "speed");
            tc.shot("02_data_import_options");
            d.saveImport();
            ds = tc.C.Dataset;
            tc.verifyEqual(ds.ModalityNames, ["lfp"; "behavior"]);
            tc.verifyEqual(ds.ModalityKinds, ["neural"; "behavior"]);
            tc.verifyEqual(ds.Event, "lever press");
            tc.verifyEqual(ds.Subject, "rat");
            tc.verifyTrue(all(isnan(ds.X(1:8, 5:6, :, 1)), 'all'), "session 1 has 4 of the 6 channels");
            % channel groups on the Channels tab
            data = d.Channels.table.Data;
            data(1:3, 3) = {'left'}; data(4:6, 3) = {'right'};
            d.Channels.table.Data = data;
            d.saveGroups();
            tc.verifyEqual(tc.C.Dataset.ChannelGroups(1:6), ["left"; "left"; "left"; "right"; "right"; "right"]);
            d.Previews.SelectedTab = d.Previews.Children(6);
            tc.shot("02_data_channels");
            % choose signals and groups on the Task tab, then check with Python
            t = tc.App.show("Task");
            tc.C.setTask("classify");
            tc.C.setLabel("go");
            tc.verifyEqual(string(t.Inputs.groups.Visible), "on");
            t.Inputs.groups.Value = {'left'};
            t.Inputs.groups.ValueChangedFcn(t.Inputs.groups, []);
            t.Inputs.signals.Value = {'lfp'};
            t.Inputs.signals.ValueChangedFcn(t.Inputs.signals, []);
            tc.verifySubstring(string(t.Inputs.using.Text), "1 of 2 signals and 3 of 7 channels");
            t.Checks.runCheck();
            r = tc.C.Check;
            tc.verifyTrue(r.ok, strjoin([t.Checks.Items.message], " | "));
            tc.verifyEqual(string(r.selection.channels(:))', ["ch01", "ch02", "ch03"]);
            tc.verifyEqual(string(r.selection.modalities), "lfp");
            tc.verifyEqual(string(r.split.scheme), "loo", "3 subjects: every one tested once");
            tc.shot("03_task_channels_and_signals");
        end
    end

    methods
        function shot(tc, name)
            tc.App.exportFigure(fullfile(tc.Png, name + ".png"));
        end

        function s = waitForRun(tc)
            run = tc.C.Run;
            t0 = tic;
            while run.isActive() && toc(t0) < 600
                pause(1);
            end
            s = run.status();
        end

        function text = runLog(tc)
            f = fullfile(tc.C.Run.RunDir, "stdout.txt");
            text = "";
            if isfile(f), text = string(fileread(f)); end
        end
    end
end
