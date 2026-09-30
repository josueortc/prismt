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
