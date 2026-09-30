%% PRISMT tutorial: from demo data to results, without the app
% Everything the app does can be done in a script like this one. Run it section by
% section (Ctrl+Enter). It uses demo data with known structure, so you can check that
% PRISMT finds what was planted:
%   - on CS+ trials, an evoked calcium response in two "posterior" channels;
%   - in late sessions, an evoked ACh response in two "frontal" channels;
%   - smooth activity shared across channels (so hidden values can be predicted),
%     one channel of pure noise, and some missing data.
% Time on a laptop: about 5 minutes in total.

%% 0. Set-up
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));                     % the matlab/ folder with +prismt
if ~exist('outFolder', 'var'), outFolder = fullfile(prismt.internal.projectFolder(), 'tutorial'); end
if ~exist('profile', 'var'), profile = "fast"; end
if ~isfolder(outFolder), mkdir(outFolder); end
figFolder = fullfile(outFolder, 'figures');
if ~isfolder(figFolder), mkdir(figFolder); end
report = prismt.env.doctor(prismt.env.python());     % is Python ready? (Setup tab in the app)
fprintf('Python ready: %d (device: %s)\n', report.ok, report.device);

%% 1. Make the demo dataset and save it as a PRISMT dataset file
[ds, truth] = prismt.demo.makeSyntheticDataset(Profile=profile);
disp(ds)
dataFile = prismt.writeDataset(ds, fullfile(outFolder, 'demo_prismt.mat'));

%% 2. Look at the data first
f = figure('Visible', 'off', 'Position', [100 100 1200 800]);
t = tiledlayout(f, 2, 2, 'TileSpacing', 'compact');
prismt.plot.meanHeatmap(nexttile(t), ds, Modality="calcium", ...
    Difference={ds.Trials.stim == 1, ds.Trials.stim == 0}, Labels=["CS+", "CS-"]);
prismt.plot.meanHeatmap(nexttile(t), ds, Modality="ach", ...
    Difference={ds.Trials.phase == "late", ds.Trials.phase == "early"}, Labels=["late", "early"]);
prismt.plot.conditionTraces(nexttile(t), ds, "stim", Channels=truth.StimulusChannels, Modality="calcium");
prismt.plot.metadataCrosstab(nexttile(t), ds, "phase", "mouse");
exportgraphics(f, fullfile(figFolder, '01_data_overview.png'), 'Resolution', 110);
close(f);

%% 3. Classify early vs late learning, testing on mice the model never saw
cfg = prismt.defaultConfig("classify", dataFile, Label="phase", RunsFolder=fullfile(outFolder, 'runs'));
check = prismt.check(cfg);                        % what will happen, before training
fprintf('%d trials, split: %s on %s (%d folds)\n', check.selection.n_trials, check.split.scheme, ...
    check.split.test_on, check.split.n_folds);
run = prismt.train(cfg, Name="phase", Wait=true);
R = prismt.loadResults(run.RunDir);

f = figure('Visible', 'off', 'Position', [100 100 1200 800]);
t = tiledlayout(f, 2, 2, 'TileSpacing', 'compact');
prismt.plot.learningCurves(nexttile(t), R);
prismt.plot.confusion(nexttile(t), R);
prismt.plot.scoreVsBaselines(nexttile(t), R);
prismt.plot.accuracyByGroup(nexttile(t), R, "subject");
exportgraphics(f, fullfile(figFolder, '02_classification.png'), 'Resolution', 110);
close(f);

%% 4. Masked autoencoder: learn the structure without labels
cfgMae = prismt.defaultConfig("mae", dataFile, RunsFolder=fullfile(outFolder, 'runs'));
runMae = prismt.train(cfgMae, Name="mae", Wait=true);
Rm = prismt.loadResults(runMae.RunDir);

f = figure('Visible', 'off', 'Position', [100 100 1400 700]);
t = tiledlayout(f, 2, 4, 'TileSpacing', 'compact');
axs = arrayfun(@(k) nexttile(t), 1:4);
prismt.plot.reconExample(axs, Rm, Example=1, Modality="ach");
prismt.plot.scoreVsBaselines(nexttile(t, [1 2]), Rm);
prismt.plot.r2Channels(nexttile(t), Rm, Modality="calcium", X=ds.ChannelX, Y=ds.ChannelY);
prismt.plot.r2ByCondition(nexttile(t), Rm, ds, "phase", Mask="modality_ach", Modality="ach");
exportgraphics(f, fullfile(figFolder, '03_masked_autoencoder.png'), 'Resolution', 110);
close(f);

%% 5. Fine-tune a classifier from the autoencoder (same split, so no leakage)
cfgFt = cfg;
cfgFt.model = struct('init_from', char(runMae.RunDir));
runFt = prismt.train(cfgFt, Name="finetune", Wait=true);
Rf = prismt.loadResults(runFt.RunDir);
f = figure('Visible', 'off', 'Position', [100 100 1000 420]);
t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact');
prismt.plot.scoreVsBaselines(nexttile(t), Rf);
prismt.plot.embedding(nexttile(t), Rf, ds.Trials.phase);
exportgraphics(f, fullfile(figFolder, '04_finetune.png'), 'Resolution', 110);
close(f);

%% 6. Summary
fprintf('\nClassification: %s\nAutoencoder:    %s\nFine-tuned:     %s\n', R.Metrics.summary_lines{1}, ...
    Rm.Metrics.summary_lines{1}, Rf.Metrics.summary_lines{1});
fprintf('Figures are in %s\n', figFolder);
