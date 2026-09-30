classdef DataTab < handle
    %DATATAB Open, import or create a dataset, and look at it before training.

    properties (SetAccess = private)
        App
        Summary
        Issues
        Notes
        Previews
        Controls struct = struct()
        Axes struct = struct()
        Info            % summary / problems / notes (left column)
        Import struct = struct()   % the import options panel
        ImportFile string = ""
        ImportPreview = []         % {ds, report} of the last preview
        Channels struct = struct()
    end

    methods
        function t = DataTab(app, parent)
            t.App = app;
            g = prismt.app.ui.grid(parent, {'fit', 'fit', 'fit', '1x'}, {330, '1x'});
            h = prismt.app.ui.heading(g, "Your data");
            h.Layout.Column = [1 2];
            b = uigridlayout(g, [1 5], 'Padding', 0, 'ColumnSpacing', 6, 'RowHeight', {'fit'}, ...
                'ColumnWidth', {'fit', 'fit', 'fit', 'fit', 'fit'});
            b.BackgroundColor = g.BackgroundColor;
            b.Layout.Column = [1 2];
            prismt.app.ui.button(b, "Open PRISMT dataset...", @() app.safely(@t.openDataset), ...
                'Tooltip', "A .mat file written by prismt.writeDataset or by this tab");
            prismt.app.ui.button(b, "Import lab file...", @() app.safely(@t.importFile), ...
                'Tooltip', "tableForModeling tables, processed_data structs, numbered variables or CDKL5 recordings");
            prismt.app.ui.button(b, "From workspace...", @() app.safely(@t.fromWorkspace), ...
                'Tooltip', "A prismt.Dataset variable made with prismt.makeDataset");
            prismt.app.ui.button(b, "Add dataset...", @() app.safely(@t.addDataset), ...
                'Tooltip', "Join other PRISMT datasets to this one, e.g. recordings with different channels or signals (matched by name)");
            prismt.app.ui.button(b, "Create demo data", @() app.safely(@t.useDemo), ...
                'Tooltip', "Synthetic data with known structure, to learn the app or test a setup");

            % left: summary, problems, import notes
            holder = uigridlayout(g, [1 1], 'Padding', 0);
            holder.BackgroundColor = g.BackgroundColor;
            holder.Layout.Row = [3 4]; holder.Layout.Column = 1;
            left = uigridlayout(holder, [3 1], 'RowHeight', {'fit', 'fit', '1x'}, 'Padding', 0);
            left.BackgroundColor = g.BackgroundColor;
            left.Layout.Row = 1; left.Layout.Column = 1;
            t.Info = left;
            t.buildImport(holder);
            p = prismt.app.ui.panel(left, "Summary");
            pg = prismt.app.ui.grid(p, {'fit'}, {'1x'});
            t.Summary = prismt.app.ui.text(pg, "No data yet. Open a PRISMT dataset, import a lab file, or create demo data.");
            p = prismt.app.ui.panel(left, "Problems found");
            pg = prismt.app.ui.grid(p, {'fit'}, {'1x'});
            t.Issues = prismt.app.ui.text(pg, "");
            p = prismt.app.ui.panel(left, "How the file was read");
            pg = prismt.app.ui.grid(p, {'1x'}, {'1x'});
            t.Notes = uitextarea(pg, 'Editable', 'off', 'FontSize', 11);

            % right: previews
            t.Previews = uitabgroup(g);
            t.Previews.Layout.Row = [3 4]; t.Previews.Layout.Column = 2;
            t.buildAverage(uitab(t.Previews, 'Title', 'Average'));
            t.buildConditions(uitab(t.Previews, 'Title', 'Conditions'));
            t.buildTrial(uitab(t.Previews, 'Title', 'Single trial'));
            t.buildMap(uitab(t.Previews, 'Title', 'Channel map'));
            t.buildCrosstab(uitab(t.Previews, 'Title', 'Trial info'));
            t.buildChannels(uitab(t.Previews, 'Title', 'Channels'));
            app.listen('DataChanged', @t.refresh);
        end

        function refresh(t)
            c = t.App.Controller;
            ds = c.Dataset;
            if isempty(ds), return; end
            t.Summary.Text = summaryText(ds, c.DatasetFile);
            issues = ds.validate();
            if isempty(issues)
                prismt.app.ui.setBanner(t.Issues, "good", "None: the file follows the PRISMT format.");
            else
                lvl = string({issues.Level});
                txt = strjoin(upper(extractBefore(lvl + " ", 2)) + extractAfter(lvl, 1) + ": " + ...
                    string({issues.Message}) + " " + string({issues.Hint}), newline);
                kind = "warning"; if any(lvl == "error"), kind = "critical"; end
                prismt.app.ui.setBanner(t.Issues, kind, txt);
            end
            t.Notes.Value = c.ImportNotes;
            mods = ds.ModalityNames;
            cols = prismt.app.ui.columnChoices(ds);
            label = string(c.value("labels.column", ""));
            if isempty(label) || strlength(label) == 0, label = cols(1); end
            set([t.Controls.avgModality, t.Controls.condModality, t.Controls.trialModality, t.Controls.mapModality], ...
                'Items', mods, 'Value', mods(1));
            t.Controls.avgBy.Items = ["(all trials)", cols];
            t.Controls.avgBy.Value = "(all trials)";
            t.Controls.condBy.Items = cols;
            t.Controls.condBy.Value = label;
            t.Controls.condChannels.Items = ds.ChannelNames;
            t.Controls.condChannels.Value = ds.ChannelNames(1);
            t.Controls.trial.Limits = [1 max(ds.N, 1)];
            t.Controls.trial.Value = 1;
            t.Controls.rows.Items = cols; t.Controls.rows.Value = label;
            t.Controls.cols.Items = cols;
            t.Controls.cols.Value = cols(min(numel(cols), 1 + (cols(1) == label)));
            if strlength(ds.Subject) && ismember(ds.Subject, cols) && ds.Subject ~= label
                t.Controls.cols.Value = ds.Subject;
            end
            t.levelsChanged();
            t.refreshChannels();
            t.drawAll();
        end

        function drawAll(t)
            for f = ["drawAverage", "drawConditions", "drawTrial", "drawMap", "drawCrosstab"]
                t.App.safely(@() t.(f)());
            end
        end

        % ---- actions ------------------------------------------------------------------
        function openDataset(t)
            f = t.App.Dialogs.getFile({'*.mat', 'PRISMT dataset (*.mat)'}, "Open a PRISMT dataset", ...
                fullfile(prismt.internal.projectFolder(), "datasets"));
            if strlength(f) == 0, return; end
            h = t.App.Dialogs.busy("Reading " + f + "...");
            cleanup = onCleanup(@() delete(h));
            t.App.Controller.openDataset(f);
        end

        function importFile(t)
            f = t.App.Dialogs.getFile({'*.mat', 'MATLAB file (*.mat)'}, "Import a lab file");
            if strlength(f) == 0, return; end
            t.startImport(f);
        end

        function startImport(t, f)
            %STARTIMPORT Show the import options for a file (what it contains, how to read it).
            h = t.App.Dialogs.busy("Reading " + f + "...");
            info = t.App.Controller.inspectFile(f);
            delete(h);
            if ~isfield(info, 'signals')          % already a PRISMT dataset
                t.App.Controller.openDataset(f);
                return
            end
            t.ImportFile = string(f);
            t.ImportPreview = [];
            c = t.Import;
            c.file.Text = "File: " + f;
            c.contains.Text = sprintf("%d trials in %d sessions. Signals: %s.", info.trials, info.sessions, ...
                strjoin(info.signals' + " (" + info.signal_sizes' + ")", "; "));
            c.signals.Items = info.signals;
            c.signals.Value = info.signals(1);
            if ismember("dff", info.signals), c.signals.Value = "dff"; end
            c.behavior.Items = info.behavior;
            c.behavior.Value = {};
            c.behavior.Enable = onoff(~isempty(info.behavior));
            cols = [info.session_columns; info.trial_columns];
            c.subject.Items = ["(none)"; cols];
            c.subject.Value = "(none)";
            pick = intersect(["mouse", "subject", "animal", "participant", "patient", "rat", "monkey"], cols, 'stable');
            if ~isempty(pick), c.subject.Value = pick(1); end
            c.fs.Value = 0; c.t0.Value = ""; c.event.Value = "";
            if ~isempty(info.fs), c.fs.Value = info.fs; end
            if ~isempty(info.t0), c.t0.Value = string(info.t0); end
            c.event.Value = info.event;
            c.atlas.Value = "(none)";
            if info.atlas_hint == "grid82", c.atlas.Value = "grid82"; end
            c.layout.Value = "independent";
            c.k.Value = 2; c.names.Value = ""; c.units.Value = ""; c.kind.Value = "signal"; c.pairs.Value = false;
            c.result.Text = "Choose how to read the file, then press Preview.";
            t.layoutChanged();
            t.Info.Visible = 'off';
            c.panel.Visible = 'on';
        end

        function opts = importOptions(t)
            %IMPORTOPTIONS The import options as set in the panel (name-value cell for importData).
            c = t.Import;
            opts = {'Signals', string(c.signals.Value), 'Layout', string(c.layout.Value), ...
                'Kind', strtrim(string(c.kind.Value)), 'Atlas', erase(string(c.atlas.Value), "(none)"), ...
                'AverageHemispheres', logical(c.pairs.Value)};
            if c.layout.Value ~= "independent", opts = [opts, {'NModalities', c.k.Value}]; end
            names = splitList(c.names.Value);
            if ~isempty(names), opts = [opts, {'ModalityNames', names}]; end
            units = splitList(c.units.Value);
            if ~isempty(units), opts = [opts, {'ModalityUnits', units}]; end
            beh = string(c.behavior.Value);
            if ~isempty(beh), opts = [opts, {'Behavior', beh}]; end
            if c.fs.Value > 0, opts = [opts, {'SamplingRate', c.fs.Value}]; end
            t0 = str2double(c.t0.Value);
            if ~isnan(t0), opts = [opts, {'TimeZero', t0}]; end
            if strlength(strtrim(c.event.Value)), opts = [opts, {'Event', strtrim(string(c.event.Value))}]; end
            if c.subject.Value ~= "(none)", opts = [opts, {'Subject', string(c.subject.Value)}]; else, opts = [opts, {'Subject', ""}]; end
        end

        function previewImport(t)
            [ds, rep] = t.App.Controller.previewImport(t.ImportFile, t.importOptions());
            t.ImportPreview = {ds, rep};
            chans = arrayfun(@(m) numel(ds.ModalityChannels{m}), 1:ds.M);
            t.Import.result.Text = sprintf("Result: %d trials; signals %s; %d time points (%.3g to %.3g s).", ds.N, ...
                strjoin(ds.ModalityNames' + " (" + string(chans) + " channels, " + ds.ModalityKinds' + ")", ", "), ...
                ds.T, ds.Times(1), ds.Times(end)) + newline + strjoin("- " + string(rep.notes(:)), newline);
            prismt.app.ui.setBanner(t.Import.result, "good", t.Import.result.Text);
        end

        function saveImport(t)
            if isempty(t.ImportPreview), t.previewImport(); end
            h = t.App.Dialogs.busy("Saving the dataset...");
            cleanup = onCleanup(@() delete(h));
            t.App.Controller.acceptImport(t.ImportPreview{1}, t.ImportPreview{2}, t.ImportFile);
            t.cancelImport();
        end

        function cancelImport(t)
            t.Import.panel.Visible = 'off';
            t.Info.Visible = 'on';
            t.ImportPreview = [];
        end

        function addDataset(t)
            if isempty(t.App.Controller.Dataset)
                t.App.Dialogs.alert("Open or import a dataset first, then add others to it.", "Add dataset", "info");
                return
            end
            f = t.App.Dialogs.getFile({'*.mat', 'PRISMT dataset (*.mat)'}, "Add a PRISMT dataset", ...
                fullfile(prismt.internal.projectFolder(), "datasets"));
            if strlength(f) == 0, return; end
            h = t.App.Dialogs.busy("Combining datasets...");
            cleanup = onCleanup(@() delete(h));
            t.App.Controller.addDatasets(f);
        end

        function saveGroups(t)
            d = t.Channels.table.Data;
            t.App.Controller.setChannelGroups(string(d(:, 3)));
        end

        function fromWorkspace(t)
            vars = evalin('base', 'whos');
            names = string({vars(strcmp({vars.class}, 'prismt.Dataset')).name});
            if isempty(names)
                t.App.Dialogs.alert("There is no prismt.Dataset variable in the workspace. Make one with " + ...
                    "ds = prismt.makeDataset(X, trials, ...) (see its help), then try again.", "From workspace", "info");
                return
            end
            name = names(1);
            if numel(names) > 1
                name = t.App.Dialogs.ask("Which dataset?", "From workspace", [names, "Cancel"]);
                if name == "Cancel" || name == "", return; end
            end
            ds = evalin('base', name);
            out = fullfile(prismt.internal.projectFolder(), "datasets", name + "_prismt.mat");
            prismt.writeDataset(ds, out);
            t.App.Controller.openDataset(out);
        end

        function useDemo(t)
            h = t.App.Dialogs.busy("Creating demo data...");
            cleanup = onCleanup(@() delete(h));
            t.App.Controller.useDemo("fast");
        end
    end

    methods (Access = private)
        function buildImport(t, holder)
            p = prismt.app.ui.panel(holder, "Import a lab file");
            p.Layout.Row = 1; p.Layout.Column = 1;
            p.Visible = 'off';
            rows = {'fit', 'fit', 60, 24, 24, 24, 60, 24, 24, 24, 24, 24, 24, 24, 24, 'fit', 30};
            g = prismt.app.ui.grid(p, rows, {120, '1x'}, 'RowSpacing', 5, 'Scrollable', 'on');
            c = struct('panel', p);
            function h = at(h, row, col)
                h.Layout.Row = row; h.Layout.Column = col;
            end
            function h = pair(row, txt, tip, control)
                at(uilabel(g, 'Text', txt, 'Tooltip', tip), row, 1);
                h = at(control, row, 2);
            end
            c.file = at(prismt.app.ui.note(g, ""), 1, [1 2]);
            c.contains = at(prismt.app.ui.text(g, ""), 2, [1 2]);
            c.signals = pair(3, "Signal", "The variable(s) holding the recordings; choose several to use each as its own signal", ...
                uilistbox(g, 'Items', "-", 'Multiselect', 'on'));
            c.layout = pair(4, "Channels", "How the channels of the signal are organized", ...
                uidropdown(g, 'Items', ["Each is one channel", "Split into signals: in blocks", "Split into signals: alternating"], ...
                'ItemsData', ["independent", "blocks", "interleaved"], ...
                'Tooltip', "Blocks: channels 1..R/K are signal 1, the next R/K signal 2... Alternating: 1, K+1, ... are signal 1; 2, K+2, ... signal 2", ...
                'ValueChangedFcn', @(~, ~) t.layoutChanged()));
            c.k = pair(5, "Number of signals", "K: into how many signals the channels are split", ...
                uispinner(g, 'Limits', [2 16], 'Value', 2, 'RoundFractionalValues', 'on'));
            c.names = pair(6, "Signal names", "Comma-separated, e.g. calcium, ach (optional)", ...
                prismt.app.ui.placeholder(uieditfield(g), "optional, e.g. calcium, ach"));
            c.behavior = pair(7, "Behavior", "Per-trial time series to add as a separate signal with its own channels (e.g. running speed, pupil)", ...
                uilistbox(g, 'Items', "-", 'Multiselect', 'on'));
            c.kind = pair(8, "Kind of signal", "What the recorded signal is; any text", ...
                uidropdown(g, 'Items', ["signal", "neural", "physiology", "behavior", "stimulus", "other"], 'Editable', 'on'));
            c.units = pair(9, "Units", "Comma-separated, one per signal (optional), e.g. dF/F", ...
                prismt.app.ui.placeholder(uieditfield(g), "optional"));
            c.subject = pair(10, "Subject column", "The column that identifies each subject (animal, participant): results are tested on new subjects", ...
                uidropdown(g, 'Items', "(none)"));
            c.fs = pair(11, "Sampling rate (Hz)", "0: from the file", uieditfield(g, 'numeric', 'Limits', [0 Inf], 'Value', 0));
            c.t0 = pair(12, "First sample at (s)", "Time of the first sample relative to the event, e.g. -1.1; empty: from the file", uieditfield(g));
            c.event = pair(13, "Time 0 is", "What time 0 is, e.g. stimulus onset, movement onset, lick", ...
                prismt.app.ui.placeholder(uieditfield(g), "e.g. stimulus onset"));
            c.atlas = pair(14, "Layout / atlas", "Channel positions for maps (only for these grid/atlas layouts)", ...
                uidropdown(g, 'Items', ["(none)", "grid82", "grid41", "allen52"]));
            c.pairs = at(uicheckbox(g, 'Text', "Average channel pairs (2k-1, 2k), e.g. left and right hemispheres"), 15, [1 2]);
            c.result = at(prismt.app.ui.text(g, ""), 16, [1 2]);
            b = at(uigridlayout(g, [1 3], 'Padding', 0, 'ColumnWidth', {'fit', 'fit', 'fit'}, 'RowHeight', {'1x'}), 17, [1 2]);
            b.BackgroundColor = g.BackgroundColor;
            prismt.app.ui.button(b, "Preview", @() t.App.safely(@t.previewImport), 'Tooltip', "Read the file with these options (not saved yet)");
            prismt.app.ui.primary(b, "Save and use", @() t.App.safely(@t.saveImport));
            prismt.app.ui.button(b, "Cancel", @() t.cancelImport());
            t.Import = c;
        end

        function layoutChanged(t)
            t.Import.k.Enable = onoff(t.Import.layout.Value ~= "independent");
        end

        function buildChannels(t, tab)
            g = prismt.app.ui.grid(tab, {'fit', '1x', 'fit'}, {'1x', 'fit'});
            n = prismt.app.ui.note(g, "Every channel and the signals it has. Give channels a group (e.g. brain area, " + ...
                "left/right, sensor, body part) to train on some groups only (Task tab) or compare them. Type in the Group column.");
            prismt.app.ui.button(g, "Save groups", @() t.App.safely(@t.saveGroups));
            t.Channels.table = uitable(g, 'ColumnName', {'Channel', 'Signals', 'Group', 'Missing'}, 'RowName', {}, ...
                'ColumnEditable', [false false true false], 'ColumnWidth', {140, 'auto', 140, 80}, 'Data', cell(0, 4));
            t.Channels.table.Layout.Column = [1 2];
            t.Channels.summary = prismt.app.ui.note(g, "");
            t.Channels.summary.Layout.Column = [1 2];
            n.Layout.Column = 1;
        end

        function refreshChannels(t)
            ds = t.App.Controller.Dataset;
            sig = strings(ds.R, 1);
            for m = 1:ds.M
                cs = ds.ModalityChannels{m};
                sig(cs) = sig(cs) + ", " + ds.ModalityNames(m);
            end
            sig = strip(extractAfter(sig, 1));
            miss = strings(ds.R, 1);
            for r = 1:ds.R
                v = [];
                for m = 1:ds.M
                    if ismember(r, ds.ModalityChannels{m}), v = [v; reshape(ds.X(:, r, :, m), [], 1)]; end %#ok<AGROW>
                end
                miss(r) = sprintf("%.0f%%", 100 * mean(isnan(v)));
            end
            groups = ds.ChannelGroups;
            if isempty(groups), groups = strings(ds.R, 1); end
            t.Channels.table.Data = [cellstr(ds.ChannelNames), cellstr(sig), cellstr(groups), cellstr(miss)];
            ug = unique(groups(strlength(groups) > 0));
            t.Channels.summary.Text = ds.R + " channels, " + ds.M + " signal(s)" + ...
                ternary(isempty(ug), "; no groups yet.", "; groups: " + strjoin(ug', ", ") + ".");
        end

        % ---- previews -----------------------------------------------------------------
        function buildAverage(t, tab)
            g = prismt.app.ui.grid(tab, {'fit', '1x', 'fit'}, {'fit', 120, 'fit', 140, 'fit', 110, 'fit', 110, '1x'});
            uilabel(g, 'Text', 'Signal');
            t.Controls.avgModality = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawAverage));
            uilabel(g, 'Text', 'Compare');
            t.Controls.avgBy = uidropdown(g, 'Items', "(all trials)", 'Tooltip', "Show the difference between two groups of trials", ...
                'ValueChangedFcn', @(~, ~) t.App.safely(@t.levelsChanged));
            uilabel(g, 'Text', 'group');
            t.Controls.avgA = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawAverage));
            uilabel(g, 'Text', 'minus');
            t.Controls.avgB = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawAverage));
            t.Axes.average = uiaxes(g);
            t.Axes.average.Layout.Row = 2; t.Axes.average.Layout.Column = [1 9];
            n = prismt.app.ui.note(g, "Trial-averaged signal of every channel over time. Missing values are left out " + ...
                "of the average; channels with no data are gray.");
            n.Layout.Column = [1 9];
        end

        function levelsChanged(t)
            c = t.Controls;
            ds = t.App.Controller.Dataset;
            on = c.avgBy.Value ~= "(all trials)";
            set([c.avgA, c.avgB], 'Enable', onoff(on));
            if on
                lv = unique(prismt.app.ui.labels(ds, c.avgBy.Value), 'stable');
                lv = sort(lv(~ismissing(lv)));
                c.avgA.Items = lv; c.avgB.Items = lv;
                c.avgA.Value = lv(end); c.avgB.Value = lv(1);
            end
            t.drawAverage();
        end

        function drawAverage(t)
            ds = t.App.Controller.Dataset;
            if isempty(ds), return; end
            c = t.Controls;
            ax = t.Axes.average;
            if c.avgBy.Value == "(all trials)"
                prismt.plot.meanHeatmap(ax, ds, Modality=c.avgModality.Value);
            else
                g = prismt.app.ui.labels(ds, c.avgBy.Value);
                prismt.plot.meanHeatmap(ax, ds, Modality=c.avgModality.Value, ...
                    Difference={g == c.avgA.Value, g == c.avgB.Value}, Labels=[string(c.avgA.Value), string(c.avgB.Value)]);
            end
        end

        function buildConditions(t, tab)
            g = prismt.app.ui.grid(tab, {'fit', '1x', 'fit'}, {'fit', 120, 'fit', 140, 170, '1x'});
            uilabel(g, 'Text', 'Signal');
            t.Controls.condModality = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawConditions));
            uilabel(g, 'Text', 'One line per');
            t.Controls.condBy = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawConditions));
            t.Controls.condChannels = uilistbox(g, 'Items', "-", 'Multiselect', 'on', ...
                'Tooltip', "Channels to average (Ctrl/Cmd-click for several)", ...
                'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawConditions));
            t.Controls.condChannels.Layout.Row = [1 3]; t.Controls.condChannels.Layout.Column = 5;
            t.Axes.conditions = uiaxes(g);
            t.Axes.conditions.Layout.Row = 2; t.Axes.conditions.Layout.Column = [1 4];
            n = prismt.app.ui.note(g, "Mean ± SEM of the selected channels (averaged together). With a subject " + ...
                "column, the SEM is across subjects.");
            n.Layout.Column = [1 4];
        end

        function drawConditions(t)
            ds = t.App.Controller.Dataset;
            if isempty(ds), return; end
            c = t.Controls;
            ch = string(c.condChannels.Value);
            if isempty(ch), ch = ds.ChannelNames(1); end
            prismt.plot.conditionTraces(t.Axes.conditions, ds, c.condBy.Value, Channels=ch, Modality=c.condModality.Value);
        end

        function buildTrial(t, tab)
            g = prismt.app.ui.grid(tab, {'fit', '1x'}, {'fit', 120, 'fit', 90, '1x'});
            uilabel(g, 'Text', 'Signal');
            t.Controls.trialModality = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawTrial));
            uilabel(g, 'Text', 'Trial');
            t.Controls.trial = uispinner(g, 'Limits', [1 2], 'Value', 1, 'RoundFractionalValues', 'on', ...
                'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawTrial));
            t.Axes.trial = uiaxes(g);
            t.Axes.trial.Layout.Row = 2; t.Axes.trial.Layout.Column = [1 5];
        end

        function drawTrial(t)
            ds = t.App.Controller.Dataset;
            if isempty(ds), return; end
            k = t.Controls.trial.Value;
            m = find(ds.ModalityNames == t.Controls.trialModality.Value, 1);
            ax = t.Axes.trial;
            s = prismt.plot.style();
            cla(ax, 'reset');
            prismt.plot.prepareAxes(ax);
            [tt, tl] = prismt.plot.timeAxis(ds);
            img = squeeze(double(ds.X(k, :, :, m)));
            if isvector(img), img = reshape(img, ds.R, ds.T); end
            im = imagesc(ax, tt, 1:ds.R, img);
            im.AlphaData = double(~isnan(img));
            ax.Color = [0.85 0.85 0.85];
            colormap(ax, s.sequential);
            lim = [min(img(:)) max(img(:))];
            if all(isfinite(lim)) && lim(2) > lim(1), prismt.plot.setLimits(ax, lim); end
            cb = colorbar(ax);
            unit = ds.ModalityUnits(m);
            cb.Label.String = ds.ModalityNames(m) + ternary(strlength(unit) > 0, " (" + unit + ")", "");
            ax.YDir = 'reverse';
            ax.YTick = 1:max(1, ceil(ds.R / 20)):ds.R;
            ax.YTickLabel = ds.ChannelNames(ax.YTick);
            xlabel(ax, tl); ylabel(ax, "Channel");
            info = strings(0, 1);
            cols = string(ds.Trials.Properties.VariableNames);
            for v = cols(1:min(end, 5))
                lv = prismt.app.ui.labels(ds, v);
                info(end + 1) = v + " " + lv(k); %#ok<AGROW>
            end
            title(ax, "Trial " + k + ":  " + strjoin(info, ",  "), 'FontWeight', 'normal', 'Interpreter', 'none');
        end

        function buildMap(t, tab)
            g = prismt.app.ui.grid(tab, {'fit', '1x', 'fit'}, {'fit', 120, 'fit', 180, '1x'});
            uilabel(g, 'Text', 'Signal');
            t.Controls.mapModality = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawMap));
            uilabel(g, 'Text', 'Show');
            t.Controls.mapWhat = uidropdown(g, 'Items', ["Average signal", "Missing data (%)"], ...
                'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawMap));
            t.Axes.map = uiaxes(g);
            t.Axes.map.Layout.Row = 2; t.Axes.map.Layout.Column = [1 5];
            n = prismt.app.ui.note(g, "Drawn on the atlas or the channel positions stored in the dataset; " + ...
                "as bars when the dataset has no layout.");
            n.Layout.Column = [1 5];
        end

        function drawMap(t)
            ds = t.App.Controller.Dataset;
            if isempty(ds), return; end
            m = find(ds.ModalityNames == t.Controls.mapModality.Value, 1);
            X = double(ds.X(:, :, :, m));
            if t.Controls.mapWhat.Value == "Average signal"
                v = squeeze(mean(mean(X, 3, 'omitnan'), 1, 'omitnan'));
                what = "Average " + ds.ModalityNames(m);
            else
                v = squeeze(100 * mean(mean(isnan(X), 3), 1));
                what = "Missing " + ds.ModalityNames(m) + " (% of values)";
            end
            args = mapArgs(ds, what);
            prismt.plot.channelMap(t.Axes.map, v(:), args{:});
        end

        function buildCrosstab(t, tab)
            g = prismt.app.ui.grid(tab, {'fit', '1x', 'fit'}, {'fit', 140, 'fit', 140, '1x'});
            uilabel(g, 'Text', 'Rows');
            t.Controls.rows = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawCrosstab));
            uilabel(g, 'Text', 'Columns');
            t.Controls.cols = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.drawCrosstab));
            t.Axes.crosstab = uiaxes(g);
            t.Axes.crosstab.Layout.Row = 2; t.Axes.crosstab.Layout.Column = [1 5];
            n = prismt.app.ui.note(g, "Trial counts. Look for empty cells: a class missing in some subjects, or a " + ...
                "column that decides the label by itself (then the model can learn that column instead).");
            n.Layout.Column = [1 5];
        end

        function drawCrosstab(t)
            ds = t.App.Controller.Dataset;
            if isempty(ds), return; end
            prismt.plot.metadataCrosstab(t.Axes.crosstab, ds, t.Controls.rows.Value, t.Controls.cols.Value);
        end
    end
end

function args = mapArgs(ds, what)
args = {'Title', what, 'Names', ds.ChannelNames};
if strlength(ds.Atlas) && ds.R == atlasSize(ds.Atlas)
    args = [args, {'Atlas', ds.Atlas}];
elseif ~isempty(ds.ChannelX) && all(isfinite(ds.ChannelX))
    args = [args, {'X', ds.ChannelX, 'Y', ds.ChannelY}];
end
end

function n = atlasSize(name)
switch name
    case "grid82", n = 82;
    case "grid41", n = 41;
    case "allen52", n = 52;
    otherwise, n = -1;
end
end

function text = summaryText(ds, file)
parts = [ds.N + " trials", ds.R + " channels × " + ds.T + " time points"];
if ~isempty(ds.ChannelGroups) && any(strlength(ds.ChannelGroups))
    parts(end + 1) = "Channel groups: " + strjoin(unique(ds.ChannelGroups(strlength(ds.ChannelGroups) > 0))', ", ");
end
sigs = strings(ds.M, 1);
for m = 1:ds.M
    sigs(m) = ds.ModalityNames(m) + ": " + numel(ds.ModalityChannels{m}) + " ch, " + ds.ModalityKinds(m);
end
parts(end + 1) = ds.M + " signal" + ternary(ds.M > 1, "s", "") + " (" + strjoin(sigs, "; ") + ")";
if strlength(ds.Subject), parts(end + 1) = numel(unique(string(ds.Trials.(ds.Subject)))) + " subjects (" + ds.Subject + ")"; end
if strlength(ds.Session)
    key = string(ds.Trials.(ds.Session));
    if strlength(ds.Subject), key = string(ds.Trials.(ds.Subject)) + "/" + key; end
    parts(end + 1) = numel(unique(key)) + " sessions";
end
if ds.T > 1
    dt = median(diff(ds.Times));
    parts(end + 1) = sprintf("%.3g to %.3g s (%.3g Hz)", ds.Times(1), ds.Times(end), 1 / dt);
end
if strlength(ds.Event), parts(end + 1) = "time 0 = " + ds.Event; end
cols = string(ds.Trials.Properties.VariableNames);
text = strjoin(parts, newline) + newline + "Trial columns: " + strjoin(cols, ", ") + newline + "File: " + file;
end

function s = onoff(tf)
if tf, s = 'on'; else, s = 'off'; end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end

function v = splitList(text)
v = strtrim(split(string(text), ","))';
v = v(strlength(v) > 0);
end
