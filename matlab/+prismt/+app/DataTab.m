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
    end

    methods
        function t = DataTab(app, parent)
            t.App = app;
            g = prismt.app.ui.grid(parent, {'fit', 'fit', 'fit', '1x'}, {330, '1x'});
            h = prismt.app.ui.heading(g, "Your data");
            h.Layout.Column = [1 2];
            b = uigridlayout(g, [1 4], 'Padding', 0, 'ColumnSpacing', 6, 'RowHeight', {'fit'}, ...
                'ColumnWidth', {'fit', 'fit', 'fit', 'fit'});
            b.BackgroundColor = g.BackgroundColor;
            b.Layout.Column = [1 2];
            prismt.app.ui.button(b, "Open PRISMT dataset...", @() app.safely(@t.openDataset), ...
                'Tooltip', "A .mat file written by prismt.writeDataset or by this tab");
            prismt.app.ui.button(b, "Import lab file...", @() app.safely(@t.importFile), ...
                'Tooltip', "tableForModeling tables, processed_data structs, numbered variables or CDKL5 recordings");
            prismt.app.ui.button(b, "From workspace...", @() app.safely(@t.fromWorkspace), ...
                'Tooltip', "A prismt.Dataset variable made with prismt.makeDataset");
            prismt.app.ui.button(b, "Create demo data", @() app.safely(@t.useDemo), ...
                'Tooltip', "Synthetic data with known structure, to learn the app or test a setup");

            % left: summary, problems, import notes
            left = uigridlayout(g, [3 1], 'RowHeight', {'fit', 'fit', '1x'}, 'Padding', 0);
            left.BackgroundColor = g.BackgroundColor;
            left.Layout.Row = [3 4]; left.Layout.Column = 1;
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
            h = t.App.Dialogs.busy("Reading " + f + " (large files take a minute)...");
            [ds, rep] = prismt.importData(f);
            delete(h);
            a = t.App.Dialogs.ask("Found " + ds.N + " trials, " + ds.R + " channels, " + ds.T + " time points, " + ...
                "signals: " + strjoin(ds.ModalityNames, ", ") + "." + newline + newline + ...
                "How the file was read:" + newline + strjoin("- " + string(rep.notes(:)), newline) + newline + newline + ...
                "Is this right? The dataset is saved in your project folder. For other choices (signals, channel " + ...
                "layout, behavior, atlas) use prismt.importData with options; see its help.", ...
                "Import", ["Save and use", "Cancel"]);
            if a ~= "Save and use", return; end
            t.App.Controller.acceptImport(ds, rep, f);
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
                "column, the SEM is across animals.");
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
            n = prismt.app.ui.note(g, "Trial counts. Look for empty cells: a class missing in some animals, or a " + ...
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
parts(end + 1) = ds.M + " signal" + ternary(ds.M > 1, "s", "") + " (" + strjoin(ds.ModalityNames, ", ") + ")";
if strlength(ds.Subject), parts(end + 1) = numel(unique(string(ds.Trials.(ds.Subject)))) + " animals"; end
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
