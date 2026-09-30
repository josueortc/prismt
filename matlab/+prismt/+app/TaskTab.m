classdef TaskTab < handle
    %TASKTAB What to learn: classify conditions, learn structure (masked autoencoder), or both.

    properties (SetAccess = private)
        App
        Empty
        Content
        Cards struct = struct()
        TaskNote
        Classes struct = struct()
        Filters struct = struct()
        Mae struct = struct()
        Testing struct = struct()
        Inputs struct = struct()
        Checks
    end
    properties (Constant)
        TaskKeys = ["classify", "mae", "finetune"]
        TaskTitles = ["Classify trial conditions", "Learn structure (masked autoencoder)", "Classify, starting from an autoencoder"]
        TaskHelp = [ ...
            "Learn to tell trial conditions apart (e.g. CS+ vs CS-, early vs late learning) from the signals, and test it on data the model never saw."
            "Hide part of each trial and learn to predict it from the rest. Needs no labels. Shows which channels, times and signals are predictable from the others."
            "Reuse a finished masked-autoencoder run as the starting point of a classifier. Helps when labelled trials are few. The autoencoder's train/test split is kept."]
    end

    methods
        function t = TaskTab(app, parent)
            t.App = app;
            outer = uigridlayout(parent, [1 1], 'Padding', 0);
            outer.BackgroundColor = prismt.plot.style().surface;
            t.Empty = prismt.app.ui.text(outer, "Load data first (Data tab). This tab then shows the trial columns to classify.");
            t.Empty.HorizontalAlignment = 'center'; t.Empty.VerticalAlignment = 'center';
            t.Content = prismt.app.ui.grid(outer, {'fit', 'fit', '1x'}, {'1x', 400});
            t.Content.Layout.Row = 1; t.Content.Layout.Column = 1;

            % task cards
            cards = uigridlayout(t.Content, [1 3], 'Padding', 0, 'ColumnSpacing', 8, 'RowHeight', {44});
            cards.BackgroundColor = t.Content.BackgroundColor;
            cards.Layout.Column = [1 2];
            for k = 1:3
                key = t.TaskKeys(k);
                t.Cards.(key) = uibutton(cards, 'state', 'Text', t.TaskTitles(k), 'FontWeight', 'bold', ...
                    'Tooltip', t.TaskHelp(k), 'ValueChangedFcn', @(~, ~) app.safely(@() app.Controller.setTask(key)));
            end
            t.TaskNote = prismt.app.ui.note(t.Content, "");
            t.TaskNote.Layout.Column = [1 2];

            left = uigridlayout(t.Content, [4 1], 'RowHeight', {300, 'fit', 'fit', 'fit'}, 'Padding', 0, 'Scrollable', 'on');
            left.BackgroundColor = t.Content.BackgroundColor;
            t.buildClasses(left);
            t.buildMae(left);
            t.buildFilters(left);
            t.buildInputs(left);
            t.buildTesting(left);
            t.Checks = prismt.app.ChecksPanel(app, t.Content);
            t.Checks.Panel.Layout.Row = 3; t.Checks.Panel.Layout.Column = 2;
            app.listen('ConfigChanged', @t.refresh);
            app.listen('DataChanged', @t.refresh);
            app.listen('CheckChanged', @t.refreshSplit);
        end

        function refresh(t)
            c = t.App.Controller;
            has = ~isempty(c.Dataset);
            t.Empty.Visible = onoff(~has);
            t.Content.Visible = onoff(has);
            if ~has, return; end
            task = c.task();
            for k = t.TaskKeys, t.Cards.(k).Value = (k == task); end
            t.TaskNote.Text = t.TaskHelp(t.TaskKeys == task);
            isMae = task == "mae";
            t.Classes.panel.Visible = onoff(~isMae);
            t.Mae.panel.Visible = onoff(isMae);
            t.refreshFilters();
            t.refreshInputs();
            if isMae, t.refreshMae(); else, t.refreshClasses(); end
            t.refreshSplit();
            t.Checks.refresh();
        end

        % ---- actions used by the widgets (and by tests) --------------------------------
        function setUsedAndNames(t, used, names)
            %SETUSEDANDNAMES Apply the class table: which values are used, and their class names.
            vals = string(t.Classes.table.Data(:, 2));
            t.App.Controller.setClasses(vals(used), names(used));
        end
    end

    methods (Access = private)
        % ---- classes ------------------------------------------------------------------
        function buildClasses(t, parent)
            p = prismt.app.ui.panel(parent, "Classes");
            t.Classes.panel = p;
            p.Layout.Row = 1;
            g = prismt.app.ui.grid(p, {'fit', '1x', 'fit', 'fit'}, {'fit', 200, '1x'});
            uilabel(g, 'Text', 'Predict');
            t.Classes.column = uidropdown(g, 'Items', "-", 'Tooltip', "The trial column whose values are the classes", ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.App.Controller.setLabel(s.Value)));
            t.Classes.count = prismt.app.ui.note(g, "");
            t.Classes.table = uitable(g, 'ColumnName', {'Use', 'Value', 'Class name', 'Trials', 'Subjects'}, ...
                'RowName', {}, 'ColumnEditable', [true false true false false], 'ColumnWidth', {45, 'auto', 'auto', 70, 70}, ...
                'Data', cell(0, 5), 'CellEditCallback', @(s, ~) t.App.safely(@() t.tableEdited(s)));
            t.Classes.table.Layout.Column = [1 3];
            n = prismt.app.ui.note(g, "Untick values to leave them out. Give two values the same class name to merge " + ...
                "them (e.g. hit and CR as 'correct').");
            n.Layout.Column = [1 3];
            t.Classes.finetune = uigridlayout(g, [1 3], 'Padding', 0, 'ColumnWidth', {'fit', '1x', 'fit'}, 'RowHeight', {'fit'});
            t.Classes.finetune.BackgroundColor = g.BackgroundColor;
            t.Classes.finetune.Layout.Column = [1 3];
            uilabel(t.Classes.finetune, 'Text', 'Start from');
            t.Classes.initFrom = uidropdown(t.Classes.finetune, 'Items', "(no finished autoencoder run)", ...
                'Tooltip', "A finished masked-autoencoder run on this dataset", ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.initFromChosen(s)));
            prismt.app.ui.button(t.Classes.finetune, "Refresh", @() t.App.safely(@t.refreshRuns));
        end

        function refreshClasses(t)
            c = t.App.Controller;
            ds = c.Dataset;
            cols = prismt.app.ui.columnChoices(ds);
            col = string(c.value("labels.column", ""));
            if strlength(col) == 0 || ~ismember(col, cols)
                t.Classes.column.Items = ["(choose)", cols];
                t.Classes.column.Value = "(choose)";
            else
                t.Classes.column.Items = cols;
                t.Classes.column.Value = col;
            end
            T = c.classCounts();
            classes = c.value("labels.classes", {});
            if isstruct(classes), classes = num2cell(classes); end
            used = true(height(T), 1);
            names = T.Value;
            if ~isempty(classes)
                used(:) = false;
                for k = 1:numel(classes)
                    hit = ismember(T.Value, string(classes{k}.values(:)));
                    used(hit) = true;
                    names(hit) = string(classes{k}.name);
                end
            end
            t.Classes.table.Data = [num2cell(used), cellstr(T.Value), cellstr(names), num2cell(T.Trials), num2cell(T.Subjects)];
            nc = numel(unique(names(used)));
            t.Classes.count.Text = nc + " classes, " + sum(T.Trials(used)) + " trials";
            fin = c.task() == "finetune";
            t.Classes.finetune.Visible = onoff(fin);
            if fin, t.refreshRuns(); end
        end

        function tableEdited(t, tbl)
            d = tbl.Data;
            names = strtrim(string(d(:, 3)));
            vals = string(d(:, 2));
            names(strlength(names) == 0) = vals(strlength(names) == 0);
            t.setUsedAndNames(logical(cell2mat(d(:, 1))), names);
        end

        function refreshRuns(t)
            c = t.App.Controller;
            T = c.finishedRuns("mae");
            cur = string(c.value("model.init_from", ""));
            if isempty(T) || height(T) == 0
                t.Classes.initFrom.Items = "(no finished autoencoder run)";
                t.Classes.initFrom.ItemsData = "";
            else
                t.Classes.initFrom.Items = ["(choose a run)"; T.Name + "   " + T.Summary];
                t.Classes.initFrom.ItemsData = [""; T.Folder];
            end
            if ismember(cur, string(t.Classes.initFrom.ItemsData)), t.Classes.initFrom.Value = cur; end
        end

        function initFromChosen(t, dd)
            v = string(dd.Value);
            if strlength(v) == 0, return; end
            t.App.Controller.setValue("model.init_from", char(v));
        end

        % ---- masked autoencoder -------------------------------------------------------
        function buildMae(t, parent)
            p = prismt.app.ui.panel(parent, "What to hide");
            t.Mae.panel = p;
            p.Layout.Row = 1;
            g = prismt.app.ui.grid(p, {24, 34, '1x'}, {'fit', 200, 150, '1x'});
            uilabel(g, 'Text', 'Hide');
            t.Mae.strategy = uidropdown(g, 'Items', ["Random values", "Whole channels", "The future (forecast)", "A whole signal"], ...
                'ItemsData', ["random", "channel", "forecast", "modality"], ...
                'Tooltip', "Which parts of each trial the model must predict while it learns", ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.App.Controller.setValue("mae.mask.strategy", char(s.Value))));
            t.Mae.amountLabel = uilabel(g, 'Text', 'Fraction hidden');
            t.Mae.amount = uislider(g, 'Limits', [0.05 0.95], 'Value', 0.9, 'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.amountChanged(s.Value)));
            uilabel(g, 'Text', 'Signal');
            t.Mae.modality = uidropdown(g, 'Items', "-", 'Tooltip', "The signal to predict from the others", ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.App.Controller.setValue("mae.mask.modality", char(s.Value))));
            t.Mae.help = prismt.app.ui.note(g, "");
            t.Mae.help.Layout.Column = [3 4];
            t.Mae.axes = uiaxes(g);
            t.Mae.axes.Layout.Row = 3; t.Mae.axes.Layout.Column = [1 4];
        end

        function amountChanged(t, v)
            c = t.App.Controller;
            v = round(v, 2);
            if string(c.value("mae.mask.strategy", "random")) == "forecast"
                c.setValue("mae.mask.context_fraction", v);
            else
                c.setValue("mae.mask.ratio", v);
            end
        end

        function refreshMae(t)
            c = t.App.Controller;
            ds = c.Dataset;
            strat = string(c.value("mae.mask.strategy", "random"));
            items = ["random", "channel", "forecast", "modality"];
            if ds.M < 2, items = items(1:3); end
            names = ["Random values", "Whole channels", "The future (forecast)", "A whole signal"];
            t.Mae.strategy.Items = names(1:numel(items));
            t.Mae.strategy.ItemsData = items;
            if ~ismember(strat, items), strat = "random"; end
            t.Mae.strategy.Value = strat;
            fore = strat == "forecast";
            modal = strat == "modality";
            t.Mae.amount.Enable = onoff(~modal);
            if fore
                t.Mae.amount.Value = double(c.value("mae.mask.context_fraction", 0.5));
                t.Mae.amountLabel.Text = sprintf('Time shown: %.0f%%', 100 * t.Mae.amount.Value);
            else
                t.Mae.amount.Value = double(c.value("mae.mask.ratio", 0.9));
                t.Mae.amountLabel.Text = sprintf('Hidden: %.0f%%', 100 * t.Mae.amount.Value);
            end
            t.Mae.modality.Items = ds.ModalityNames;
            m = string(c.value("mae.mask.modality", ds.ModalityNames(end)));
            if ~ismember(m, ds.ModalityNames), m = ds.ModalityNames(end); end
            t.Mae.modality.Value = m;
            t.Mae.modality.Enable = onoff(modal);
            switch strat
                case "random", h = "Hides this fraction of all values. 0.9 (the default) makes the task hard enough to learn structure.";
                case "channel", h = "Hides whole channels for the entire trial: how well can each channel be predicted from the others?";
                case "forecast", h = "Shows the first part of each trial and hides the rest: how predictable is what comes next?";
                otherwise, h = "Hides one signal entirely: how well is it predicted from the other signals?";
            end
            t.Mae.help.Text = h + " Every run also scores all patterns, so they can be compared.";
            t.drawMask(strat, t.Mae.amount.Value, find(ds.ModalityNames == m, 1));
        end

        function drawMask(t, strat, amount, mHidden)
            ds = t.App.Controller.Dataset;
            ax = t.Mae.axes;
            cla(ax, 'reset');
            prismt.plot.prepareAxes(ax);
            s = prismt.plot.style();
            [tt, tl] = prismt.plot.timeAxis(ds);
            keep = find(t.App.Controller.keptTrials(), 1);
            if isempty(keep), return; end
            R = ds.R; T = ds.T; M = ds.M;
            rs = RandStream('mt19937ar', 'Seed', 1);
            hidden = false(R, T, M);
            switch strat
                case "random", hidden = rand(rs, R, T, M) < amount;
                case "channel", hidden(randperm(rs, R, max(1, floor(amount * R))), :, :) = true;
                case "forecast", hidden(:, max(2, ceil(amount * T) + 1):end, :) = true;
                otherwise, hidden(:, :, mHidden) = true;
            end
            img = zeros(0, T);
            names = strings(0, 1);
            for m = 1:M
                x = squeeze(double(ds.X(keep, :, :, m)));
                x = reshape(x, R, T);
                x = (x - mean(x, 'all', 'omitnan')) ./ max(std(x, 0, 'all', 'omitnan'), eps);
                x(hidden(:, :, m)) = NaN;
                img = [img; x; nan(1, T)]; %#ok<AGROW>
                names = [names; ds.ModalityNames(m) + ": " + ds.ChannelNames; ""]; %#ok<AGROW>
            end
            im = imagesc(ax, tt, 1:size(img, 1), img);
            im.AlphaData = double(~isnan(img));
            ax.Color = [0.62 0.62 0.62];
            colormap(ax, s.diverging);
            prismt.plot.setLimits(ax, [-2.5 2.5]);
            ax.YDir = 'reverse';
            step = max(1, ceil(numel(names) / 16));
            ax.YTick = 1:step:numel(names);
            ax.YTickLabel = names(ax.YTick);
            ax.TickLabelInterpreter = 'none';
            xlabel(ax, tl);
            frac = mean(hidden, 'all');
            title(ax, sprintf("One trial as the model sees it: gray = hidden (%.0f%% of values)", 100 * frac), ...
                'FontWeight', 'normal');
        end

        % ---- filters ------------------------------------------------------------------
        function buildFilters(t, parent)
            p = prismt.app.ui.panel(parent, "Which trials");
            p.Layout.Row = 2;
            g = prismt.app.ui.grid(p, {'fit', 80, 'fit'}, {'fit', 160, '1x', 'fit'});
            uilabel(g, 'Text', 'Column');
            t.Filters.column = uidropdown(g, 'Items', "-", 'ValueChangedFcn', @(~, ~) t.App.safely(@t.filterColumnChanged));
            t.Filters.summary = prismt.app.ui.text(g, "");
            t.Filters.summary.Layout.Row = [1 2]; t.Filters.summary.Layout.Column = 3;
            b = uigridlayout(g, [2 1], 'Padding', 0, 'RowHeight', {'fit', 'fit'});
            b.BackgroundColor = g.BackgroundColor;
            b.Layout.Row = [1 2]; b.Layout.Column = 4;
            prismt.app.ui.button(b, "Keep only selected", @() t.App.safely(@t.applyFilter), ...
                'Tooltip', "Keep trials whose value in this column is one of the selected values");
            prismt.app.ui.button(b, "Remove filter", @() t.App.safely(@t.removeFilter));
            t.Filters.values = uilistbox(g, 'Items', "-", 'Multiselect', 'on');
            t.Filters.values.Layout.Row = 2; t.Filters.values.Layout.Column = [1 2];
            t.Filters.using = prismt.app.ui.note(g, "");
            t.Filters.using.Layout.Column = [1 4];
        end

        function refreshFilters(t)
            c = t.App.Controller;
            ds = c.Dataset;
            cols = prismt.app.ui.columnChoices(ds);
            cur = string(t.Filters.column.Value);
            t.Filters.column.Items = cols;
            if ismember(cur, cols), t.Filters.column.Value = cur; end
            t.filterColumnChanged();
            f = c.filters();
            if isempty(f)
                t.Filters.summary.Text = "All trials are used.";
            else
                lines = strings(numel(f), 1);
                for k = 1:numel(f)
                    lines(k) = string(f{k}.column) + " is " + strjoin(string(f{k}.values(:)'), " or ");
                end
                t.Filters.summary.Text = "Only trials where:" + newline + strjoin("  " + lines, newline);
            end
            t.Filters.using.Text = "Using " + nnz(c.keptTrials()) + " of " + ds.N + " trials.";
        end

        function filterColumnChanged(t)
            ds = t.App.Controller.Dataset;
            col = string(t.Filters.column.Value);
            lv = unique(prismt.app.ui.labels(ds, col));
            lv = lv(~ismissing(lv));
            t.Filters.values.Items = lv;
            cur = {};
            for f = t.App.Controller.filters()
                if string(f{1}.column) == col, cur = cellstr(string(f{1}.values)); end
            end
            t.Filters.values.Value = cur;
        end

        function applyFilter(t)
            v = string(t.Filters.values.Value);
            if isempty(v), return; end
            t.App.Controller.setFilter(string(t.Filters.column.Value), v);
        end

        function removeFilter(t)
            t.App.Controller.setFilter(string(t.Filters.column.Value), strings(0, 1));
        end

        % ---- channels and signals -----------------------------------------------------
        function buildInputs(t, parent)
            p = prismt.app.ui.panel(parent, "Which channels and signals");
            p.Layout.Row = 3;
            g = prismt.app.ui.grid(p, {70, 'fit'}, {'fit', 180, 'fit', 180, '1x'});
            uilabel(g, 'Text', 'Signals', 'Tooltip', "The model uses every selected signal (e.g. neural activity and behavior)");
            t.Inputs.signals = uilistbox(g, 'Items', "-", 'Multiselect', 'on', ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.inputsChanged("modalities", s)));
            t.Inputs.groupsLabel = uilabel(g, 'Text', 'Channel groups', 'Tooltip', "Use only channels of these groups (set groups on the Data tab > Channels)");
            t.Inputs.groups = uilistbox(g, 'Items', "-", 'Multiselect', 'on', ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.inputsChanged("channel_groups", s)));
            t.Inputs.using = prismt.app.ui.note(g, "");
            t.Inputs.using.Layout.Row = 2; t.Inputs.using.Layout.Column = [1 5];
        end

        function refreshInputs(t)
            c = t.App.Controller;
            ds = c.Dataset;
            t.Inputs.signals.Items = ds.ModalityNames;
            sel = string(c.value("selection.modalities", ds.ModalityNames));
            t.Inputs.signals.Value = cellstr(intersect(sel, ds.ModalityNames, 'stable'));
            groups = ds.ChannelGroups;
            ug = unique(groups(strlength(groups) > 0));
            has = ~isempty(ug);
            t.Inputs.groups.Visible = onoff(has); t.Inputs.groupsLabel.Visible = onoff(has);
            if has
                t.Inputs.groups.Items = ug;
                gsel = string(c.value("selection.channel_groups", ug));
                t.Inputs.groups.Value = cellstr(intersect(gsel, ug, 'stable'));
            end
            mods = find(ismember(ds.ModalityNames, sel));
            chans = unique(vertcat(ds.ModalityChannels{mods}));
            if has
                gs = string(c.value("selection.channel_groups", ug));
                chans = chans(ismember(groups(chans), gs));
            end
            pairs = 0;
            for m = mods(:)', pairs = pairs + nnz(ismember(ds.ModalityChannels{m}, chans)); end
            t.Inputs.using.Text = "Using " + numel(mods) + " of " + ds.M + " signals and " + numel(chans) + " of " + ds.R + ...
                " channels (" + pairs + " channel-signal pairs, each a row of tokens for the model).";
        end

        function inputsChanged(t, what, box)
            items = string(box.Items);
            v = string(box.Value);
            if isempty(v)
                t.refreshInputs();
                error('prismt:app', 'Keep at least one %s selected.', ternary(what == "modalities", "signal", "channel group"));
            end
            if numel(v) == numel(items), v = []; else, v = cellstr(v); end   % all = the default
            t.App.Controller.setValue("selection." + what, v);
        end

        % ---- testing ------------------------------------------------------------------
        function buildTesting(t, parent)
            p = prismt.app.ui.panel(parent, "How the result is tested");
            p.Layout.Row = 4;
            g = prismt.app.ui.grid(p, {'fit', 'fit'}, {'fit', 240, '1x'});
            uilabel(g, 'Text', 'Test on');
            t.Testing.testOn = uidropdown(g, 'Items', ["Automatic (recommended)", "New subjects", "New sessions", "Held-out trials"], ...
                'ItemsData', ["auto", "subject", "session", "trial"], ...
                'Tooltip', "Trials the model never trains on. New subjects (animals, participants) is the strictest and says whether the result generalizes.", ...
                'ValueChangedFcn', @(s, ~) t.App.safely(@() t.App.Controller.setValue("split.test_on", char(s.Value))));
            t.Testing.note = prismt.app.ui.note(g, "Automatic tests on new subjects when there are at least 3, else on " + ...
                "new sessions, with cross-validation so each is tested once. Held-out trials of the same sessions give optimistic results.");
            t.Testing.split = prismt.app.ui.text(g, "");
            t.Testing.split.Layout.Column = [1 3];
        end

        function refreshSplit(t)
            c = t.App.Controller;
            if isempty(c.Dataset), return; end
            t.Testing.testOn.Value = string(c.value("split.test_on", "auto"));
            fin = c.task() == "finetune";
            t.Testing.testOn.Enable = onoff(~fin);
            r = c.Check;
            if fin
                t.Testing.split.Text = "Fine-tuning keeps the autoencoder run's split, so the test trials stay unseen.";
            elseif isempty(r) || ~isfield(r, 'split') || isempty(r.split)
                t.Testing.split.Text = "Press Check settings to see exactly how the trials will be split.";
            else
                t.Testing.split.Text = splitText(r.split);
            end
        end
    end
end

function text = splitText(sp)
f = sp.folds;
if iscell(f), f = [f{:}]; end
lines = strings(0, 1);
switch string(sp.scheme)
    case "single", lines(end + 1) = "One split:";
    case "kfold", lines(end + 1) = sp.n_folds + "-fold cross-validation (every group is tested once):";
    case "loo", lines(end + 1) = "Leave one out: each of the " + sp.n_groups + " groups is tested once:";
end
for k = 1:min(numel(f), 6)
    fd = f(k);
    lines(end + 1) = "  fold " + fd.fold + ": train " + fd.train.n_trials + ", validate " + fd.val.n_trials + ...
        ", test " + fd.test.n_trials + " trials" + testWho(fd.test); %#ok<AGROW>
end
if numel(f) > 6, lines(end + 1) = "  ..."; end
text = strjoin(lines, newline);
end

function w = testWho(part)
w = "";
if isfield(part, 'subjects') && ~isempty(part.subjects) && numel(part.subjects) <= 3
    w = " (" + strjoin(string(part.subjects(:)'), ", ") + ")";
end
end

function s = onoff(tf)
if tf, s = 'on'; else, s = 'off'; end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
