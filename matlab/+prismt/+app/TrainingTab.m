classdef TrainingTab < handle
    %TRAININGTAB Model size and training budget: a preset, an estimate, and (optionally)
    %every setting, generated from PRISMT's settings list so each has a plain explanation.

    properties (SetAccess = private)
        App
        Empty
        Content
        Presets struct = struct()
        PresetNote
        Estimate
        Basic struct = struct()
        Tune struct = struct()
        ShowAdvanced
        Advanced
        Fields struct = struct()     % path key -> control
        AdvancedHeight double = 0
    end
    properties (Constant)
        AdvancedSections = ["preprocess", "split", "model", "mae", "train", "baselines", "output", "selection", "labels"]
        Skip = ["model.init_from", "output.root", "selection.filters", "labels.classes", "labels.column", ...
            "mae.mask.strategy", "mae.mask.ratio", "mae.mask.context_fraction", "mae.mask.modality", "split.test_on"]
    end

    methods
        function t = TrainingTab(app, parent)
            t.App = app;
            schema = app.Controller.Schema;
            outer = uigridlayout(parent, [1 1], 'Padding', 0);
            outer.BackgroundColor = prismt.plot.style().surface;
            t.Empty = prismt.app.ui.text(outer, "Load data first (Data tab).");
            t.Empty.HorizontalAlignment = 'center'; t.Empty.VerticalAlignment = 'center';
            t.Content = prismt.app.ui.grid(outer, {26, 44, 18, 140, 96, 22, 400}, {'1x', '1x'}, ...
                'Scrollable', 'on');
            t.Content.Layout.Row = 1; t.Content.Layout.Column = 1;

            h = prismt.app.ui.heading(t.Content, "Model and training");
            h.Layout.Column = [1 2];
            cards = uigridlayout(t.Content, [1 3], 'Padding', 0, 'ColumnSpacing', 8, 'RowHeight', {44});
            cards.BackgroundColor = t.Content.BackgroundColor;
            cards.Layout.Column = [1 2];
            presets = schema.presets;
            if iscell(presets), presets = [presets{:}]; end
            for k = 1:numel(presets)
                name = string(presets(k).name);
                t.Presets.(name) = uibutton(cards, 'state', 'Text', presets(k).label, 'FontWeight', 'bold', ...
                    'Tooltip', presets(k).description, ...
                    'ValueChangedFcn', @(~, ~) app.safely(@() app.Controller.setPreset(name)));
            end
            t.PresetNote = prismt.app.ui.note(t.Content, "");
            t.PresetNote.Layout.Column = [1 2];

            % basic settings and estimate
            p = prismt.app.ui.panel(t.Content, "Training");
            g = prismt.app.ui.grid(p, {24, 24, 24}, {'fit', 110, '1x'});
            t.Basic.epochs = t.fieldControl(g, fieldOf(schema, "train.epochs"));
            t.Basic.patience = t.fieldControl(g, fieldOf(schema, "train.patience"));
            t.Basic.device = t.fieldControl(g, fieldOf(schema, "train.device"));
            p = prismt.app.ui.panel(t.Content, "Size and time");
            g = prismt.app.ui.grid(p, {'1x', 30}, {'1x', 'fit'});
            t.Estimate = prismt.app.ui.text(g, "");
            prismt.app.ui.button(g, "Estimate time", @() app.safely(@t.estimate), ...
                'Tooltip', "Runs a few training steps on this computer to estimate the run time (10-60 s)");
            n = prismt.app.ui.note(g, "Training stops early when the validation score stops improving, so the " + ...
                "time is an upper bound.");
            n.Layout.Column = [1 2];

            % automatic tuning
            p = prismt.app.ui.panel(t.Content, "Tune automatically (optional)");
            p.Layout.Column = [1 2];
            g = prismt.app.ui.grid(p, {24, 20}, {'fit', 'fit', 90, 'fit', 90, '1x'});
            t.Tune.on = uicheckbox(g, 'Text', 'Try several settings and keep the best', ...
                'Tooltip', "Trains many models with different learning rates, sizes and dropout (compared on validation " + ...
                "trials only), then retrains the best one with several seeds and tests it once.", ...
                'ValueChangedFcn', @(s, ~) app.safely(@() t.tuneToggled(s.Value)));
            uilabel(g, 'Text', 'Settings to try');
            t.Tune.n = uispinner(g, 'Limits', [2 500], 'Value', 20, 'RoundFractionalValues', 'on', ...
                'ValueChangedFcn', @(s, ~) app.safely(@() app.Controller.setValue("hpo.n_trials", s.Value)));
            uilabel(g, 'Text', 'Stop after (min)');
            t.Tune.timeout = prismt.app.ui.placeholder(uieditfield(g, 'numeric', 'Limits', [0 Inf], 'Value', 0, ...
                'Tooltip', "0: no time limit", 'ValueChangedFcn', @(s, ~) app.safely(@() t.timeoutChanged(s.Value))), "0 = no limit");
            n = prismt.app.ui.note(g, "Tuning multiplies the run time by roughly the number of settings; it is " + ...
                "best done on a GPU or a cluster (Run tab > Cluster).");
            n.Layout.Row = 2; n.Layout.Column = [1 6];

            t.ShowAdvanced = uicheckbox(t.Content, 'Text', 'Show all settings', 'Value', false, ...
                'ValueChangedFcn', @(~, ~) t.refresh());
            t.ShowAdvanced.Layout.Column = [1 2];
            t.Advanced = uipanel(t.Content, 'BorderType', 'none', 'BackgroundColor', t.Content.BackgroundColor);
            t.Advanced.Layout.Column = [1 2];
            t.buildAdvanced(schema);
            app.listen('ConfigChanged', @t.refresh);
            app.listen('DataChanged', @t.refresh);
            app.listen('CheckChanged', @t.refreshEstimate);
        end

        function refresh(t)
            c = t.App.Controller;
            has = ~isempty(c.Dataset);
            t.Empty.Visible = onoff(~has);
            t.Content.Visible = onoff(has);
            if ~has, return; end
            preset = string(c.value("preset", "quick"));
            for k = string(fieldnames(t.Presets))', t.Presets.(k).Value = (k == preset); end
            [defaults, label, desc] = presetDefaults(c.Schema, preset);
            changed = userOverrides(c.Config, defaults);
            if isempty(changed)
                t.PresetNote.Text = label + ": " + desc;
            else
                t.PresetNote.Text = "Custom (based on " + label + "): you changed " + strjoin(changed, ", ") + ".";
            end
            for s = [struct2cell(t.Basic); struct2cell(t.Fields)]'
                t.showValue(s{1}, defaults);
            end
            hpo = isfield(c.Config, 'hpo');
            t.Tune.on.Value = hpo;
            t.Tune.n.Enable = onoff(hpo); t.Tune.timeout.Enable = onoff(hpo);
            if hpo
                t.Tune.n.Value = double(c.value("hpo.n_trials", 20));
                tm = c.value("hpo.timeout_minutes", 0);
                t.Tune.timeout.Value = double(tm);
            end
            t.Advanced.Visible = onoff(t.ShowAdvanced.Value);
            t.Content.RowHeight{end} = ternary(t.ShowAdvanced.Value, t.AdvancedHeight, 0);
            t.refreshEstimate();
        end

        function refreshEstimate(t)
            r = t.App.Controller.Check;
            if isempty(r) || ~isfield(r, 'model')
                t.Estimate.Text = "Press Estimate time (or Check settings) to see the model size and run time.";
                return
            end
            lines = [sprintf("%s parameters, %d tokens per trial", compact(r.model.parameters), r.model.tokens_per_trial)
                sprintf("Attention memory about %.2g GB per batch", r.model.attention_memory_gb)];
            if isfield(r, 'estimate') && ~isempty(r.estimate)
                e = r.estimate;
                lines(end + 1) = sprintf("%.2g s per epoch on %s; at most about %.2g minutes%s", ...
                    e.seconds_per_epoch, upper(string(e.device)), e.max_total_minutes, foldNote(r));
            end
            t.Estimate.Text = strjoin(lines, newline);
        end

        function estimate(t)
            h = t.App.Dialogs.busy("Timing a few training steps...");
            cleanup = onCleanup(@() delete(h));
            t.App.Controller.runCheck(true);
        end

        function setField(t, path, text)
            %SETFIELD Type text into a setting, as a user would (used by tests too).
            f = fieldOf(t.App.Controller.Schema, path);
            t.App.Controller.setValue(path, parseValue(f, text));
        end
    end

    methods (Access = private)
        function buildAdvanced(t, schema)
            fields = schema.fields;
            if iscell(fields), fields = [fields{:}]; end
            sections = schema.sections;
            if iscell(sections), sections = [sections{:}]; end
            g = prismt.app.ui.grid(t.Advanced, {'fit'}, {'1x', '1x'}, 'Padding', [0 0 0 0]);
            col = 0;
            rows = {};
            heights = [0 0];
            for sec = t.AdvancedSections
                list = fields(startsWith(string({fields.path}), sec + "."));
                list = list(~ismember(string({list.path}), [t.Skip, "train.epochs", "train.patience", "train.device"]));
                if isempty(list), continue; end
                col = mod(col, 2) + 1;
                title = string(sections(string({sections.key}) == sec).label);
                p = prismt.app.ui.panel(g, title);
                if col == 1, rows{end + 1} = 0; heights = [0 0]; end %#ok<AGROW>
                p.Layout.Row = numel(rows); p.Layout.Column = col;
                heights(col) = 44 + 28 * numel(list);
                rows{end} = max(heights);
                pg = prismt.app.ui.grid(p, repmat({24}, 1, numel(list)), {'fit', 130, '1x'}, 'RowSpacing', 4);
                for k = 1:numel(list)
                    key = keyOf(list(k).path);
                    t.Fields.(key) = t.fieldControl(pg, list(k));
                end
            end
            g.RowHeight = rows;
            t.AdvancedHeight = sum([rows{:}]) + 10 * numel(rows);
        end

        function h = fieldControl(t, parent, f)
            %FIELDCONTROL Label + control + "default" note for one setting.
            tip = string(f.help);
            lab = uilabel(parent, 'Text', string(f.label), 'Tooltip', tip);
            path = string(f.path);
            type = string(f.type);
            if type == "choice"
                h = uidropdown(parent, 'Items', "(default)", 'Tooltip', tip, ...
                    'ValueChangedFcn', @(s, ~) t.App.safely(@() t.App.Controller.setValue(path, choiceValue(s.Value))));
            elseif type == "bool"
                h = uidropdown(parent, 'Items', ["(default)", "on", "off"], 'Tooltip', tip, ...
                    'ValueChangedFcn', @(s, ~) t.App.safely(@() t.App.Controller.setValue(path, boolValue(s.Value))));
            else
                h = uieditfield(parent, 'Tooltip', tip, ...
                    'ValueChangedFcn', @(s, ~) t.App.safely(@() t.edited(s, path)));
            end
            note = prismt.app.ui.note(parent, "");
            h.UserData = struct('field', f, 'label', lab, 'note', note);
        end

        function edited(t, s, path)
            try
                t.setField(path, s.Value);
            catch err
                t.refresh();         % show the valid value again
                rethrow(err);
            end
        end

        function showValue(t, h, defaults)
            f = h.UserData.field;
            path = string(f.path);
            c = t.App.Controller;
            def = f.default;
            if isfield(defaults, keyOf(path)), def = defaults.(keyOf(path)); end
            v = c.value(path, []);
            isSet = hasPath(c.Config, path);
            type = string(f.type);
            h.UserData.note.Text = "default: " + display(def);
            h.UserData.label.FontWeight = ternary(isSet, 'bold', 'normal');
            if type == "choice"
                items = ["(default: " + display(def) + ")", string(f.choices(:)')];
                h.Items = items;
                h.ItemsData = ["", string(f.choices(:)')];
                if isSet, h.Value = string(v); else, h.Value = ""; end
            elseif type == "bool"
                h.Items = ["(default: " + display(def) + ")", "on", "off"];
                h.ItemsData = ["", "on", "off"];
                if isSet, h.Value = ternary(logical(v), "on", "off"); else, h.Value = ""; end
            else
                if isSet, h.Value = display(v); else, h.Value = ""; end
                prismt.app.ui.placeholder(h, display(def));
            end
            if isfield(h.UserData, 'field') && t.App.Controller.task() ~= "mae" && startsWith(path, "mae.")
                h.Enable = 'off';
            else
                h.Enable = 'on';
            end
        end

        function tuneToggled(t, on)
            c = t.App.Controller;
            if on
                c.setValue("hpo.n_trials", t.Tune.n.Value);
            else
                c.setValue("hpo", []);
            end
        end

        function timeoutChanged(t, v)
            if v <= 0, v = []; end
            t.App.Controller.setValue("hpo.timeout_minutes", v);
        end
    end
end

% ---- schema helpers -----------------------------------------------------------------------
function f = fieldOf(schema, path)
fields = schema.fields;
if iscell(fields), fields = [fields{:}]; end
f = fields(string({fields.path}) == path);
if isempty(f), error('prismt:app', 'Unknown setting %s.', path); end
end

function k = keyOf(path)
k = replace(string(path), ".", "__");
end

function [d, label, desc] = presetDefaults(schema, preset)
d = struct();
label = preset; desc = "";
presets = schema.presets;
if iscell(presets), presets = [presets{:}]; end
p = presets(string({presets.name}) == preset);
if isempty(p), return; end
label = string(p.label); desc = string(p.description);
vals = p.values;
if iscell(vals), vals = [vals{:}]; end
for k = 1:numel(vals), d.(keyOf(vals(k).path)) = vals(k).value; end
end

function names = userOverrides(cfg, defaults)
%USEROVERRIDES Settings the user changed that the preset also sets.
names = strings(1, 0);
for k = string(fieldnames(defaults))'
    path = replace(k, "__", ".");
    if hasPath(cfg, path), names(end + 1) = path; end %#ok<AGROW>
end
for s = ["model", "train"]
    if ~isfield(cfg, s), continue; end
    for f = string(fieldnames(cfg.(s)))'
        path = s + "." + f;
        if path ~= "model.init_from" && ~ismember(path, names), names(end + 1) = path; end %#ok<AGROW>
    end
end
end

function tf = hasPath(cfg, path)
tf = true;
node = cfg;
for p = split(string(path), ".")'
    if isstruct(node) && isfield(node, p), node = node.(p); else, tf = false; return; end
end
end

function v = parseValue(f, text)
text = strtrim(string(text));
if strlength(text) == 0, v = []; return; end
type = string(f.type);
switch type
    case {"int", "number"}
        v = str2double(text);
        if isnan(v), error('prismt:app', '%s must be a number, not "%s".', f.label, text); end
        if type == "int" && v ~= round(v), error('prismt:app', '%s must be a whole number.', f.label); end
        if isfield(f, 'min') && ~isempty(f.min) && v < f.min
            error('prismt:app', '%s must be at least %g.', f.label, f.min);
        end
        if isfield(f, 'max') && ~isempty(f.max) && v > f.max
            error('prismt:app', '%s must be at most %g.', f.label, f.max);
        end
        if isfield(f, 'choices') && ~isempty(f.choices) && ~ismember(v, f.choices)
            error('prismt:app', '%s must be one of %s.', f.label, strjoin(string(f.choices(:)'), ", "));
        end
    case "range"
        v = str2double(split(replace(text, ["[", "]", "to"], ["", "", ","]), ","))';
        if numel(v) ~= 2 || any(isnan(v)) || v(2) <= v(1)
            error('prismt:app', '%s needs two numbers, start and end in seconds (e.g. "0, 3").', f.label);
        end
    case "list"
        v = cellstr(strtrim(split(text, ",")))';
    case {"patch", "folds"}
        if any(text == ["all", "auto"]), v = char(text); return; end
        v = str2double(text);
        if isnan(v) || v < 1 || v ~= round(v)
            error('prismt:app', '%s must be a whole number or "%s".', f.label, ternary(type == "patch", "all", "auto"));
        end
    otherwise
        v = char(text);
end
end

function v = choiceValue(x)
if strlength(string(x)) == 0, v = []; else, v = char(x); end
end

function v = boolValue(x)
switch string(x)
    case "on", v = true;
    case "off", v = false;
    otherwise, v = [];
end
end

function s = display(v)
if isempty(v)
    s = "none";
elseif islogical(v)
    s = ternary(all(v), "on", "off");
elseif isnumeric(v)
    s = strjoin(string(num2str(v(:), 6))', ", ");
    s = strjoin(strtrim(split(s, ",")), ", ");
elseif iscell(v)
    s = strjoin(string(v), ", ");
else
    s = strjoin(string(v), ", ");
end
end

function s = compact(n)
if n >= 1e6, s = sprintf("%.2g M", n / 1e6); elseif n >= 1e3, s = sprintf("%.3g k", n / 1e3); else, s = string(n); end
end

function s = foldNote(r)
s = "";
if isfield(r, 'split') && isfield(r.split, 'n_folds') && r.split.n_folds > 1
    s = sprintf(" per fold (%d folds)", r.split.n_folds);
end
end

function s = onoff(tf)
if tf, s = 'on'; else, s = 'off'; end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
