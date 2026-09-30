classdef ChecksPanel < handle
    %CHECKSPANEL The "Checks" list: what is ready, what needs attention, and how to fix it.

    properties (SetAccess = private)
        App
        Panel
        Grid
        CheckButton
        Items = []
    end

    methods
        function p = ChecksPanel(app, parent)
            p.App = app;
            p.Panel = prismt.app.ui.panel(parent, "Checks");
            outer = prismt.app.ui.grid(p.Panel, {'fit', '1x'}, {'1x', 'fit'}, 'Padding', [8 8 8 8]);
            prismt.app.ui.note(outer, "Start is possible when there are no errors (✖).");
            p.CheckButton = prismt.app.ui.button(outer, "Check settings", @() app.safely(@p.runCheck), ...
                'Tooltip', "Ask PRISMT how it would run: classes, test split, model size (a few seconds)");
            p.Grid = prismt.app.ui.grid(outer, {'fit'}, {22, '1x', 'fit'}, 'Scrollable', 'on', 'Padding', [0 0 0 0]);
            p.Grid.Layout.Column = [1 2];
            app.listen('ConfigChanged', @p.refresh);
            app.listen('CheckChanged', @p.refresh);
            app.listen('EnvChanged', @p.refresh);
            app.listen('DataChanged', @p.refresh);
        end

        function runCheck(p)
            h = p.App.Dialogs.busy("Checking the settings...");
            cleanup = onCleanup(@() delete(h));
            p.App.Controller.runCheck(false);
        end

        function tf = hasErrors(p)
            tf = any(string({p.Items.level}) == "error");
        end

        function refresh(p)
            items = prismt.app.listIssues(p.App.Controller);
            p.Items = items;
            delete(p.Grid.Children);
            s = prismt.plot.style();
            p.Grid.RowHeight = repmat({'fit'}, 1, max(1, numel(items)));
            for k = 1:numel(items)
                it = items(k);
                switch it.level
                    case "error", mark = "✖"; col = s.critical;
                    case "warning", mark = "⚠"; col = s.warning;
                    case "ok", mark = "✓"; col = s.good;
                    otherwise, mark = "i"; col = s.muted;
                end
                uilabel(p.Grid, 'Text', mark, 'FontColor', col, 'FontWeight', 'bold', 'FontSize', 14, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
                txt = it.message;
                if strlength(it.hint), txt = txt + newline + it.hint; end
                uilabel(p.Grid, 'Text', txt, 'WordWrap', 'on', 'FontColor', s.ink, 'VerticalAlignment', 'top');
                if strlength(it.field)
                    f = it.field;
                    prismt.app.ui.button(p.Grid, "Go to setting", @() p.App.showField(f));
                else
                    uilabel(p.Grid, 'Text', '');
                end
            end
            c = p.App.Controller;
            p.CheckButton.Enable = onoff(c.pythonReady() && ~isempty(c.Dataset));
        end
    end
end

function s = onoff(tf)
if tf, s = 'on'; else, s = 'off'; end
end
