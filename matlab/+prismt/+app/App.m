classdef App < handle
    %APP The PRISMT window: six tabs over one AppController.
    %   Tabs are built the first time they are shown. Each tab redraws itself when the
    %   controller announces a change, so the window never holds state of its own.

    properties (SetAccess = private)
        Controller      % prismt.app.AppController
        Dialogs         % prismt.app.Dialogs
        Figure
        TabGroup
        Tabs struct = struct()
    end
    properties (Constant)
        TabNames = ["Setup", "Data", "Task", "Training", "Run", "Results"]
        TabTitles = ["1  Setup", "2  Data", "3  Task", "4  Model & training", "5  Run", "6  Results"]
    end
    properties (Access = private)
        Timer = []
        Listeners = {}
        Pages struct = struct()
    end

    methods
        function app = App(opts)
            arguments
                opts.Visible (1, 1) string = "on"
                opts.Controller = []
                opts.Dialogs = []
            end
            app.Controller = opts.Controller;
            if isempty(app.Controller), app.Controller = prismt.app.AppController(); end
            app.Dialogs = opts.Dialogs;
            if isempty(app.Dialogs), app.Dialogs = prismt.app.Dialogs(); end
            if opts.Visible == "off", app.Dialogs.Headless = true; end

            s = prismt.plot.style();
            app.Figure = uifigure('Name', "PRISMT " + prismt.version().Package, 'Visible', 'off', ...
                'Position', [100 80 1180 780], 'Color', s.surface, 'CloseRequestFcn', @(~, ~) app.close());
            if isprop(app.Figure, 'Theme'), app.Figure.Theme = 'light'; end   % the plots use the light palette
            if isprop(app.Figure, 'AutoResizeChildren'), app.Figure.AutoResizeChildren = 'off'; end
            app.Dialogs.Figure = app.Figure;
            g = uigridlayout(app.Figure, [1 1], 'Padding', [0 0 0 0]);
            app.TabGroup = uitabgroup(g, 'SelectionChangedFcn', @(~, e) app.build(string(e.NewValue.Tag)));
            for k = 1:numel(app.TabNames)
                t = uitab(app.TabGroup, 'Title', app.TabTitles(k), 'Tag', app.TabNames(k), 'BackgroundColor', s.surface);
                app.Pages.(app.TabNames(k)) = t;
            end
            app.build("Setup");
            app.Figure.Visible = char(opts.Visible);
            app.Timer = timer('ExecutionMode', 'fixedSpacing', 'Period', 1, 'BusyMode', 'drop', ...
                'Name', 'prismt-app', 'TimerFcn', @(~, ~) app.tick());
            start(app.Timer);
            if strlength(app.Controller.Python) == 0 && ~app.Dialogs.Headless
                drawnow;
                app.Tabs.Setup.findPython();
            end
        end

        function t = show(app, name)
            %SHOW Select a tab (building it if needed) and return its object.
            name = string(name);
            t = app.build(name);
            app.TabGroup.SelectedTab = app.Pages.(name);
            drawnow;
        end

        function showField(app, field)
            %SHOWFIELD Go to the tab where a setting (e.g. "labels.column") is changed.
            section = extractBefore(string(field) + ".", ".");
            switch section
                case {"setup", "python"}, app.show("Setup");
                case {"data", "dataset"}, app.show("Data");
                case {"task", "selection", "labels", "split", "mae"}, app.show("Task");
                case "model"
                    if string(field) == "model.init_from", app.show("Task"); else, app.show("Training"); end
                otherwise, app.show("Training");
            end
        end

        function file = exportFigure(app, name)
            %EXPORTFIGURE Save a picture of the whole window (used by the tests and for bug reports).
            drawnow;
            file = string(name);
            exportapp(app.Figure, file);
        end

        function close(app, force)
            if nargin < 2, force = false; end
            c = app.Controller;
            if ~force && ~isempty(c.Run) && ~isempty(c.Run.Process) && c.Run.isActive()
                a = app.Dialogs.ask("A run is still going on. It can keep running without MATLAB; " + ...
                    "reopen it later from the Results tab.", "Close PRISMT", ...
                    ["Keep it running", "Stop it", "Cancel"]);
                if a == "Cancel" || a == "", return; end
                if a == "Stop it", c.stopRun(); end
            end
            if ~isempty(app.Timer) && isvalid(app.Timer), stop(app.Timer); delete(app.Timer); end
            for k = 1:numel(app.Listeners), delete(app.Listeners{k}); end
            app.Listeners = {};
            if isvalid(app.Figure), delete(app.Figure); end
        end

        function delete(app)
            if ~isempty(app.Figure) && isvalid(app.Figure), app.close(true); end
        end

        function listen(app, event, fcn)
            %LISTEN Run fcn when the controller raises event (cleaned up when the app closes).
            app.Listeners{end + 1} = addlistener(app.Controller, event, @(~, ~) app.safely(fcn));
        end

        function safely(app, fcn, varargin)
            %SAFELY Run a callback; show an error as a message instead of a red trace.
            try
                fcn(varargin{:});
            catch err
                if app.Dialogs.Headless, rethrow(err); end
                [msg, title] = prismt.app.explainError(err);
                app.Dialogs.alert(msg, title);
            end
        end
    end

    methods (Access = private)
        function t = build(app, name)
            if isfield(app.Tabs, name)
                t = app.Tabs.(name);
                return
            end
            parent = app.Pages.(name);
            switch name
                case "Setup", t = prismt.app.SetupTab(app, parent);
                case "Data", t = prismt.app.DataTab(app, parent);
                case "Task", t = prismt.app.TaskTab(app, parent);
                case "Training", t = prismt.app.TrainingTab(app, parent);
                case "Run", t = prismt.app.RunTab(app, parent);
                case "Results", t = prismt.app.ResultsTab(app, parent);
            end
            app.Tabs.(name) = t;
            t.refresh();
        end

        function tick(app)
            if ~isvalid(app.Figure), return; end
            try
                if isfield(app.Tabs, "Setup"), app.Tabs.Setup.tick(); end
                if isfield(app.Tabs, "Run"), app.Tabs.Run.tick(); end
            catch err
                warning('prismt:app', 'Updating the app failed: %s', err.message);
            end
        end
    end
end
