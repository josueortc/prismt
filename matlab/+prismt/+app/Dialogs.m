classdef Dialogs < handle
    %DIALOGS File pickers and questions used by the app, replaceable for tests.
    %   Queue answers with d.Answers = {...}: each call then takes the next answer instead of
    %   opening a window ("" or 0 means the user cancelled). Messages shown are kept in d.Shown.

    properties
        Answers cell = {}
        Figure = []
        Headless logical = false    % never open a window; unanswered questions count as cancelled
    end
    properties (SetAccess = private)
        Shown string = strings(0, 1)
    end

    methods
        function file = getFile(d, filter, title, folder)
            %GETFILE Full path of a file to open ("" if cancelled).
            if nargin < 4, folder = ""; end
            if d.scripted(), file = string(d.next()); return; end
            [f, p] = uigetfile(filter, title, char(folder));
            d.focus();
            if isequal(f, 0), file = ""; else, file = string(fullfile(p, f)); end
        end

        function file = putFile(d, filter, title, suggestion)
            if d.scripted(), file = string(d.next()); return; end
            [f, p] = uiputfile(filter, title, char(suggestion));
            d.focus();
            if isequal(f, 0), file = ""; else, file = string(fullfile(p, f)); end
        end

        function folder = getFolder(d, title, start)
            if nargin < 3, start = ""; end
            if d.scripted(), folder = string(d.next()); return; end
            f = uigetdir(char(start), title);
            d.focus();
            if isequal(f, 0), folder = ""; else, folder = string(f); end
        end

        function choice = ask(d, message, title, options, default)
            %ASK One of options (the last one is the cancel choice).
            if nargin < 5, default = options(1); end
            d.Shown(end + 1) = title + ": " + message;
            if d.scripted(), choice = string(d.next()); return; end
            choice = string(uiconfirm(d.Figure, message, title, 'Options', cellstr(options), ...
                'DefaultOption', char(default), 'CancelOption', char(options(end))));
        end

        function alert(d, message, title, icon)
            if nargin < 4, icon = "error"; end
            d.Shown(end + 1) = title + ": " + message;
            if d.scripted() || isempty(d.Figure) || ~strcmp(d.Figure.Visible, 'on'), return; end
            uialert(d.Figure, message, title, 'Icon', char(icon));
        end

        function h = busy(d, message)
            %BUSY An indeterminate progress bar; delete(h) to close it.
            h = [];
            if isempty(d.Figure) || ~strcmp(d.Figure.Visible, 'on'), return; end
            h = uiprogressdlg(d.Figure, 'Message', char(message), 'Indeterminate', 'on', 'Title', 'PRISMT');
        end
    end

    methods (Access = private)
        function tf = scripted(d)
            tf = ~isempty(d.Answers) || d.Headless;
        end

        function a = next(d)
            a = "";
            if isempty(d.Answers), return; end
            a = d.Answers{1};
            d.Answers(1) = [];
        end

        function focus(d)
            if ~isempty(d.Figure) && isvalid(d.Figure), figure(d.Figure); end
        end
    end
end
