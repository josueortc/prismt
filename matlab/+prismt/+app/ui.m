classdef ui
    %UI Small builders shared by the app's tabs (consistent fonts, colors and spacing).

    methods (Static)
        function g = grid(parent, rows, cols, varargin)
            g = uigridlayout(parent, [numel(rows) numel(cols)], 'RowHeight', rows, 'ColumnWidth', cols, ...
                'Padding', [10 10 10 10], 'RowSpacing', 8, 'ColumnSpacing', 10, varargin{:});
            s = prismt.plot.style();
            g.BackgroundColor = s.surface;
        end

        function h = heading(parent, text)
            s = prismt.plot.style();
            h = uilabel(parent, 'Text', text, 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', s.ink);
        end

        function h = text(parent, text, varargin)
            s = prismt.plot.style();
            h = uilabel(parent, 'Text', text, 'WordWrap', 'on', 'FontColor', s.ink, 'VerticalAlignment', 'top', varargin{:});
        end

        function h = note(parent, text)
            s = prismt.plot.style();
            h = uilabel(parent, 'Text', text, 'WordWrap', 'on', 'FontColor', s.muted, 'FontSize', 11, ...
                'VerticalAlignment', 'top');
        end

        function h = button(parent, text, fcn, varargin)
            h = uibutton(parent, 'Text', text, 'ButtonPushedFcn', @(~, ~) fcn(), varargin{:});
        end

        function h = primary(parent, text, fcn)
            s = prismt.plot.style();
            h = uibutton(parent, 'Text', text, 'ButtonPushedFcn', @(~, ~) fcn(), 'FontWeight', 'bold', ...
                'BackgroundColor', s.categorical(1, :), 'FontColor', [1 1 1]);
        end

        function p = panel(parent, title)
            s = prismt.plot.style();
            p = uipanel(parent, 'Title', title, 'FontWeight', 'bold', 'BackgroundColor', s.surface, ...
                'ForegroundColor', s.ink, 'BorderType', 'line');
        end

        function h = placeholder(h, text)
            % 'Placeholder' exists on edit fields from R2021a on some platforms only.
            if isprop(h, 'Placeholder'), h.Placeholder = char(text); end
        end

        function setBanner(label, kind, text)
            %SETBANNER Color a status line: "good", "warning", "critical" or "info". The text
            %always starts with a word (Done, Warning, Error...) so color is never the only signal.
            s = prismt.plot.style();
            switch kind
                case "good", c = s.good;
                case "warning", c = s.warning;
                case "critical", c = s.critical;
                otherwise, c = s.muted;
            end
            label.Text = text;
            label.FontColor = c;
        end

        function openFolder(f)
            %OPENFOLDER Show a folder in Finder / Explorer / the file manager.
            f = string(f);
            if ~isfolder(f), mkdir(f); end
            if ismac, system(sprintf('open %s', prismt.internal.shellQuote(f, "posix")));
            elseif ispc, winopen(f);
            else, system(sprintf('xdg-open %s &', prismt.internal.shellQuote(f, "posix")));
            end
        end

        function items = columnChoices(ds)
            %COLUMNCHOICES Trial columns that group trials (text, categories, or few numeric values).
            items = strings(1, 0);
            if isempty(ds), return; end
            T = ds.Trials;
            for v = string(T.Properties.VariableNames)
                x = T.(v);
                if isnumeric(x) || islogical(x)
                    if numel(unique(x(~isnan(double(x))))) > 20, continue; end
                end
                items(end + 1) = v; %#ok<AGROW>
            end
        end

        function g = labels(ds, col)
            %LABELS A trial column as text, using the dataset's value labels (0 -> "CS-").
            v = ds.Trials.(col);
            g = string(v);
            if isnumeric(v) || islogical(v)
                L = ds.ValueLabels(ds.ValueLabels.Column == col, :);
                for k = 1:height(L), g(double(v) == L.Value(k)) = L.Label(k); end
            end
        end
    end
end
