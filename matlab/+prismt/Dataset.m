classdef Dataset
    %DATASET Trials x channels x time x modalities, with channel, time and trial metadata.
    %
    %   A Dataset is what PRISMT trains on. Create one from your own arrays with
    %   prismt.makeDataset, convert other files with prismt.importData, or make demo data
    %   with prismt.demo.makeSyntheticDataset. Save it with prismt.writeDataset and read it
    %   back with prismt.loadDataset.
    %
    %   NaN in X means "missing": a channel outside the imaging window, a dropped frame, or
    %   padding when one modality has fewer channels than another. PRISMT never uses
    %   missing values as inputs or scores predictions on them.
    %
    %   Main properties
    %     X                 single, [trials x channels x time x modalities]
    %     ChannelNames      one name per channel
    %     ChannelX/ChannelY optional channel positions, used to draw maps
    %     Atlas, AtlasImage optional atlas name and label image (pixel value k = channel k)
    %     ModalityNames     e.g. ["calcium"; "ach"] or ["emg"; "pupil"]; also ModalityUnits, ModalityKinds
    %     ModalityChannels  which channels exist for each modality (1-based, cell array)
    %     Times             time of each sample in seconds; Event says what time 0 is
    %     Trials            table with one row per trial (mouse, session, phase, stim, ...)
    %     ValueLabels       table (Column, Value, Label), e.g. stim 0 = "CS-"
    %     Subject, Session  names of the Trials columns that identify subjects (animals, participants) and sessions
    %
    %   Methods
    %     issues = validate(ds)   list every problem (struct array: Level, Code, Message, Hint, Field)
    %     ds2 = subset(ds, idx)   keep some trials
    %     txt = summary(ds)       one-paragraph description

    properties
        X single = zeros(0, 0, 0, 1, 'single')
        ChannelNames (:, 1) string = strings(0, 1)
        ChannelX (:, 1) double = zeros(0, 1)
        ChannelY (:, 1) double = zeros(0, 1)
        Hemisphere (:, 1) string = strings(0, 1)
        ChannelGroups (:, 1) string = strings(0, 1)   % e.g. "left"/"right", brain area, sensor, body part ("" = none)
        Atlas (1, 1) string = ""
        AtlasImage uint16 = uint16([])
        ModalityNames (:, 1) string = strings(0, 1)
        ModalityUnits (:, 1) string = strings(0, 1)
        ModalityKinds (:, 1) string = strings(0, 1)
        ModalityChannels (:, 1) cell = cell(0, 1)
        Times (:, 1) double = zeros(0, 1)
        Event (1, 1) string = ""
        Trials table = table()
        ValueLabels table = table(strings(0, 1), zeros(0, 1), strings(0, 1), ...
            'VariableNames', {'Column', 'Value', 'Label'})
        Subject (1, 1) string = ""
        Session (1, 1) string = ""
        TrialUid (:, 1) string = strings(0, 1)
        Provenance struct = struct()
        Uid (1, 1) string = ""
    end

    properties (Dependent)
        N  % number of trials
        R  % number of channels
        T  % number of time points
        M  % number of modalities
    end

    methods
        function n = get.N(ds), n = size(ds.X, 1); end
        function n = get.R(ds), n = size(ds.X, 2); end
        function n = get.T(ds), n = size(ds.X, 3); end
        function n = get.M(ds), n = size(ds.X, 4); end

        function issues = validate(ds)
            %VALIDATE List every problem with the dataset (errors and warnings).
            issues = prismt.internal.newIssues();
            X = ds.X;
            if ndims(X) > 4
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_SHAPE", ...
                    sprintf("X must be trials x channels x time x modalities (at most 4 dimensions), found %d.", ndims(X)));
                return
            end
            [N, R, T, M] = size(X);
            if N < 1
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_EMPTY", "The dataset has no trials.");
            end
            k = find(isinf(X), 1);
            if ~isempty(k)
                [n, r, t, m] = ind2sub([N R T M], k);
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_INF", ...
                    sprintf("X contains %d infinite value(s); the first is X(%d,%d,%d,%d).", nnz(isinf(X)), n, r, t, m), ...
                    "Replace infinite values with NaN (missing).", "X");
            end
            % Channels
            if numel(ds.ChannelNames) ~= R
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_CHANNELS", ...
                    sprintf("There are %d channel names but X has %d channels.", numel(ds.ChannelNames), R), ...
                    "Give one name per channel (the second dimension of X).", "ChannelNames");
            elseif any(strlength(strtrim(ds.ChannelNames)) == 0 | ismissing(ds.ChannelNames))
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_CHANNELS", "Every channel needs a non-empty name.", "", "ChannelNames");
            elseif numel(unique(ds.ChannelNames)) < R
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_CHANNELS", "Channel names must be unique; repeated: " + ...
                    strjoin(repeated(ds.ChannelNames), ", ") + ".", "", "ChannelNames");
            end
            for f = ["ChannelX", "ChannelY", "Hemisphere", "ChannelGroups"]
                if ~isempty(ds.(f)) && numel(ds.(f)) ~= R
                    issues = prismt.internal.addIssue(issues, "error", "E_DATA_CHANNELS", sprintf("%s must have one value per channel (%d).", f, R), "", f);
                end
            end
            % Modalities
            if numel(ds.ModalityNames) ~= M
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_MODALITIES", ...
                    sprintf("There are %d modality names but X has %d modalities.", numel(ds.ModalityNames), M), ...
                    "Give one name per modality (the fourth dimension of X), e.g. ""calcium"" or ""pupil"".", "ModalityNames");
            elseif numel(unique(ds.ModalityNames)) < M || any(strlength(ds.ModalityNames) == 0)
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_MODALITIES", "Modality names must be unique and non-empty.", "", "ModalityNames");
            end
            for f = ["ModalityUnits", "ModalityKinds"]
                if numel(ds.(f)) ~= M
                    issues = prismt.internal.addIssue(issues, "error", "E_DATA_MODALITIES", sprintf("%s must have one entry per modality (%d).", f, M), "", f);
                end
            end
            if any(strlength(strtrim(ds.ModalityKinds)) == 0 | ismissing(ds.ModalityKinds))
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_MODALITIES", "Every modality needs a kind: " + ...
                    "short text such as neural, behavior, physiology, stimulus or other.", "", "ModalityKinds");
            end
            if numel(ds.ModalityChannels) ~= M
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_MODALITIES", sprintf("ModalityChannels must list the channels of each of the %d modalities.", M), ...
                    "", "ModalityChannels");
            else
                for m = 1:M
                    cs = ds.ModalityChannels{m};
                    if isempty(cs) || ~isnumeric(cs) || any(cs ~= round(cs)) || any(cs < 1 | cs > R) || numel(unique(cs)) < numel(cs)
                        issues = prismt.internal.addIssue(issues, "error", "E_DATA_MODALITIES", sprintf( ...
                            "ModalityChannels{%d} must be distinct channel numbers between 1 and %d.", m, R), "", "ModalityChannels");
                    end
                end
            end
            % Time
            if numel(ds.Times) ~= T
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_TIME", sprintf("There are %d time stamps but X has %d time points.", numel(ds.Times), T), ...
                    "Set SamplingRate (and TimeZero) or Times in prismt.makeDataset.", "Times");
            elseif any(~isfinite(ds.Times)) || any(diff(ds.Times) <= 0)
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_TIME", "Times must be finite and strictly increasing.", "", "Times");
            end
            % Trials
            names = string(ds.Trials.Properties.VariableNames);
            if width(ds.Trials) > 0 && height(ds.Trials) ~= N
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_COLUMNS", sprintf("The Trials table has %d rows but X has %d trials.", height(ds.Trials), N), ...
                    "Each row of Trials describes one trial (the first dimension of X).", "Trials");
            end
            for c = 1:width(ds.Trials)
                col = ds.Trials.(c);
                if ~(isnumeric(col) || islogical(col) || iscategorical(col) || isstring(col) || iscellstr(col))
                    issues = prismt.internal.addIssue(issues, "error", "E_DATA_COLUMNS", "Trial column """ + names(c) + """ has type " + class(col) + ...
                        "; use numbers, true/false, text or categorical.", "", "Trials");
                elseif size(col, 2) ~= 1
                    issues = prismt.internal.addIssue(issues, "error", "E_DATA_COLUMNS", "Trial column """ + names(c) + """ has " + size(col, 2) + ...
                        " values per trial.", "Per-trial time series belong in X as another modality.", "Trials");
                elseif isnumeric(col) && ~isreal(col)
                    issues = prismt.internal.addIssue(issues, "error", "E_DATA_COLUMNS", "Trial column """ + names(c) + """ is complex.", "", "Trials");
                end
            end
            for role = ["Subject", "Session"]
                ref = ds.(role);
                if strlength(ref) > 0 && ~ismember(ref, names)
                    issues = prismt.internal.addIssue(issues, "error", "E_DATA_ROLES", role + " refers to a Trials column """ + ref + """ that does not exist.", ...
                        "Available columns: " + strjoin(names, ", "), role);
                end
            end
            if strlength(ds.Subject) == 0
                issues = prismt.internal.addIssue(issues, "warning", "W_DATA_NO_SUBJECT", ...
                    "No column is marked as the subject (animal, participant...), so PRISMT cannot keep subjects separate between training and testing.", ...
                    "Set Subject to the Trials column that identifies each subject.", "Subject");
            end
            if ~isempty(ds.TrialUid) && (numel(ds.TrialUid) ~= N || numel(unique(ds.TrialUid)) < N)
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_COLUMNS", "TrialUid must have one unique value per trial.", "", "TrialUid");
            end
            for k = 1:height(ds.ValueLabels)
                col = ds.ValueLabels.Column(k);
                if ~ismember(col, names) || ~isnumeric(ds.Trials.(col))
                    issues = prismt.internal.addIssue(issues, "error", "E_DATA_COLUMNS", "Value labels refer to """ + col + ...
                        """, which is not a numeric Trials column.", "", "ValueLabels");
                    break
                end
            end
            if ~isempty(ds.AtlasImage) && (~ismatrix(ds.AtlasImage) || max(ds.AtlasImage(:)) > R)
                issues = prismt.internal.addIssue(issues, "error", "E_DATA_CHANNELS", "AtlasImage must be a 2-D label image with values 0..R (0 = background).", "", "AtlasImage");
            end
            if N >= 1 && numel(ds.ModalityNames) == M
                for m = 1:M
                    if ~any(isfinite(X(:, :, :, m)), 'all')
                        issues = prismt.internal.addIssue(issues, "error", "E_DATA_EMPTY_MODALITY", "Modality """ + ds.ModalityNames(m) + """ contains no values (all NaN).", "", "X");
                    end
                end
            end
        end

        function ds = subset(ds, idx)
            %SUBSET Keep the trials in idx (logical mask or indices); metadata follows.
            ds.X = ds.X(idx, :, :, :);
            if width(ds.Trials) > 0
                ds.Trials = ds.Trials(idx, :);
            end
            if ~isempty(ds.TrialUid)
                ds.TrialUid = ds.TrialUid(idx);
            end
        end

        function txt = summary(ds)
            %SUMMARY One-paragraph, plain-language description.
            parts = sprintf("%d trials · %d channels × %d time points × %d %s", ds.N, ds.R, ds.T, ds.M, ...
                plural(ds.M, "modality", "modalities"));
            if ds.M > 0 && numel(ds.ModalityNames) == ds.M
                parts = parts + " (" + strjoin(ds.ModalityNames, ", ") + ")";
            end
            if strlength(ds.Subject) > 0 && ismember(ds.Subject, string(ds.Trials.Properties.VariableNames))
                parts = parts + sprintf(" · %d %s", numel(unique(string(ds.Trials.(ds.Subject)))), ...
                    plural(numel(unique(string(ds.Trials.(ds.Subject)))), "subject", "subjects"));
            end
            if numel(ds.Times) > 1
                parts = parts + sprintf(" · %.3g to %.3g s", ds.Times(1), ds.Times(end));
            end
            txt = parts;
        end

        function disp(ds)
            fprintf('  prismt.Dataset: %s\n', ds.summary());
            if width(ds.Trials) > 0
                fprintf('  Trial columns: %s\n', strjoin(string(ds.Trials.Properties.VariableNames), ', '));
            end
        end
    end
end

function r = repeated(s)
[u, ~, j] = unique(s);
r = u(accumarray(j, 1) > 1);
end

function w = plural(n, one, many)
if n == 1, w = one; else, w = many; end
end
