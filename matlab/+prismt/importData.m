function [ds, report] = importData(source, opts)
%IMPORTDATA Convert lab data files into a PRISMT dataset. Nothing is guessed silently.
%
%   [ds, report] = prismt.importData("tableForModeling_v2.mat")
%   [ds, report] = prismt.importData(file, Signal="dff", Behavior=["runSpeed","faceMotion"], ...
%                                    Layout="independent", Atlas="grid82")
%   prismt.writeDataset(ds, "tableForModeling_prismt.mat")
%
%   Understood inputs
%     - a table T with one row per session (the widefield "tableForModeling" files):
%       3-D per-session signals (e.g. dff: trials x channels x time), per-trial vectors
%       (stim, response, ...), per-session values (mouse, phase, day, ...) and, in the v2
%       files, per-trial time series (runSpeed, lickRaster, ...) and a meta struct with
%       frameRateHz, windowSeconds and *Coding value labels;
%     - processed_data / standardized_data structs (dataset_001, dataset_002, ...);
%     - numbered variables (dff_001, stim_001, ..., with mice / phases lists);
%     - CDKL5 structs with continuous recordings, cut into windows (Window, Stride).
%   Use prismt.makeDataset for arrays that are already in your workspace.
%
%   The report says which conventions were applied (report.notes), e.g. the axis order,
%   so they can be checked; call prismt.importData(file) with no options to see what the
%   file contains and what would be done (report.plan).
%
%   Options
%     Signal        which 3-D signal to use (default: dff, else the first one found)
%     Signals       several signals as modalities, e.g. ["dff_rcamp","dff_gcamp"]
%     AxisOrder     order of each per-session signal's dimensions. Default for tables:
%                   "trials,channels,time" (as in tableForModeling); for processed_data:
%                   "trials,time,channels" (as written by preprocess_matlab_table.m)
%     Layout        how channels are organized: "independent" (default), "blocks" (K
%                   modalities one after the other) or "interleaved" (alternating)
%     NModalities   K for "blocks"/"interleaved"; ModalityNames, ModalityUnits
%     AverageHemispheres  average channels 2k-1 and 2k (82 grid tiles -> 41 regions)
%     Behavior      per-trial time-series columns to add as a "behavior" modality
%     Atlas         "grid82", "grid41", "allen52" or "" (channel positions for maps)
%     Event         what time 0 is, e.g. "stimulus onset" (default: from the file, if it says)
%     Kind          what the signal is, e.g. "neural", "physiology" (default "signal");
%                   Behavior columns are always of kind "behavior"
%     ModalityNames names of the signals (default: the names of the variables in the file)
%
%   [~, info] = prismt.importData(file, Inspect=true) only reads what the file contains
%   (signals and their sizes, behavior columns, trial columns, timing) without importing;
%   the app uses it to offer the import options.
%
%   Sessions with different numbers of channels are accepted: channels are matched by name
%   when the table has a per-session list of names (a column of string arrays), otherwise by
%   position. To join recordings of different kinds, import each and use
%   prismt.combineDatasets.
%     SamplingRate, TimeZero   (default: from meta, else 10 Hz starting at 0)
%     Subject, Session         column names (default: mouse, and one session per table row)
%     Window, Stride           samples per window and step, for continuous recordings
arguments
    source
    opts.Signal (1, 1) string = ""
    opts.Signals (1, :) string = strings(1, 0)
    opts.AxisOrder (1, 1) string = ""
    opts.Layout (1, 1) string {mustBeMember(opts.Layout, ["independent", "blocks", "interleaved"])} = "independent"
    opts.NModalities (1, 1) double = 1
    opts.ModalityNames (1, :) string = strings(1, 0)
    opts.ModalityUnits (1, :) string = strings(1, 0)
    opts.AverageHemispheres (1, 1) logical = false
    opts.Behavior (1, :) string = strings(1, 0)
    opts.Atlas (1, 1) string = ""
    opts.Event (1, 1) string = ""
    opts.Kind (1, 1) string = "signal"
    opts.Inspect (1, 1) logical = false
    opts.SamplingRate double = []
    opts.TimeZero double = []
    opts.Subject (1, 1) string = "mouse"
    opts.Session (1, 1) string = ""
    opts.Window (1, 1) double = 30
    opts.Stride double = []
end
[S, file] = prismt.import.readSource(source);
kind = prismt.import.detect(S);
switch kind
    case "prismt"
        ds = prismt.loadDataset(file);
        report = struct('kind', kind, 'notes', "Already a PRISMT dataset.", 'plan', "");
        if opts.Inspect, ds = []; end
        return
    case "table"
        parts = prismt.import.fromTable(S, opts);
    case {"processed", "standardized"}
        parts = prismt.import.fromStructs(S, kind, opts);
    case "numbered"
        parts = prismt.import.fromNumbered(S, opts);
    case "cdkl5"
        parts = prismt.import.fromCdkl5(S, opts);
    otherwise
        error('prismt:E_IMPORT_UNKNOWN', ['%s: no recognizable data (it contains: %s). Build the dataset with ' ...
            'prismt.makeDataset(X, trialTable, ...) instead.'], file, strjoin(string(fieldnames(S))', ', '));
end
if opts.Inspect
    ds = [];
    report = inspectParts(kind, parts);
    return
end
[ds, notes] = prismt.import.assemble(parts, opts, file);
report = struct('kind', kind, 'notes', notes, 'plan', strjoin(notes, newline));
end

function info = inspectParts(kind, p)
sig = string(fieldnames(p.signals));
sizes = strings(numel(sig), 1);
nS = 0; N = 0;
for k = 1:numel(sig)
    parts = p.signals.(sig(k));
    nS = numel(parts);
    sz = cell2mat(cellfun(@(x) [size(x, 1) size(x, 2) size(x, 3)], parts(:), 'UniformOutput', false));
    N = sum(sz(:, 1));
    ch = unique(sz(:, 2));
    if isscalar(ch), chText = string(ch); else, chText = min(ch) + "-" + max(ch); end
    sizes(k) = chText + " channels x " + strjoin(string(unique(sz(:, 3))'), "/") + " time points";
end
info = struct('kind', kind, 'signals', sig, 'signal_sizes', sizes, 'sessions', nS, 'trials', N, ...
    'behavior', string(fieldnames(p.behavior)), 'trial_columns', string(fieldnames(p.trialCols)), ...
    'session_columns', string(fieldnames(p.sessionCols)), 'fs', p.fs, 't0', p.t0, 'event', p.event, ...
    'atlas_hint', p.atlasHint, 'channel_names', ~isempty(p.channelNames), 'notes', p.notes(:));
end
