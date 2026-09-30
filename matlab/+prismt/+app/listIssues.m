function items = listIssues(c)
%LISTISSUES What stands between the current settings and a run: fast checks done in MATLAB,
%then the result of the last "Check settings" (Python's prismt check).
%
%   items(k).level    "error" (blocks Start), "warning", "info" or "ok"
%   items(k).message, .hint (what to do), .field (the setting concerned, for "Go to setting")
items = struct('level', {}, 'message', {}, 'hint', {}, 'field', {});
if ~c.pythonReady()
    items = add(items, "error", "The Python environment is not set up.", ...
        "Go to Setup and press Find automatically or Create environment.", "setup");
end
if isempty(c.Dataset)
    items = add(items, "error", "No data loaded.", "Go to Data and open, import or create a dataset.", "dataset");
    return
end
task = c.task();
if task ~= "mae"
    col = string(c.value("labels.column", ""));
    if strlength(col) == 0
        items = add(items, "error", "No label chosen: which trial column should the model predict?", ...
            "Choose it under Classes on the Task tab.", "labels.column");
    else
        T = c.classCounts();
        [used, nClasses] = usedValues(c, T);
        if nClasses < 2
            items = add(items, "error", "At least two classes are needed; " + nClasses + " is selected.", ...
                "Tick two or more values in the class table (Task tab), or remove a trial filter.", "labels.classes");
        elseif any(T.Trials(used) < 10)
            small = T.Value(used & T.Trials < 10);
            items = add(items, "warning", "Very few trials for " + strjoin(small, ", ") + ".", ...
                "Results for small classes are unreliable; consider merging classes.", "labels.classes");
        end
    end
end
if task == "finetune" && strlength(string(c.value("model.init_from", ""))) == 0
    items = add(items, "error", "No autoencoder run chosen to start from.", ...
        "Choose a finished masked-autoencoder run on the Task tab, or train one first.", "model.init_from");
end
if nnz(c.keptTrials()) == 0
    items = add(items, "error", "The trial filters remove every trial.", "Remove a filter on the Task tab.", "selection.filters");
end
if any(string({items.level}) == "error"), return; end

r = c.Check;
if isempty(r)
    items = add(items, "info", "Settings not checked yet.", ...
        "Press Check settings: PRISMT then reports the classes, how trials are split for testing, and the model size.", "");
    return
end
if ~r.ok
    e = r.error;
    items = add(items, "error", string(e.message), string(e.hint), string(field(e)));
end
if isfield(r, 'warnings')
    w = r.warnings;
    if iscell(w), w = [w{:}]; end
    for k = 1:numel(w)
        lvl = string(w(k).level);
        if lvl ~= "info", lvl = "warning"; end
        items = add(items, lvl, string(w(k).message), string(w(k).hint), string(field(w(k)))); %#ok<AGROW>
    end
end
if r.ok
    items = add(items, "ok", readyText(r), "", "");
end
end

function [used, n] = usedValues(c, T)
% Values of the label column in use, and the number of classes they form (merged values
% count once).
classes = c.value("labels.classes", {});
if isstruct(classes), classes = num2cell(classes); end
if isempty(classes)
    used = true(height(T), 1);
    n = height(T);
    return
end
used = false(height(T), 1);
n = 0;
for k = 1:numel(classes)
    hit = ismember(T.Value, string(classes{k}.values(:)));
    used = used | hit;
    n = n + any(hit);
end
end

function f = field(s)
f = "";
if isfield(s, 'field') && ~isempty(s.field), f = string(s.field); end
end

function t = readyText(r)
parts = "Ready";
if isfield(r, 'selection')
    sel = r.selection;
    parts = parts + ": " + sel.n_trials + " trials";
    if isfield(sel, 'class_names') && ~isempty(sel.class_names)
        parts = parts + " in " + numel(sel.class_names) + " classes (" + ...
            strjoin(string(sel.class_names(:)') + " " + string(sel.class_counts(:)'), ", ") + ")";
    end
end
if isfield(r, 'split')
    sp = r.split;
    switch string(sp.scheme)
        case "single", parts = parts + "; tested once on held-out " + plural(sp.test_on);
        case "kfold", parts = parts + "; " + sp.n_folds + "-fold cross-validation over " + plural(sp.test_on);
        case "loo", parts = parts + "; each of the " + sp.n_groups + " " + plural(sp.test_on) + " is tested once";
        otherwise, parts = parts + "; test on " + plural(sp.test_on);
    end
end
if isfield(r, 'model')
    parts = parts + "; " + r.model.tokens_per_trial + " tokens per trial, " + ...
        sprintf("%.2g", r.model.parameters / 1e6) + " M parameters";
end
if isfield(r, 'estimate') && isfield(r.estimate, 'max_total_minutes')
    parts = parts + "; at most about " + sprintf("%.2g", r.estimate.max_total_minutes) + " min on " + upper(string(r.estimate.device));
end
t = parts + ".";
end

function w = plural(level)
switch string(level)
    case "subject", w = "subjects";
    case "session", w = "sessions";
    otherwise, w = "trials";
end
end

function items = add(items, level, message, hint, field)
items(end + 1) = struct('level', string(level), 'message', string(message), 'hint', string(hint), 'field', string(field));
end
