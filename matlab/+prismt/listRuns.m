function T = listRuns(folders)
%LISTRUNS Table of runs found in one or more folders (default: the project's runs folder).
%   Columns: Name, Task, State, Started, Summary, Folder. Newest first.
if nargin == 0
    folders = fullfile(prismt.internal.projectFolder(), ["runs", "cluster"]);
end
rows = {};
for folder = string(folders)
    if ~isfolder(folder), continue; end
    items = dir(folder);
    for k = 1:numel(items)
        if ~items(k).isdir || startsWith(items(k).name, "."), continue; end
        d = fullfile(folder, string(items(k).name));
        if isfile(fullfile(d, "job.env")), d = fullfile(d, "results"); end
        s = fullfile(d, "status.json");
        if ~isfile(s), continue; end
        try, st = jsondecode(fileread(s)); catch, continue; end
        summary = "";
        mf = fullfile(d, "metrics.json");
        if isfile(mf)
            try, m = jsondecode(fileread(mf)); summary = string(m.summary_lines(1)); catch, end
        end
        task = ""; if isfield(st, 'task'), task = string(st.task); end
        started = ""; if isfield(st, 'started'), started = string(st.started); end
        rows(end + 1, :) = {string(items(k).name), task, string(st.state), started, summary, d}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'Name', 'Task', 'State', 'Started', 'Summary', 'Folder'});
if height(T), T = sortrows(T, 'Started', 'descend'); end
end
