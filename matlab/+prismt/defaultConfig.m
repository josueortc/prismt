function cfg = defaultConfig(task, dataset, opts)
%DEFAULTCONFIG Run settings to start from (only what you choose; Python fills in the rest).
%
%   cfg = prismt.defaultConfig("classify", "mydata_prismt.mat", Label="phase")
%   cfg = prismt.defaultConfig("mae", "mydata_prismt.mat")
%
%   Change fields directly, e.g. cfg.preset = "standard"; cfg.split.test_on = "subject";
%   cfg.labels.classes = {struct('name',"early",'values',{{"early"}}), ...}.
%   Every setting, its default and a plain-language explanation are in
%   prismt.config.schema() (the same list the app's tooltips come from).
arguments
    task (1, 1) string {mustBeMember(task, ["classify", "mae"])}
    dataset (1, 1) string
    opts.Label (1, 1) string = ""
    opts.Preset (1, 1) string {mustBeMember(opts.Preset, ["quick", "standard", "paper", "custom"])} = "quick"
    opts.RunsFolder (1, 1) string = ""
end
cfg = struct();
cfg.task = char(task);
cfg.preset = char(opts.Preset);
cfg.dataset = struct('path', char(absolute(dataset)));
if task == "classify"
    cfg.labels = struct('column', char(opts.Label));
end
if strlength(opts.RunsFolder)
    cfg.output = struct('root', char(absolute(opts.RunsFolder)));
end
end

function p = absolute(p)
p = string(p);
if ~startsWith(p, ["/", "~"]) && ~(ispc && ~isempty(regexp(p, '^[A-Za-z]:', 'once')))
    p = fullfile(string(pwd), p);
end
end
