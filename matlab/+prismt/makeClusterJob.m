function job = makeClusterJob(cfg, profile, opts)
%MAKECLUSTERJOB Write a self-contained SLURM job folder with copy-paste instructions.
%
%   job = prismt.makeClusterJob(cfg, profile, RemoteDataset="~/prismt_data/mydata_prismt.mat")
%   disp(job.Readme)      % the exact commands: copy up, set up once, submit, copy back
%
%   profile is a struct with any of: host, user, remote_root, partition, account, qos,
%   gpus, cpus, mem ("32G"), time ("04:00:00"), modules ("miniconda"), conda_env,
%   runtime ("conda" or "apptainer"). Mode="hpo" makes a tuning job (Workers in parallel).
%   The folder also holds a snapshot of PRISMT's code, so the cluster runs this version.
arguments
    cfg struct
    profile struct = struct()
    opts.Output (1, 1) string = ""
    opts.Mode (1, 1) string {mustBeMember(opts.Mode, ["train", "hpo"])} = "train"
    opts.Workers (1, 1) double = 4
    opts.RemoteDataset (1, 1) string = ""
    opts.Name (1, 1) string = ""
    opts.Python (1, 1) string = ""
end
out = opts.Output;
if strlength(out) == 0, out = fullfile(prismt.internal.projectFolder(), "cluster"); end
tmp = string(tempname);
cfgFile = tmp + "_run.json"; profFile = tmp + "_profile.json";
cleanup = onCleanup(@() cellfun(@(f) delete(f), {char(cfgFile), char(profFile)}));
prismt.internal.atomicWrite(cfgFile, jsonencode(cfg));
prismt.internal.atomicWrite(profFile, jsonencode(profile));
args = ["jobfolder", "--config", cfgFile, "--profile", profFile, "--out", out, "--mode", opts.Mode, ...
        "--workers", string(opts.Workers)];
if strlength(opts.RemoteDataset), args = [args, "--remote-dataset", opts.RemoteDataset]; end
if strlength(opts.Name), args = [args, "--name", opts.Name]; end
[res, status, log] = prismt.run.runPython(args, Python=opts.Python);
if status ~= 0 || isempty(res)
    msg = log;
    if ~isempty(res) && isfield(res, 'error'), msg = string(res.error.message) + " " + string(res.error.hint); end
    error('prismt:E_CLUSTER_JOB', 'The cluster job could not be written: %s', msg);
end
job = struct('Folder', string(res.job_dir), 'Folds', res.n_folds, 'Readme', string(res.readme));
end
