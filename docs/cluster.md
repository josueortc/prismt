# Training on a SLURM cluster

Large datasets, the Paper preset and automatic tuning are much faster on a cluster GPU.
PRISMT does not log in to the cluster for you (that would break with Duo / two-factor
login). Instead it writes a **job folder** that holds everything the cluster needs, plus the
exact commands to copy it up, submit it and fetch the results.

## Once: describe your cluster

On the **Setup** tab, fill in the *Cluster* panel and press **Save**:

| Field | Example | Notes |
|---|---|---|
| Login host | `mccleary.ycrc.yale.edu` | The address you `ssh` to. |
| User name | `abc123` | Your cluster user name (NetID). |
| Folder on the cluster | `~/prismt_jobs` | Job folders are copied here. |
| Partition | `gpu` | Ask your cluster's documentation which partition has GPUs. |
| Account, QOS | | Only if your cluster requires them. |
| GPUs | `1` | `0` runs on CPUs (no `--gpus` option is sent). |
| CPUs, Memory, Time limit | `4`, `32G`, `04:00:00` | Per job. Tuning runs one job per worker. |
| Modules to load | `miniconda` | What `module load` needs so that `conda` works. |
| Conda environment | `prismt` | Created once by `setup_env.sh`. |
| Runtime | `conda` or `apptainer` | Apptainer uses the [container image](containers.md) instead of conda. |

**Export...** saves the profile as a `.json` file that others in the lab can **Import...**.
From a script, pass the same fields as a struct:

```matlab
profile = struct('host', "mccleary.ycrc.yale.edu", 'user', "abc123", 'partition', "gpu", 'gpus', 1);
job = prismt.makeClusterJob(cfg, profile);
disp(job.Readme)
```

## Each time: make and run a job

1. Set up the run as usual (Data, Task, Model & training tabs), then on the **Run** tab
   choose **On a cluster (SLURM)**.
2. *Dataset on the cluster*: leave it empty to copy the dataset inside the job folder, or
   give its path on the cluster if it is already there (useful for large files you use
   often).
3. Press **Create job folder**. The folder appears under `cluster/` in the project folder,
   and the commands to run appear in the window. For example:

```text
1. Copy this folder to the cluster (on your computer; Windows without rsync: use scp -r):
     rsync -av "/Users/me/Documents/PRISMT/cluster/classify_20260930-144838" abc123@mccleary.ycrc.yale.edu:~/prismt_jobs/
2. Log in:
     ssh abc123@mccleary.ycrc.yale.edu
3. The first time on this cluster only (about 10 minutes, on the login node):
     bash ~/prismt_jobs/classify_20260930-144838/setup_env.sh
4. Submit:
     bash ~/prismt_jobs/classify_20260930-144838/submit.sh
5. Check progress:
     squeue --me      tail -f ~/prismt_jobs/classify_20260930-144838/logs/*.out
6. When finished, copy the results back (on your computer):
     rsync -av abc123@mccleary.ycrc.yale.edu:~/prismt_jobs/classify_20260930-144838/results/ "/Users/me/Documents/PRISMT/cluster/classify_20260930-144838/results/"
   then open the job folder in the PRISMT app (Results tab).
```

4. After copying the results back, press **I downloaded the results...** (Run tab) or
   **Open folder...** (Results tab) and choose the job folder.

Steps 1 and 6 run on **your computer**, steps 2–5 **on the cluster**.

## What is in a job folder

```text
README.txt           the commands above, filled in
job.env              resources (partition, GPUs, time...): edit here, then submit again
config.json          the run settings (paths relative to the folder, so it can be moved)
data/                the dataset, when no cluster path was given
init_from/           the autoencoder run a fine-tuning starts from, if any
prismt_src/          a copy of PRISMT's code, so the cluster runs exactly this version
setup_env.sh         creates the conda environment (or pulls the container image), once
submit.sh            submits the job(s); safe to run again
train.sbatch  hpo_worker.sbatch  hpo_final.sbatch  common.sh  summarize.sh
logs/  results/
```

- **Cross-validation**: each fold is a separate job of an array, so folds run in parallel.
  When all have finished, `bash summarize.sh` combines them; the README says so when the
  run has several folds.
- **Tuning**: `submit.sh` starts an array of tuning workers that share one study file, and a
  final job that waits for them (`afterany`), then retrains the best settings with several
  seeds and tests them. A worker that reaches its time limit is stopped, but the settings it
  finished trying are kept; running `submit.sh` again continues the same study.
- **Time limits**: 5 minutes before its time limit, a training job stops and writes its
  results from the best model so far (the log says it stopped early). It does not resume
  afterwards, so give jobs enough time: the *Estimate time* on the Model & training tab is
  an upper bound for this computer, and a cluster GPU is usually several times faster.

## When something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| `run once: bash setup_env.sh` in the log | The environment does not exist yet on this cluster. Run step 3. |
| `conda: command not found` | *Modules to load* is wrong for this cluster. Check `module avail`, edit `job.env` (`PRISMT_MODULES`), submit again. |
| Job pending for a long time (`squeue --me` shows `PD`) | The partition is busy or the request is large. A shorter time limit or fewer GPUs often starts sooner. |
| `CUDA error: no kernel image` or `driver too old` | The PyTorch build does not match the cluster's GPUs or driver. Set `PRISMT_CUDA_TAG` in `job.env` (e.g. `cu118`) and run `setup_env.sh` again. |
| Out of memory | Increase *Memory*, use a smaller batch (`train.batch_size`), or fewer tokens per trial (see [settings](hyperparameters.md#what-to-change-first)). |
| `Folds [...] have not finished yet` from `summarize.sh` | Wait for every fold job to finish, then run it again. |

The job's own log is in `logs/`, and the run's in `results/log.txt`.

## Tips

- **One Duo prompt per session.** Add this to `~/.ssh/config` on your computer, and `ssh`,
  `rsync` and `scp` then reuse one login for an hour:

```text
Host mccleary.ycrc.yale.edu
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 1h
```

- Keep large datasets on the cluster (give *Dataset on the cluster*) instead of copying them
  with every job.
- A job folder records the code version it ran (`prismt_src/snapshot_info.json`), so old
  results stay reproducible after PRISMT is updated.
