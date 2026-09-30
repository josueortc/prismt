# Containers (optional)

Most users do not need a container: the app's **Create environment** installs everything
PRISMT needs, and it is the only way to use the Apple GPU on a Mac (containers cannot see
it). The container image is useful when:

- a cluster prefers Apptainer (Singularity) images to conda environments;
- you want the same dependencies on several machines without installing conda;
- you run PRISMT from the command line on a Linux server with Docker.

The image holds Python and PRISMT's dependencies only. The PRISMT code is read from your
checkout or from a job folder's `prismt_src/` (through `PYTHONPATH`), so the image only
needs rebuilding when the dependencies change, not with every PRISMT update.

## Build

```bash
docker build -f docker/Dockerfile -t prismt:cpu  --build-arg VARIANT=cpu  .
docker build -f docker/Dockerfile -t prismt:cuda --build-arg VARIANT=cuda .   # NVIDIA GPUs, CUDA 12.1
```

The image has not been published to a registry yet. Until it is, build it yourself, or ask
whoever maintains PRISMT in your lab for its address.

## On a cluster with Apptainer

In the cluster profile (Setup tab), set *Runtime* to `apptainer`. The job folder's
`setup_env.sh` then pulls the image instead of creating a conda environment, and the jobs
run inside it with the GPUs visible (`apptainer exec --nv`). The image address is
`PRISMT_IMAGE_URI` in the job folder's `job.env`; change it there if yours differs. To build
the `.sif` file yourself from a local Docker image:

```bash
apptainer build prismt.sif docker-daemon://prismt:cuda
```

and copy it into the job folder (or set `PRISMT_IMAGE` in `job.env` to its path).

## With Docker on your computer

The app launches runs with the native environment. To run the same run in Docker instead,
use the command line with the run settings the app (or `prismt.train`) writes, mounting
the project folder so that paths resolve:

```bash
docker run --rm -v "$HOME/Documents/PRISMT:$HOME/Documents/PRISMT" -v "$PWD/src:/opt/prismt/src" \
    --user "$(id -u):$(id -g)" prismt:cpu \
    train --config "$HOME/Documents/PRISMT/runs/<run>/run.json" --run-dir "$HOME/Documents/PRISMT/runs/<run>"
```

(add `--gpus all` with the `cuda` image on a Linux machine with an NVIDIA GPU). The run
folder then fills exactly as with a local run, and the Results tab opens it as usual.
`docker run --rm prismt:cpu doctor` checks the image.
