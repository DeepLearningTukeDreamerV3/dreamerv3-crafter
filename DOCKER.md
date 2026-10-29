# Docker Compose

`compose.yaml` wraps the `Dockerfile` (adapted from upstream) and
`upstream/entrypoint.sh`. Changes to the upstream `Dockerfile`: DMLab
install disabled (see below), paths point to `upstream/`.

| Service     | Profile | Purpose                                   |
|-------------|---------|-------------------------------------------|
| `train`     | `gpu`   | Training on NVIDIA GPU                    |
| `train-cpu` | `cpu`   | Training on CPU (slow, for debugging)     |
| `viewer`    | none    | Scope web viewer at http://localhost:8000 |

## Setup

```sh
./setup.sh             # or do it by hand: cp .env.example .env
docker compose build   # first build is long: the image includes every env
```

GPU support needs the NVIDIA Container Toolkit. Fedora repositories do not
ship it, so add the NVIDIA repository first:

```sh
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Usage

```sh
# CPU smoke test.
docker compose --profile cpu up train-cpu

# GPU training with defaults from .env.
docker compose --profile gpu up train

# One-off run with custom flags.
docker compose --profile gpu run --rm train \
  python upstream/dreamerv3/main.py --logdir /logdir/pong --configs atari --task atari_pong

# Check that JAX sees the GPU.
docker compose --profile gpu run --rm train \
  python -c "import jax; print(jax.devices())"

# Results viewer.
docker compose up viewer
```

Logs go to `./logdir` on the host. To resume a run, point `--logdir` to the
same directory again.

The whole repo is bind-mounted at `/app`, so edits apply without a
rebuild. Changes to `upstream/requirements.txt` or the `Dockerfile` need
`docker compose build`.

### Small GPUs (4 GB)

- Use a smaller model: `CONFIGS="crafter size12m"`.
- Lower `XLA_MEM_FRACTION` if other processes use the GPU.
- Add `--batch_size 8` if you get out-of-memory errors.

## HPC (Slurm + Apptainer)

The PERUN cluster runs Apptainer instead of Docker. Convert the same image
with Apptainer inside Docker (no Apptainer install needed on the laptop):

```sh
mkdir -p ~/hpc-build && cd ~/hpc-build
docker save dreamerv3:latest -o dreamerv3.tar
docker run --rm --privileged -v ~/hpc-build:/work --entrypoint apptainer \
  ghcr.io/apptainer/apptainer:1.3.6 \
  build /work/dreamerv3.sif docker-archive:///work/dreamerv3.tar
```

Upload, run and the login-node rule: [`docs/hpc.md`](docs/hpc.md). The
SLURM scripts call `apptainer exec` for you (`slurm/common.sh`).

## Known quirks of the upstream files

- DMLab no longer builds, so the `Dockerfile` skips it and `dmlab_*` tasks
  are unavailable. DMLab's `WORKSPACE` fetches the latest abseil, googletest,
  and skylib commits, which now require Bazel module rules
  (`cannot load '@@rules_cc//cc:cc_test.bzl'`). Pinning Bazel 7 does not help.
- The `Dockerfile` installs `jax[cuda]==0.5.0`, then `requirements.txt`
  replaces it with `jax[cuda12]==0.4.33`. The final image works; the build is
  slower.
- `entrypoint.sh` queries the GCP metadata server and prints `NA` outside GCP.
  This is harmless.
- `upstream/README.md` (the original README) uses `python -m scope.viewer`. Recent `scope` releases ship the
  viewer as the `scope` command, which the `viewer` service uses.
- Consider adding `logdir/` and `.env` to `.gitignore`.
