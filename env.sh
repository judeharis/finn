# Jude: Created
# Source me from anywhere:  source /path/to/finn/env.sh
#
# Adds to (never replaces) FINN_DOCKER_EXTRA, so the Xilinx license mount exported
# by ~/.bashrc survives. Idempotent: re-sourcing will not duplicate a -v, which
# docker would reject as a duplicate mount point.

# This file's own directory, so the repo can live anywhere. Deliberately not named
# FINN_DIR: callers (finn-examples' build_on_host.sh) use that name themselves, and
# this file unsets its scratch variables at the end.
_FINN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# These normally come from ~/.bashrc, which a non-interactive shell (cron, ssh, an agent)
# never sources -- it returns early unless interactive. Without them run-docker.sh only
# warns and the build fails much later at HLS; and an unset FINN_DOCKER_EXTRA trips a
# caller's `set -u` below. Defaults only: anything already exported wins.
: "${FINN_XILINX_PATH:=/mnt/Crucial/Xilinx2024/}"
: "${FINN_XILINX_VERSION:=2024.1}"
: "${NUM_DEFAULT_WORKERS:=16}"
: "${FINN_DOCKER_EXTRA:=}"
export FINN_XILINX_PATH FINN_XILINX_VERSION NUM_DEFAULT_WORKERS FINN_DOCKER_EXTRA

FINN_LICENSE_MOUNT=" -v $HOME/.Xilinx/:$HOME/.Xilinx/ -e XILINXD_LICENSE_FILE=$HOME/.Xilinx/ "

case " $FINN_DOCKER_EXTRA " in
  *"$HOME/.Xilinx/:$HOME/.Xilinx/"*) ;;                    # already mounted, leave alone
  *) export FINN_DOCKER_EXTRA="${FINN_DOCKER_EXTRA}${FINN_LICENSE_MOUNT}" ;;
esac

# finn_entrypoint.sh overrides HOME=/tmp/home_dir *after* run-docker.sh has already
# expanded $HOME on the host, so its "-v $FINN_SSH_KEY_DIR:$HOME/.ssh" lands on
# /home/<user>/.ssh while ~ inside the container is /tmp/home_dir. That leaves
# ~/.ssh missing, which breaks ssh-copy-id (it mktemp's a scratch dir there).
#
# Mounting ssh_keys straight onto /tmp/home_dir/.ssh does fix ssh-copy-id, but it
# also makes docker auto-create the *parent* /tmp/home_dir owned by root. The
# container runs as uid 1000, so $HOME then isn't writable, and every one of the
# entrypoint's four `pip install --user -e` calls (qonnx, finn-experimental,
# brevitas, finn) dies with "Permission denied: '/tmp/home_dir/.local'" -- leaving
# FINN not installed at all. Symptom: the build exits 0 having done nothing.
#
# So mount a host-owned directory at /tmp/home_dir itself and let .ssh nest inside
# it. Docker mounts parent before child, so both work: $HOME is writable by uid
# 1000 and ~/.ssh still holds the keys. It persists between runs too, so the pip
# installs are only paid for once.
FINN_DOCKER_HOME="$_FINN_ENV_DIR/docker_home"
mkdir -p "$FINN_DOCKER_HOME"
FINN_HOME_MOUNT=" -v $FINN_DOCKER_HOME:/tmp/home_dir "
FINN_SSH_MOUNT=" -v $_FINN_ENV_DIR/ssh_keys:/tmp/home_dir/.ssh "

case " $FINN_DOCKER_EXTRA " in
  *":/tmp/home_dir "*) ;;                                  # already mounted, leave alone
  *) export FINN_DOCKER_EXTRA="${FINN_DOCKER_EXTRA}${FINN_HOME_MOUNT}" ;;
esac

case " $FINN_DOCKER_EXTRA " in
  *"/tmp/home_dir/.ssh"*) ;;                               # already mounted, leave alone
  *) export FINN_DOCKER_EXTRA="${FINN_DOCKER_EXTRA}${FINN_SSH_MOUNT}" ;;
esac

export FINN_DOCKER_NO_CACHE=0
export FINN_HOST_BUILD_DIR="$_FINN_ENV_DIR/build"

unset FINN_LICENSE_MOUNT FINN_SSH_MOUNT FINN_HOME_MOUNT FINN_DOCKER_HOME _FINN_ENV_DIR
