#!/usr/bin/env bash
set -e
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$workspace_dir/install/setup.bash"
if [[ "${1:-}" == "--ssh" ]]; then
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Usage: $0 --ssh user@robot [RViz arguments]" >&2
        exit 2
    fi
    fp_rviz_host="$2"
    shift 2
    fp_rviz_port="${FP_RVIZ_PORT:-8765}"
    fp_rviz_remote_port="${FP_RVIZ_REMOTE_PORT:-8765}"
    # Keep local Jazzy graph discovery away from the Foxy robot's network.
    export ROS_DOMAIN_ID="${FP_RVIZ_DOMAIN_ID:-166}"
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    export ROS_LOCALHOST_ONLY=1
    unset ROS_STATIC_PEERS ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT
    fp_rviz_control_dir="$(mktemp -d /tmp/fp-rviz-ssh-XXXXXX)"
    fp_rviz_relay_pid=""
    cleanup_fp_rviz() {
        if [[ -n "$fp_rviz_relay_pid" ]]; then
            kill "$fp_rviz_relay_pid" 2>/dev/null || true
            wait "$fp_rviz_relay_pid" 2>/dev/null || true
        fi
        ssh -S "$fp_rviz_control_dir/control" -O exit "$fp_rviz_host" 2>/dev/null || true
        rmdir "$fp_rviz_control_dir" 2>/dev/null || true
    }
    trap cleanup_fp_rviz EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    # Authenticate in the foreground, then keep our own tunnel in the background.
    ssh -M -S "$fp_rviz_control_dir/control" -fNT \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -L "127.0.0.1:${fp_rviz_port}:127.0.0.1:${fp_rviz_remote_port}" "$fp_rviz_host"
    # Run the executable directly so cleanup signals reach the relay process.
    "$(ros2 pkg prefix rb2301_fp)/lib/rb2301_fp/visualization_relay" \
        --port "$fp_rviz_port" --timeout "${FP_RVIZ_TIMEOUT:-3.0}" \
        --stale-after "${FP_RVIZ_STALE_AFTER:-3.0}" --poll-rate "${FP_RVIZ_POLL_RATE:-5.0}" &
    fp_rviz_relay_pid=$!
fi
rviz2 -d "$(ros2 pkg prefix rb2301_fp)/share/rb2301_fp/rviz/obstacle_course.rviz" "$@"
