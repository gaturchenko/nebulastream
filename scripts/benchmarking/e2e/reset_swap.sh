#!/usr/bin/env bash
# Empties swap before a measurement: pages an earlier experiment left swapped out would come back during the next run and add to
# its memory and latency. Deployed on the Pi as ~/navi/scripts/reset_swap.sh and used by every Pi driver.
#
#   reset_swap.sh        turn every active swap area off and on again
#   reset_swap.sh off    turn them off and remember them, for drivers that measure without swap
#   reset_swap.sh on     turn the remembered areas on again
#
# Each area is switched the way it was enabled. One that systemd manages -- on the Pi, the rpi-swap zram device with /var/swap as
# its write-back file -- goes through its unit, which resets and sets up the zram device again: swapoff/swapon behind systemd's
# back makes the zram setup fail, and `swapon -a` only knows /etc/fstab, which lists no swap on the Pi, so `swapoff -a && swapon
# -a` left it without swap. Other areas get their priority back. The areas are remembered in ~/.cache, which outlives the login
# session of a driver started with nohup (/run/user does not). Exits non-zero if an area could not be switched.
set -u

STATE="${HOME}/.cache/reset_swap.areas"

swap_off() {
    local line area priority unit status=0
    mapfile -t lines < <(tail -n +2 /proc/swaps)
    # A second `off` finds nothing active and must not forget what the first one turned off.
    (( ${#lines[@]} == 0 )) && return 0
    mkdir -p "$(dirname "$STATE")"
    : > "$STATE"
    for line in "${lines[@]}"; do
        read -r area _ _ _ priority <<< "$line"
        unit=$(systemd-escape -p --suffix=swap "$area")
        if systemctl is-active --quiet "$unit"; then
            echo "unit $unit" >> "$STATE"
            sudo -n systemctl stop "$unit" || status=1
        else
            echo "area $area $priority" >> "$STATE"
            sudo -n swapoff "$area" || status=1
        fi
    done
    return "$status"
}

swap_on() {
    local kind name priority status=0
    [[ -f "$STATE" ]] || return 0
    while read -r kind name priority; do
        if [[ "$kind" == unit ]]; then
            sudo -n systemctl start "$name" || status=1
        elif (( priority >= 0 )); then
            sudo -n swapon -p "$priority" "$name" || status=1
        else
            # A negative priority was assigned by the kernel; swapon -p accepts none but -1, so let it assign one again.
            sudo -n swapon "$name" || status=1
        fi
    done < "$STATE"
    (( status == 0 )) && rm -f "$STATE"
    return "$status"
}

case "${1:-cycle}" in
    off) swap_off ;;
    on) swap_on ;;
    # Turn back on whatever did go off, even if another area failed to.
    cycle) swap_off; off=$?; swap_on && exit "$off" ;;
    *) echo "usage: $0 [off|on]" >&2; exit 2 ;;
esac
