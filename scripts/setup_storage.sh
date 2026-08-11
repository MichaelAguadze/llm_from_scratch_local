#!/usr/bin/env bash
# Inspect and (if safe) mount /dev/nvme0n1p4 as the SLM build storage.
#
#   sudo bash scripts/setup_storage.sh            # inspect only, changes nothing
#   sudo bash scripts/setup_storage.sh --commit   # inspect, then mount rw + fstab
#
# Safety model:
#   - The inspect pass mounts READ-ONLY and unmounts before exiting.
#   - --commit refuses to touch a partition that holds anything other than
#     lost+found, unless you also pass --force-nonempty.
#   - /etc/fstab is backed up before edit, and the entry uses UUID + nofail so a
#     missing disk can never block boot.

set -euo pipefail

DEV="/dev/nvme0n1p4"
UUID="9d929f21-9db7-436b-ae3c-d995ef89befa"
MNT="/mnt/slm"
INSPECT_MNT="/mnt/slm-inspect"
OWNER="michael"
SUBDIR="slm-125m"

COMMIT=0
FORCE_NONEMPTY=0
for arg in "$@"; do
  case "$arg" in
    --commit)         COMMIT=1 ;;
    --force-nonempty) FORCE_NONEMPTY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "must run as root: sudo bash $0 $*" >&2; exit 1; }

say() { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------- preflight --
say "preflight"
blkid "$DEV"

actual_uuid=$(blkid -s UUID -o value "$DEV")
if [[ "$actual_uuid" != "$UUID" ]]; then
  echo "ABORT: $DEV has UUID $actual_uuid, expected $UUID" >&2
  echo "The partition layout changed. Re-check before proceeding." >&2
  exit 1
fi

if findmnt --source "$DEV" >/dev/null 2>&1; then
  echo "ABORT: $DEV is already mounted:" >&2
  findmnt --source "$DEV" >&2
  exit 1
fi

if grep -q "$UUID" /etc/fstab; then
  echo "NOTE: an /etc/fstab entry for this UUID already exists:"
  grep "$UUID" /etc/fstab
fi

# ------------------------------------------------------------- inspect (ro) --
say "read-only inspect"
mkdir -p "$INSPECT_MNT"
mount -o ro,noexec,nodev "$DEV" "$INSPECT_MNT"
cleanup() { mountpoint -q "$INSPECT_MNT" && umount "$INSPECT_MNT"; rmdir "$INSPECT_MNT" 2>/dev/null || true; }
trap cleanup EXIT

df -hT "$INSPECT_MNT"

echo
echo "-- top level --"
ls -la "$INSPECT_MNT"

echo
echo "-- entries excluding lost+found --"
mapfile -t entries < <(find "$INSPECT_MNT" -mindepth 1 -maxdepth 1 ! -name lost+found -printf '%f\n')
if [[ ${#entries[@]} -eq 0 ]]; then
  echo "(none — partition is empty)"
else
  printf '%s\n' "${entries[@]}"
  echo
  echo "-- size of each --"
  du -sh "$INSPECT_MNT"/* 2>/dev/null | sort -rh | head -20
  echo
  echo "-- total file count (may take a moment) --"
  find "$INSPECT_MNT" -xdev -type f 2>/dev/null | wc -l
fi

echo
echo "-- does this look like an OS root filesystem? --"
os_markers=0
for d in etc boot var usr home root bin sbin lib; do
  if [[ -e "$INSPECT_MNT/$d" ]]; then
    echo "  FOUND: /$d"
    os_markers=$((os_markers + 1))
  fi
done
if [[ -f "$INSPECT_MNT/etc/os-release" ]]; then
  echo "  /etc/os-release says:"
  sed 's/^/    /' "$INSPECT_MNT/etc/os-release" | head -5
fi
if [[ $os_markers -ge 3 ]]; then
  echo "  >> VERDICT: looks like a LINUX ROOT FILESYSTEM. Do not overwrite."
elif [[ ${#entries[@]} -eq 0 ]]; then
  echo "  >> VERDICT: empty. Safe to use."
else
  echo "  >> VERDICT: has data, but not an OS root. Review the listing above."
fi

# --------------------------------------------------------------- commit -----
if [[ $COMMIT -eq 0 ]]; then
  echo
  echo "Inspect-only run. Nothing was changed; partition unmounted on exit."
  echo "If the verdict above is 'empty', re-run with:  sudo bash $0 --commit"
  exit 0
fi

if [[ ${#entries[@]} -gt 0 && $FORCE_NONEMPTY -eq 0 ]]; then
  echo
  echo "ABORT: partition is not empty and --force-nonempty was not given." >&2
  echo "Review the listing above. Nothing was changed." >&2
  exit 1
fi

if [[ $os_markers -ge 3 ]]; then
  echo
  echo "ABORT: refusing to claim what looks like an OS root filesystem." >&2
  exit 1
fi

say "committing"
cleanup
trap - EXIT

mkdir -p "$MNT"
mount -o rw,noatime "$DEV" "$MNT"
mkdir -p "$MNT/$SUBDIR"
chown "$OWNER:$OWNER" "$MNT/$SUBDIR"
chmod 755 "$MNT/$SUBDIR"

if ! grep -q "$UUID" /etc/fstab; then
  cp -a /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d-%H%M%S)"
  printf 'UUID=%s  %s  ext4  defaults,noatime,nofail  0  2\n' "$UUID" "$MNT" >> /etc/fstab
  echo "added to /etc/fstab (backup written alongside it)"
fi

# Verify the fstab entry is valid without rebooting.
umount "$MNT"
mount -a
findmnt "$MNT" || { echo "ABORT: fstab entry did not mount. Check /etc/fstab." >&2; exit 1; }

say "result"
df -hT "$MNT"
ls -la "$MNT"
echo
echo "SLM_ROOT=$MNT/$SUBDIR"
echo "Add that line to .env.local"
