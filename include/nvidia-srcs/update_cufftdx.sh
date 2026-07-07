#!/usr/bin/env bash
#
# update_cufftdx.sh - fetch a pristine NVIDIA MathDx (cuFFTDx) release and apply the FastFFT
# customizations to it, so that moving to a new cufftdx version is a single command instead of
# a manual porting session.
#
# What FastFFT changes relative to a pristine mathdx release (see also patches/*.patch):
#
#   1. PTX database (cufftdx/database/*.inc): every `tid.y` becomes `tid.z`.
#      cuFFTDx's pre-generated PTX uses tid.y to implicitly batch multiple 1d FFTs per block.
#      FastFFT reserves threadIdx.y for its transform-decomposition kernels, so implicit
#      batching is moved to tid.z (which is fine: FastFFT always builds FFTs with
#      ffts_per_block == 1, and tid.z remains available for batching in other contexts).
#
#   2. Header guards (10 sites in 6 headers, all the same shape): every place the C++ helper
#      code computes a shared-memory batch offset from threadIdx.y is wrapped in
#          if constexpr (FFT::ffts_per_block > 1) { <original offset> } else { <no offset> }
#      so that threadIdx.y is never consulted when there is a single FFT per block.
#      In fft_execution.hpp this requires threading an `FPB` template parameter through
#      shared_to_registers_impl. In preprocess_fold.hpp the unsupported "shared memory API"
#      path is additionally blocked with a deferred static_assert (static_no_shared_api).
#      Known intentional exception: `smem[threadIdx.y]` in postprocess_r2c_packed
#      (fft_block_postprocess.hpp) is left pristine - FastFFT never uses the packed real
#      layout, and it indexes (not offsets) shared memory.
#
#   3. block_fft.hpp: SM86 records forward to SM80 records instead of SM70, enabling larger
#      FFT sizes on 8.6.
#
# The header changes are stored as a version-specific patch in patches/. For a new mathdx
# version, this script tries the newest available patch with fuzz; if hunks fail you get
# .rej files plus a report of every unguarded threadIdx.y site that still needs porting.
#
# Usage:
#   ./update_cufftdx.sh apply  <version> [cuda12|cuda13]   # fetch + customize + emit .mk stub
#   ./update_cufftdx.sh apply  <version> --tarball <file>  # same, from a local tarball
#   ./update_cufftdx.sh verify <version>                   # re-run invariant checks on an installed tree
#   ./update_cufftdx.sh diff   <version> [cuda12|cuda13]   # diff installed tree vs pristine (mod audit)
#
# Examples:
#   ./update_cufftdx.sh apply 25.06.1 cuda13
#   ./update_cufftdx.sh apply 26.06.0 cuda13     # future version; expect to review patch fuzz
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/patches"
DOWNLOAD_DIR="${SCRIPT_DIR}/.downloads"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

# NVIDIA download URL candidates for a given version/flavor, newest layout first.
url_candidates() {
    local version=$1 flavor=$2
    echo "https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/${flavor}/nvidia-mathdx-${version}-${flavor}.tar.gz"
    echo "https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/nvidia-mathdx-${version}.tar.gz"
}

fetch_tarball() {
    local version=$1 flavor=$2 out=""
    mkdir -p "${DOWNLOAD_DIR}"
    local url
    while read -r url; do
        out="${DOWNLOAD_DIR}/$(basename "$url")"
        if [ -s "$out" ]; then
            info "Using cached ${out}" >&2
            echo "$out"
            return 0
        fi
        info "Trying ${url}" >&2
        if curl -fsSL -o "$out" "$url" 2>/dev/null && [ -s "$out" ] && gzip -t "$out" 2>/dev/null; then
            echo "$out"
            return 0
        fi
        rm -f "$out"
    done < <(url_candidates "$version" "$flavor")
    die "Could not download nvidia-mathdx ${version} (${flavor}). Download it manually from
       https://developer.nvidia.com/cufftdx-downloads and re-run with: apply ${version} --tarball <file>"
}

# Innermost directory containing cufftdx.hpp (the -I root); layout differs across versions.
find_include_root() {
    local tree=$1
    local hpp
    hpp=$(find "$tree" -name cufftdx.hpp -printf '%d %p\n' | sort -n | head -n1 | cut -d' ' -f2-)
    [ -n "$hpp" ] || die "cufftdx.hpp not found under ${tree}"
    dirname "$hpp"
}

# The cufftdx PTX database directory under an include root (not cublasdx/cusolverdx, not backups).
find_database_dir() {
    local include_root=$1
    find "$include_root" -type d -name database \
        -not -path '*cublasdx*' -not -path '*cusolverdx*' -not -path '*nvshmem*' -not -path '*.bak*' \
        | head -n1
}

count_in_incs() { # count occurrences of a pattern across the database .inc files
    # grep -c exits non-zero when the count is 0, which is a success state for us; do not let
    # it kill the script under set -e/pipefail.
    local db=$1 pat=$2
    { grep -c "$pat" "$db"/*.inc 2>/dev/null || true; } | awk -F: '{s+=$2} END{print s+0}'
}

tid_swap_database() {
    local db=$1
    local n_tidy n_tidz
    n_tidy=$(count_in_incs "$db" 'tid\.y')
    n_tidz=$(count_in_incs "$db" 'tid\.z')
    info "Database ${db}: ${n_tidy} tid.y / ${n_tidz} tid.z before swap"
    if [ "$n_tidy" -eq 0 ] && [ "$n_tidz" -gt 0 ]; then
        info "Database already swapped; skipping"
        return 0
    fi
    [ "$n_tidy" -gt 0 ] || die "No tid.y found in ${db} - unexpected database layout"
    [ "$n_tidz" -eq 0 ] || die "Found pre-existing tid.z in a database that still has tid.y - refusing to guess"

    if [ ! -d "${db}.bak" ]; then
        cp -r "$db" "${db}.bak"
        info "Pristine database backed up to ${db}.bak"
    fi

    sed -i 's/tid\.y/tid.z/g' "$db"/*.inc

    local n_tidy_new n_tidz_new
    n_tidy_new=$(count_in_incs "$db" 'tid\.y')
    n_tidz_new=$(count_in_incs "$db" 'tid\.z')
    [ "$n_tidy_new" -eq 0 ] || { restore_database "$db"; die "tid.y remains after swap (${n_tidy_new})"; }
    [ "$n_tidz_new" -eq "$n_tidy" ] || { restore_database "$db"; die "tid.z count (${n_tidz_new}) != original tid.y count (${n_tidy})"; }
    info "Swap OK: ${n_tidz_new} tid.z, 0 tid.y"
}

restore_database() {
    local db=$1
    if [ -d "${db}.bak" ]; then
        rm -f "$db"/*.inc
        cp "${db}.bak"/*.inc "$db"/
    fi
}

pick_patch() {
    local version=$1
    local exact="${PATCH_DIR}/nvidia-mathdx-${version}-fastfft-headers.patch"
    if [ -f "$exact" ]; then
        echo "$exact"
        return 0
    fi
    # Fall back to the newest patch we have; patch fuzz usually copes with small upstream drift.
    local latest
    latest=$(ls -1 "${PATCH_DIR}"/nvidia-mathdx-*-fastfft-headers.patch 2>/dev/null | sort -V | tail -n1)
    [ -n "$latest" ] || die "No header patch found in ${PATCH_DIR}"
    echo "WARNING: no patch for ${version}; falling back to $(basename "$latest")" >&2
    echo "$latest"
}

apply_header_patch() {
    local include_root=$1 version=$2
    local patch_file
    patch_file=$(pick_patch "$version")
    info "Applying $(basename "$patch_file") in ${include_root}"
    if grep -rq "static_no_shared_api" "$include_root/cufftdx" 2>/dev/null; then
        info "Headers appear to be already patched; skipping"
        return 0
    fi
    if ! (cd "$include_root" && patch -p1 --forward --no-backup-if-mismatch < "$patch_file"); then
        echo "" >&2
        echo "Header patch did not apply cleanly. Look for .rej files under ${include_root}:" >&2
        find "$include_root" -name '*.rej' >&2 || true
        echo "Port the failed hunks by hand, keeping the invariant: no shared-memory batch offset" >&2
        echo "may read threadIdx.y outside an 'if constexpr (FFT::ffts_per_block > 1)' branch." >&2
        echo "Then re-run: $0 verify <version>" >&2
        exit 1
    fi
}

# Report threadIdx.y uses in headers that are not inside/near an ffts_per_block/FPB guard.
# This is the tripwire for new versions: any new upstream use of threadIdx.y shows up here.
check_unguarded_tidy() {
    local include_root=$1
    local files
    files=$(grep -rl "threadIdx\.y" "$include_root/cufftdx" --include='*.hpp' 2>/dev/null || true)
    [ -n "$files" ] || return 0
    local report
    report=$(awk '
        FNR == 1 { guard = 0 }
        /^[[:space:]]*\/\// { next }  # comments are not code sites
        /ffts_per_block > 1|FPB > 1/ { guard = FNR }
        /threadIdx\.y/ {
            if (!(FNR - guard <= 8 && guard > 0)) {
                # known intentional exception: packed-layout mid-element shuffle, unused by FastFFT
                if ($0 ~ /smem\[threadIdx\.y\]/) next
                printf "%s:%d:%s\n", FILENAME, FNR, $0
            }
        }
    ' $files 2>/dev/null)
    if [ -n "$report" ]; then
        echo "Unguarded threadIdx.y sites (need the ffts_per_block > 1 guard or a whitelist entry):" >&2
        echo "$report" >&2
        return 1
    fi
    return 0
}

emit_mk_stub() {
    local version=$1 include_root=$2
    local rel="${include_root#"${SCRIPT_DIR}"/}"
    local mk="${SCRIPT_DIR}/nvidia-mathdx-${version}.mk"
    echo "CUFFTDX_VERSION_INCLUDE_FLAGS := -I../include/nvidia-srcs/${rel}/" > "$mk"
    info "Wrote ${mk}"
    info "Select this version in build/Makefile with: include ../include/nvidia-srcs/nvidia-mathdx-${version}.mk"
}

verify_tree() {
    local version=$1
    local tree="${SCRIPT_DIR}/nvidia-mathdx-${version}"
    [ -d "$tree" ] || die "${tree} does not exist"
    local include_root db rc=0
    include_root=$(find_include_root "$tree")
    db=$(find_database_dir "$include_root")
    [ -n "$db" ] || die "No cufftdx database dir found under ${include_root}"

    local n_tidy n_tidz
    n_tidy=$(count_in_incs "$db" 'tid\.y')
    n_tidz=$(count_in_incs "$db" 'tid\.z')
    if [ "$n_tidy" -eq 0 ] && [ "$n_tidz" -gt 0 ]; then
        info "PTX database OK (0 tid.y, ${n_tidz} tid.z)"
    else
        echo "FAIL: database has ${n_tidy} tid.y / ${n_tidz} tid.z" >&2; rc=1
    fi
    if grep -q "static_no_shared_api" "$include_root"/cufftdx/detail/processing/preprocess_fold.hpp 2>/dev/null; then
        info "Header mods present (static_no_shared_api found)"
    else
        echo "FAIL: header mods missing (no static_no_shared_api in preprocess_fold.hpp)" >&2; rc=1
    fi
    if check_unguarded_tidy "$include_root"; then
        info "No unguarded threadIdx.y sites"
    else
        rc=1
    fi
    [ $rc -eq 0 ] && info "verify PASSED for ${version}" || die "verify FAILED for ${version}"
}

# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------

cmd_apply() {
    local version=${1:-}; shift || true
    [ -n "$version" ] || die "apply requires a version, e.g. 25.06.1"
    local flavor="cuda13" tarball=""
    while [ $# -gt 0 ]; do
        case "$1" in
            cuda12|cuda13) flavor=$1 ;;
            --tarball) shift; tarball=${1:-}; [ -f "$tarball" ] || die "--tarball file not found: ${tarball}" ;;
            *) die "unknown argument: $1" ;;
        esac
        shift
    done

    local dest="${SCRIPT_DIR}/nvidia-mathdx-${version}"
    if [ -d "$dest" ]; then
        info "${dest} already exists; applying mods in place (no re-extract)."
        info "To start from pristine: remove it (or restore from database.bak) and re-run."
    else
        [ -n "$tarball" ] || tarball=$(fetch_tarball "$version" "$flavor")
        info "Extracting $(basename "$tarball")"
        local staging
        staging=$(mktemp -d "${SCRIPT_DIR}/.extract-XXXXXX")
        tar -xzf "$tarball" -C "$staging"
        # Tarballs unpack as nvidia-mathdx-<version>/...; tolerate a missing top dir too.
        if [ -d "${staging}/nvidia-mathdx-${version}" ]; then
            mv "${staging}/nvidia-mathdx-${version}" "$dest"
        elif [ -d "${staging}/nvidia" ]; then
            mkdir -p "$dest" && mv "${staging}/nvidia" "$dest/"
        else
            rm -rf "$staging"
            die "Unrecognized tarball layout"
        fi
        rm -rf "$staging"
    fi

    local include_root db
    include_root=$(find_include_root "$dest")
    info "Include root: ${include_root}"
    db=$(find_database_dir "$include_root")
    [ -n "$db" ] || die "No cufftdx database dir found under ${include_root}"

    tid_swap_database "$db"
    apply_header_patch "$include_root" "$version"
    emit_mk_stub "$version" "$include_root"
    verify_tree "$version"
    info "Done. nvidia-mathdx-${version} is ready for FastFFT."
}

cmd_diff() {
    local version=${1:-}; shift || true
    [ -n "$version" ] || die "diff requires a version"
    local flavor=${1:-cuda13}
    local tree="${SCRIPT_DIR}/nvidia-mathdx-${version}"
    [ -d "$tree" ] || die "${tree} does not exist"
    local tarball
    tarball=$(fetch_tarball "$version" "$flavor")
    local work
    work=$(mktemp -d)
    tar -xzf "$tarball" -C "$work"
    local pristine="${work}/nvidia-mathdx-${version}"
    [ -d "$pristine" ] || die "Unexpected tarball layout"
    echo "=== Modified files (excluding the ~367 tid-swapped database .inc files and backups) ==="
    diff -qr "$pristine" "$tree" 2>/dev/null \
        | grep -v '\.inc ' | grep -v 'database.bak\|\.downloads\|\.DS_Store' || true
    echo ""
    echo "=== Database .inc files differing (tid swap): $(diff -qr "$pristine" "$tree" 2>/dev/null | grep -c '\.inc ') ==="
    echo ""
    echo "Full header diff (pristine -> installed):"
    local f rel
    while read -r f; do
        rel=${f#"$pristine"/}
        [ -f "$tree/$rel" ] || continue
        case "$rel" in *.hpp|*.h) ;; *) continue ;; esac
        if ! cmp -s "$f" "$tree/$rel"; then
            diff -u "$f" "$tree/$rel" | head -80 || true
        fi
    done < <(find "$pristine" -type f \( -name '*.hpp' -o -name '*.h' \) -not -path '*.inc')
    rm -rf "$work"
}

usage() {
    sed -n '2,45p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 1
}

case "${1:-}" in
    apply)  shift; cmd_apply "$@" ;;
    verify) shift; verify_tree "${1:?verify requires a version}" ;;
    diff)   shift; cmd_diff "$@" ;;
    *) usage ;;
esac
