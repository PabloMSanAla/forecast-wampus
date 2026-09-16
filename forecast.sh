#!/bin/bash

help() {
    cat << EOF

          ~~~~~  FORECAST Full Pipeline Parallel Batch Runner  ~~~~~

This script runs all 4 FORECAST modules (LC -> DF -> DC -> IGM) consecutively
for each lightcone plane in parallel across multiple CPU workers using GNU parallel.

Workflow per CPU worker:
  1. LC  (Lightcone Geometry & Particle Slicing)
  2. DF  (Dust-Free Flux Calculations)
  3. DC  (Dust Correction Post-Processing)
  4. IGM (Intergalactic Medium Attenuation)

Usage: ./forecast.sh [options]

Options:
  -n:            Number of parallel processors/workers (default: 1)
  -i:            Initial plane number (default: 0)
  -max:          Maximum plane number to process (default: max from planes_list.txt)
  -lc:           Path to the planes_list.txt file
  -ini-dir:      Directory containing all ini files (lc.ini, df.ini, dc.ini, igm.ini)
  -ini-lc:       Path to lc.ini (default: lc/lc.ini)
  -ini-df:       Path to df.ini (default: df/df.ini)
  -ini-dc:       Path to dc.ini (default: dc/dc.ini)
  -ini-igm:      Path to igm.ini (default: igm/igm.ini)
  -m, --mem:     Memory ceiling in GB per process (default: 4.0)
  -j, --joblog:  Path to file where GNU parallel logs job run statistics
  -r, --results: Directory path where stdout/stderr of each job is stored
  -f, --force:   Force recomputation of all modules even if output files exist
  -h, --help:    Show this help message and exit

Example usage:
  ./forecast.sh -n 8 -i 0
  ./forecast.sh -n 16 -i 0 -m 4.0 -j forecast_jobs.log -r forecast_logs/
  ./forecast.sh -n 4 -i 10 -max 20 -f

EOF
}

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

num_processors=1  # Default number of processors
init_plane=0     # Default initial plane number
max_plane_cli="" # Optional maximum plane number
planes_list_path=""
ini_dir=""
ini_lc=""
ini_df=""
ini_dc=""
ini_igm=""
memory_ceiling_gb=4.0
force_run=false
joblog_file=""
results_dir=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            help
            exit 0
            ;;
        -n)
            num_processors="$2"
            shift 2
            ;;
        -i)
            init_plane="$2"
            shift 2
            ;;
        -max)
            max_plane_cli="$2"
            shift 2
            ;;
        -lc)
            planes_list_path="$2"
            shift 2
            ;;
        -ini-dir)
            ini_dir="$2"
            shift 2
            ;;
        -ini-lc)
            ini_lc="$2"
            shift 2
            ;;
        -ini-df)
            ini_df="$2"
            shift 2
            ;;
        -ini-dc)
            ini_dc="$2"
            shift 2
            ;;
        -ini-igm)
            ini_igm="$2"
            shift 2
            ;;
        -m|--mem)
            memory_ceiling_gb="$2"
            shift 2
            ;;
        -j|--joblog)
            joblog_file="$2"
            shift 2
            ;;
        -r|--results)
            results_dir="$2"
            shift 2
            ;;
        -f|--force)
            force_run=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            help
            exit 1
            ;;
    esac
done

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel is not installed. Please install it (e.g. brew install parallel or apt install parallel)."
    exit 1
fi

# 1. Resolve INI files
if [ -n "$ini_dir" ]; then
    [ -z "$ini_lc" ] && ini_lc="${ini_dir}/lc.ini"
    [ -z "$ini_df" ] && ini_df="${ini_dir}/df.ini"
    [ -z "$ini_dc" ] && ini_dc="${ini_dir}/dc.ini"
    [ -z "$ini_igm" ] && ini_igm="${ini_dir}/igm.ini"
fi

# Fallback resolution for lc.ini
if [ -z "$ini_lc" ]; then
    if [ -n "$FORECAST_LC_INI" ] && [ -f "$FORECAST_LC_INI" ]; then
        ini_lc="$FORECAST_LC_INI"
    elif [ -f "${SCRIPT_DIR}/lc/lc.ini" ]; then
        ini_lc="${SCRIPT_DIR}/lc/lc.ini"
    elif [ -f "lc.ini" ]; then
        ini_lc="lc.ini"
    fi
fi

# Fallback resolution for df.ini
if [ -z "$ini_df" ]; then
    if [ -n "$FORECAST_DF_INI" ] && [ -f "$FORECAST_DF_INI" ]; then
        ini_df="$FORECAST_DF_INI"
    elif [ -f "${SCRIPT_DIR}/df/df.ini" ]; then
        ini_df="${SCRIPT_DIR}/df/df.ini"
    elif [ -f "df.ini" ]; then
        ini_df="df.ini"
    fi
fi

# Fallback resolution for dc.ini
if [ -z "$ini_dc" ]; then
    if [ -n "$FORECAST_DC_INI" ] && [ -f "$FORECAST_DC_INI" ]; then
        ini_dc="$FORECAST_DC_INI"
    elif [ -f "${SCRIPT_DIR}/dc/dc.ini" ]; then
        ini_dc="${SCRIPT_DIR}/dc/dc.ini"
    elif [ -f "dc.ini" ]; then
        ini_dc="dc.ini"
    fi
fi

# Fallback resolution for igm.ini
if [ -z "$ini_igm" ]; then
    if [ -n "$FORECAST_IGM_INI" ] && [ -f "$FORECAST_IGM_INI" ]; then
        ini_igm="$FORECAST_IGM_INI"
    elif [ -f "${SCRIPT_DIR}/igm/igm.ini" ]; then
        ini_igm="${SCRIPT_DIR}/igm/igm.ini"
    elif [ -f "igm.ini" ]; then
        ini_igm="igm.ini"
    fi
fi

# Validate INI files exist
for ini_f in "$ini_lc" "$ini_df" "$ini_dc" "$ini_igm"; do
    if [ -z "$ini_f" ] || [ ! -f "$ini_f" ]; then
        echo "Error: Configuration file '$ini_f' not found."
        exit 1
    fi
done

# Convert to absolute paths
ini_lc="$(cd "$(dirname "$ini_lc")" && pwd)/$(basename "$ini_lc")"
ini_df="$(cd "$(dirname "$ini_df")" && pwd)/$(basename "$ini_df")"
ini_dc="$(cd "$(dirname "$ini_dc")" && pwd)/$(basename "$ini_dc")"
ini_igm="$(cd "$(dirname "$ini_igm")" && pwd)/$(basename "$ini_igm")"

# 2. Extract output directories from each INI file
extract_outdir() {
    local ini_f=$1
    local out=""
    if [ -n "$FORECAST_OUTPUT_DIR" ]; then
        out="$FORECAST_OUTPUT_DIR"
    else
        out=$(awk '/!...PATH_WHERE_TO_OUTPUT_RESULTS/{getline; print $1}' "$ini_f")
    fi
    [[ "$out" != */ ]] && out="${out}/"
    echo "$out"
}

out_dir_lc=$(extract_outdir "$ini_lc")
out_dir_df=$(extract_outdir "$ini_df")
out_dir_dc=$(extract_outdir "$ini_dc")
out_dir_igm=$(extract_outdir "$ini_igm")

# 3. Resolve planes_list.txt path
if [ -z "$planes_list_path" ]; then
    if [ -n "$FORECAST_PLANES_LIST" ] && [ -f "$FORECAST_PLANES_LIST" ]; then
        planes_list_path="$FORECAST_PLANES_LIST"
    else
        planes_list_path=$(awk '/!...PLANES_LIST_FILE/{getline; print $1}' "$ini_lc")
    fi
    if [ -z "$planes_list_path" ] || [ ! -f "$planes_list_path" ]; then
        if [ -f "${SCRIPT_DIR}/lc/planes_list.txt" ]; then
            planes_list_path="${SCRIPT_DIR}/lc/planes_list.txt"
        elif [ -f "planes_list.txt" ]; then
            planes_list_path="planes_list.txt"
        elif [ -f "lc/planes_list.txt" ]; then
            planes_list_path="lc/planes_list.txt"
        fi
    fi
fi

if [ ! -f "$planes_list_path" ]; then
    echo "Error: planes_list.txt could not be found at '$planes_list_path'."
    exit 1
fi

planes_list_path="$(cd "$(dirname "$planes_list_path")" && pwd)/$(basename "$planes_list_path")"
export FORECAST_PLANES_LIST="$planes_list_path"

# 4. Parse planes and snapshot numbers
nl=$(cat "$planes_list_path" | wc -l)
declare -a planes
declare -a snaps
for i in $(seq 0 $((nl-1)))
do
    # col 1: display plane number, col 5: replica index (0-based), col 6: snapshot number
    planes[i]="$(cat "$planes_list_path" | awk -v p="$((i+1))" '{if(NR==p) print $5}')"
    snaps[i]="$(cat "$planes_list_path" | awk -v p="$((i+1))" '{if(NR==p) print $6}')"
done

max_planes=$(printf "%s\n" "${planes[@]}" | sort -nr | head -n 1)
if [ -n "$max_plane_cli" ]; then
    max_planes="$max_plane_cli"
fi

# Ensure executables exist and are executable
for mod in lc df dc igm; do
    exe="${SCRIPT_DIR}/${mod}/executable_${mod}"
    if [ ! -x "$exe" ]; then
        echo "Error: Executable '$exe' not found or not executable. Please compile it first with 'make -C $mod'."
        exit 1
    fi
done

echo "============================================================"
echo "          FORECAST Full Pipeline Batch Processor            "
echo "============================================================"
echo "  Planes list:   $planes_list_path"
echo "  Processors:    $num_processors"
echo "  Plane range:   $init_plane to $max_planes"
echo "  Memory limit:  $memory_ceiling_gb GB per process"
echo "  Force rerun:   $force_run"
echo "  Script root:   $SCRIPT_DIR"
echo "  LC  Output:    $out_dir_lc (ini: $ini_lc)"
echo "  DF  Output:    $out_dir_df (ini: $ini_df)"
echo "  DC  Output:    $out_dir_dc (ini: $ini_dc)"
echo "  IGM Output:    $out_dir_igm (ini: $ini_igm)"
[ -n "$joblog_file" ] && echo "  Job log:       $joblog_file"
[ -n "$results_dir" ] && echo "  Results dir:   $results_dir"
echo "============================================================"
echo ""

# Assemble extra GNU parallel arguments
parallel_args=()
if [ -n "$joblog_file" ]; then
    parallel_args+=(--joblog "$joblog_file")
fi
if [ -n "$results_dir" ]; then
    mkdir -p "$results_dir"
    parallel_args+=(--results "$results_dir")
fi

# Master function executed per CPU worker for a single plane
run_forecast_pipeline() {
    local snapshot=$1
    local plane=$2
    local mem_limit=$3
    local force=$4
    local root_dir=$5
    local ini_lc=$6
    local ini_df=$7
    local ini_dc=$8
    local ini_igm=$9
    local out_lc=${10}
    local out_df=${11}
    local out_dc=${12}
    local out_igm=${13}

    local exp_lc="${out_lc}coords.lc.${snapshot}_${plane}.txt"
    local exp_df="${out_df}flux.df.${snapshot}_${plane}.txt"
    local exp_dc="${out_dc}flux.dc.${snapshot}_${plane}.txt"
    local exp_igm="${out_igm}flux.igm.${snapshot}_${plane}.txt"

    echo ">>> [Plane ${plane} / Snap ${snapshot}] Starting FORECAST pipeline..."

    # -------------------------------------------------------------
    # 1. Module LC
    # -------------------------------------------------------------
    if [ "$force" != "true" ] && [ -s "$exp_lc" ]; then
        echo "  [1/4 LC]  [SKIP] Output exists: $exp_lc"
    else
        echo "  [1/4 LC]  Running executable_lc for snap ${snapshot}, plane ${plane}..."
        (cd "${root_dir}/lc" && ./executable_lc "${snapshot}" "${plane}" "${mem_limit}" -ini "${ini_lc}")
        if [ $? -ne 0 ]; then
            echo "  [ERROR] Module LC failed on snapshot ${snapshot}, plane ${plane}."
            return 1
        fi
    fi

    # -------------------------------------------------------------
    # 2. Module DF
    # -------------------------------------------------------------
    if [ "$force" != "true" ] && [ -s "$exp_df" ]; then
        echo "  [2/4 DF]  [SKIP] Output exists: $exp_df"
    else
        echo "  [2/4 DF]  Running executable_df for snap ${snapshot}, plane ${plane}..."
        (cd "${root_dir}/df" && ./executable_df "${snapshot}" "${plane}" "${mem_limit}" -ini "${ini_df}")
        if [ $? -ne 0 ]; then
            echo "  [ERROR] Module DF failed on snapshot ${snapshot}, plane ${plane}."
            return 1
        fi
    fi

    # -------------------------------------------------------------
    # 3. Module DC
    # -------------------------------------------------------------
    if [ "$force" != "true" ] && [ -s "$exp_dc" ]; then
        echo "  [3/4 DC]  [SKIP] Output exists: $exp_dc"
    else
        echo "  [3/4 DC]  Running executable_dc for snap ${snapshot}, plane ${plane}..."
        (cd "${root_dir}/dc" && ./executable_dc "${snapshot}" "${plane}" "${mem_limit}" -ini "${ini_dc}")
        if [ $? -ne 0 ]; then
            echo "  [ERROR] Module DC failed on snapshot ${snapshot}, plane ${plane}."
            return 1
        fi
    fi

    # -------------------------------------------------------------
    # 4. Module IGM
    # -------------------------------------------------------------
    if [ "$force" != "true" ] && [ -s "$exp_igm" ]; then
        echo "  [4/4 IGM] [SKIP] Output exists: $exp_igm"
    else
        echo "  [4/4 IGM] Running executable_igm for snap ${snapshot}, plane ${plane}..."
        (cd "${root_dir}/igm" && ./executable_igm "${snapshot}" "${plane}" "${mem_limit}" -ini "${ini_igm}")
        if [ $? -ne 0 ]; then
            echo "  [ERROR] Module IGM failed on snapshot ${snapshot}, plane ${plane}."
            return 1
        fi
    fi

    echo "<<< [Plane ${plane} / Snap ${snapshot}] FORECAST pipeline COMPLETED successfully."
    return 0
}

export -f run_forecast_pipeline

# Generate plane parameters and pipe into GNU parallel
for (( i = 0; i < $nl; i++ ))
do
    if [[ "${planes[$i]}" -ge "$init_plane" && "${planes[$i]}" -le "$max_planes" ]]; then
        echo "${snaps[$i]} ${planes[$i]} ${memory_ceiling_gb} ${force_run} ${SCRIPT_DIR} ${ini_lc} ${ini_df} ${ini_dc} ${ini_igm} ${out_dir_lc} ${out_dir_df} ${out_dir_dc} ${out_dir_igm}"
    fi
done | parallel --bar --jobs "$num_processors" "${parallel_args[@]}" --colsep ' ' run_forecast_pipeline {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}

echo ""
echo "Pipeline batch execution finished."
