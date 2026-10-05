#!/bin/bash

help() {
    cat << EOF

          ~~~~~  FORECAST Image Generation Batch Processing Script  ~~~~~

This script runs the create_image postprocessing module for multiple planes in parallel using GNU parallel.
It dynamically extracts parameters (filter name, module, output directory, catalog directory, and planes list)
from image.ini, checks if the target FITS output file already exists to avoid redundant computation,
and verifies the input catalog exists before executing create_image.

Usage: ./im.sh [options]

Options:
  -n: Number of processors / concurrent jobs to use (default: 1)
  -i: Initial plane number to process (default: 0)
  -lc: Path to the planes_list.txt file (default: from image.ini or ../../lc/planes_list.txt)
  -ini: Path to the image.ini file (default: image.ini)
  -f, --force: Force recomputation even if output FITS file already exists
  -h, --help: Show this help message and exit

Example usage:
  ./im.sh -n 4 -i 0
  ./im.sh -n 4 -i 10 -f
  ./im.sh -n 2 -ini /path/to/custom_image.ini

EOF
}

num_processors=1   # Default number of processors
init_plane=0      # Default initial plane number
planes_list_path=""
ini_file="image.ini"
force_run=false

# Parse command-line arguments
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
        -lc)
            planes_list_path="$2"
            shift 2
            ;;
        -ini)
            ini_file="$2"
            shift 2
            ;;
        -f|--force)
            force_run=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel is not installed. Please install it with: brew install parallel"
    exit 1
fi

# 1. Resolve image.ini path
if [ -n "$FORECAST_IMAGE_INI" ] && [ -f "$FORECAST_IMAGE_INI" ]; then
    ini_file="$FORECAST_IMAGE_INI"
elif [ ! -f "$ini_file" ] && [ -f "postprocessing/create_image/image.ini" ]; then
    ini_file="postprocessing/create_image/image.ini"
fi

if [ ! -f "$ini_file" ]; then
    echo "Error: image.ini configuration file not found at '$ini_file'."
    exit 1
fi

# 2. Extract configuration from image.ini
filter_name=$(awk '/!...FILTER_NAME_FROM_FILTER_FILE/{getline; print $1}' "$ini_file")
module_name=$(awk '/!...MODULE_NAME_df_dc_OR_igm/{getline; print $1}' "$ini_file")
output_dir=$(awk '/!...PATH_WHERE_TO_OUTPUT_RESULTS/{getline; print $1}' "$ini_file")
cat_dir=$(awk '/!...PATH_WHERE_CATALOGUES_ARE_LOCATED/{getline; print $1}' "$ini_file")

if [ -z "$filter_name" ]; then
    echo "Error: FILTER_NAME_FROM_FILTER_FILE not found in '$ini_file'."
    exit 1
fi

if [ -z "$module_name" ]; then
    echo "Error: MODULE_NAME_df_dc_OR_igm not found in '$ini_file'."
    exit 1
fi

if [ -z "$output_dir" ]; then
    echo "Error: PATH_WHERE_TO_OUTPUT_RESULTS not found in '$ini_file'."
    exit 1
fi
[[ "$output_dir" != */ ]] && output_dir="${output_dir}/"

if [ -z "$cat_dir" ]; then
    echo "Error: PATH_WHERE_CATALOGUES_ARE_LOCATED not found in '$ini_file'."
    exit 1
fi
[[ "$cat_dir" != */ ]] && cat_dir="${cat_dir}/"

# 3. Resolve planes_list.txt path
if [ -z "$planes_list_path" ]; then
    if [ -n "$FORECAST_PLANES_LIST" ] && [ -f "$FORECAST_PLANES_LIST" ]; then
        planes_list_path="$FORECAST_PLANES_LIST"
    elif [ -f "$ini_file" ]; then
        planes_list_path=$(awk '/!...PLANES_LIST_FILE/{getline; print $1}' "$ini_file")
    fi
    if [ -z "$planes_list_path" ] || [ ! -f "$planes_list_path" ]; then
        if [ -f "../../lc/planes_list.txt" ]; then
            planes_list_path="../../lc/planes_list.txt"
        elif [ -f "../lc/planes_list.txt" ]; then
            planes_list_path="../lc/planes_list.txt"
        elif [ -f "lc/planes_list.txt" ]; then
            planes_list_path="lc/planes_list.txt"
        elif [ -f "planes_list.txt" ]; then
            planes_list_path="planes_list.txt"
        fi
    fi
fi

if [ ! -f "$planes_list_path" ]; then
    echo "Error: planes_list.txt could not be found at '$planes_list_path'."
    exit 1
fi

nl=$(cat "$planes_list_path" | wc -l)
declare -a planes
declare -a snaps
for i in $(seq 0 $((nl-1)))
do
    # col 1 is display plane number (1-based), col 5 is replica index (0-based), col 6 is snapshot number
    planes[i]="$(cat "$planes_list_path" | awk -v p="$((i+1))" '{if(NR==p) print $5}')"
    snaps[i]="$(cat "$planes_list_path" | awk -v p="$((i+1))" '{if(NR==p) print $6}')"
done

max_planes=$(printf "%s\n" "${planes[@]}" | sort -nr | head -n 1)

echo "Configuration:"
echo "  Ini file:     $ini_file"
echo "  Filter:       $filter_name"
echo "  Module:       $module_name"
echo "  Catalog dir:  $cat_dir"
echo "  Output dir:   $output_dir"
echo "  Planes list:  $planes_list_path"
echo "  Processors:   $num_processors"
echo "  Plane range:  $init_plane to $max_planes"
echo "  Force rerun:  $force_run"
echo ""

# Function to run create_image with existence check
run_im() {
    local snapshot=$1
    local plane=$2
    local out_dir=$3
    local cat_path=$4
    local filter=$5
    local module=$6
    local force=$7
    local ini_path=$8

    local expected_output="${out_dir}${filter}.${module}.${snapshot}_${plane}.fits"
    local input_cat="${cat_path}flux.${module}.${snapshot}_${plane}.txt"

    # Check if target FITS file already exists and is not empty
    if [ "$force" != "true" ] && [ -s "$expected_output" ]; then
        echo "[SKIP] FITS image already exists: $expected_output"
        return 0
    fi

    # Check if input catalog exists
    if [ ! -s "$input_cat" ]; then
        echo "[WARN] Input catalog missing or empty: $input_cat (skipping snapshot $snapshot plane $plane)"
        return 0
    fi

    echo "Processing image for snapshot: $snapshot, plane: $plane -> $expected_output"
    
    ./create_image "${snapshot}" "${plane}" -ini "${ini_path}"
    local status=$?
    
    if [ $status -ne 0 ]; then
        echo "Error processing image for snapshot: $snapshot plane: $plane (exit code $status)."
        return 1
    fi
    return 0
}

export -f run_im

# Generate plane/snapshot tuples and process in parallel using GNU parallel
for (( i = 0; i < $nl; i++ ))
do
    if [[ "${planes[$i]}" -ge "$init_plane" && "${planes[$i]}" -le "$max_planes" ]]; then
        echo "${snaps[$i]} ${planes[$i]} ${output_dir} ${cat_dir} ${filter_name} ${module_name} ${force_run} ${ini_file}"
    fi
done | parallel --bar --jobs "$num_processors" --colsep ' ' run_im {1} {2} {3} {4} {5} {6} {7} {8}

