#!/bin/bash


help() {
    cat << EOF

          ~~~~~  FORECAST DF module batch processing script  ~~~~~

This script runs the DF module for multiple planes in parallel using GNU parallel.
It reads the plane numbers and their corresponding snapshot numbers from planes_list.txt,
checks if the output catalog file already exists to avoid redundant computation,
and then runs executable_df for each plane-snapshot pair in parallel.

Usage: ./df.sh [options]

Options:
  -n: Number of processors to use (default: 1)
  -i: Initial plane number (default: 0)
  -lc: Path to the planes_list.txt file (default: from df.ini or ../lc/planes_list.txt)
  -ini: Path to the df.ini file (default: df.ini)
  -f, --force: Force recomputation even if output file already exists
  -h, --help: Show this help message and exit

Example usage:
  ./df.sh -n 4 -i 36
  ./df.sh -n 4 -i 0 -f

EOF
}

num_processors=1  # Default number of processors
init_plane=0     # Default initial plane number
planes_list_path=""
ini_file="df.ini"
output_dir=""
force_run=false

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

# 1. Resolve df.ini path
if [ -n "$FORECAST_DF_INI" ] && [ -f "$FORECAST_DF_INI" ]; then
    ini_file="$FORECAST_DF_INI"
elif [ ! -f "$ini_file" ] && [ -f "df/df.ini" ]; then
    ini_file="df/df.ini"
fi

if [ ! -f "$ini_file" ]; then
    echo "Error: df.ini configuration file not found at '$ini_file'."
    exit 1
fi

# 2. Extract output directory directly from df.ini
output_dir=$(awk '/!...PATH_WHERE_TO_OUTPUT_RESULTS/{getline; print $1}' "$ini_file")
if [ -z "$output_dir" ]; then
    echo "Error: PATH_WHERE_TO_OUTPUT_RESULTS not found in '$ini_file'."
    exit 1
fi
# Ensure trailing slash
[[ "$output_dir" != */ ]] && output_dir="${output_dir}/"

# 3. Resolve planes_list.txt path
if [ -z "$planes_list_path" ]; then
    if [ -n "$FORECAST_PLANES_LIST" ] && [ -f "$FORECAST_PLANES_LIST" ]; then
        planes_list_path="$FORECAST_PLANES_LIST"
    elif [ -f "$ini_file" ]; then
        planes_list_path=$(awk '/!...PLANES_LIST_FILE/{getline; print $1}' "$ini_file")
    fi
    if [ -z "$planes_list_path" ] || [ ! -f "$planes_list_path" ]; then
        if [ -f "../lc/planes_list.txt" ]; then
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
echo "  Planes list: $planes_list_path"
echo "  Output dir:  $output_dir"
echo "  Processors:  $num_processors"
echo "  Plane range: $init_plane to $max_planes"
echo "  Force rerun: $force_run"
echo ""

# Function to run executable_df with existence check
run_df() {
    local snapshot=$1
    local plane=$2
    local out_dir=$3
    local force=$4
    local ini_path=$5

    local expected_output="${out_dir}flux.df.${snapshot}_${plane}.txt"

    if [ "$force" != "true" ] && [ -s "$expected_output" ]; then
        echo "[SKIP] Output file already exists: $expected_output"
        return 0
    fi

    echo "Processing snapshot: $snapshot with plane number: $plane"
    
    ./executable_df "${snapshot}" "${plane}" -ini "${ini_path}"
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error processing snapshot: $snapshot with plane number: $plane."
        return 1
    fi
    return 0
}

export -f run_df

# Generate pairs and run in parallel with GNU Parallel
for (( i = 0; i < $nl; i++ ))
do
    # Only process planes within range [init_plane, max_planes]
    if [[ "${planes[$i]}" -ge "$init_plane" && "${planes[$i]}" -le "$max_planes" ]];
    then
        echo "${snaps[$i]} ${planes[$i]} ${output_dir} ${force_run} ${ini_file}"
    fi        
done | parallel --bar --jobs "$num_processors" --colsep ' ' run_df {1} {2} {3} {4} {5}

