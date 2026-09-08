#!/bin/bash

help() {
    cat << EOF

          ~~~~~  FORECAST DF module batch processing script  ~~~~~

This script runs the DF module for multiple planes in parallel using GNU parallel.
It reads the plane numbers and their corresponding snapshot numbers from planes_list.txt,
creates .d files for each plane, and then runs the executable_df for each plane-snapshot pair in parallel.

Usage: ./df.sh [options]

Options:
  -n: Number of processors to use (default: 1)
  -i: Initial plane number (default: 0)
  -lc: Path to the planes_list.txt file (default: ../lc/planes_list.txt)
  -h, --help: Show this help message and exit


Example usage:
  ./df.sh -n 4 -i 36 -lc ../lc/planes_list.txt

EOF
}



num_processors=1  # Default number of processors
init_plane=0     # Default initial plane number
planes_list_path="../lc/planes_list.txt"  # Default path to planes_list.txt

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

nl=$(cat "$planes_list_path" | wc -l)
declare -a planes
declare -a snaps
for i in $(seq 0 $((nl-1)))
do
    #plane number - read line i+1 since seq starts from 0 but awk NR starts from 1
    snaps[i]="$(cat "$planes_list_path" | awk -v p="$((i+1))" '{if(NR==p) print $1}')"
    #snapshot number
    planes[i]="$(cat "$planes_list_path" | awk -v p="$((i+1))" '{if(NR==p) print $6}')"
done

max_planes=$(printf "%s\n" "${planes[@]}" | sort -nr | head -n 1)

echo "Starting batch processing with $num_processors processors"
echo "Processing planes $init_plane to $max_planes"


# Function to run executable_df
run_df() {
    local plane=$1
    local snapshot=$2
    local plane_file=$3

    echo "Processing plane number: $plane with snapshot: $snapshot using file: $plane_file"
    
    ./executable_df ${snapshot} ${plane} ${plane_file}
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error processing plane number: $plane."
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
        plane_num="${planes[$i]}"
        echo "${snaps[$i]} ${plane_num} ${planes_list_path}" 
    fi        
done | parallel --bar --jobs $num_processors --colsep ' ' run_df {1} {2} {3}

