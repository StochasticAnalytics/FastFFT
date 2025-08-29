#!/bin/bash

# We need to do a few things to make cufftdx work with FastFFT
# 1. Replace all instances of tid.y with tid.z in the assembly code
#   - TODO: see X for explanation
# 2. Find the correct include directories

# include "../cufftdx/include/cufftdx/include/detail/system_checks.hpp"
# include "../cufftdx/include/cufftdx/include/cufftdx.hpp"
# include "../cufftdx/include/cufftdx/include/cufftdx.hpp"
# include "../cufftdx/include/cufftdx/include/operators/direction.hpp"
# include "../../include/cufftdx/include/cufftdx.hpp"
# include "../cufftdx/include/cufftdx/include/cufftdx.hpp"

# Incase you need to handle this in steps
DO_STEP_1_TIDY_TO_TIDZ=false
DO_STEP_2=true

# Make sure we are in the correct directory
if [[ $(ls -d ../../../FastFFT/include/nvidia-srcs/) == "../../../FastFFT/include/nvidia-srcs/" ]]; then
  echo "We are in the correct directory to run setup.sh"
else
  echo "We are not in the correct directory to run setup.sh, please move to FastFFT/include/nvidia-srcs"
  exit 1
fi


# Version info
using_version=nvidia-mathdx-24.08.0

# Check that the directory exists
if [ -d "$using_version" ]; then
    echo "Directory $using_version exists."
else
  echo "Directory $using_version does not exist."
  exit 1
fi

# Get the cufftdx.hpp include directory. There are two versions of this file, the outer one sources the inner, select the one closer to the project root
cufftdx_hpp_dir=$(dirname $(find ./$using_version -name cufftdx.hpp  -printf '%d %p\n' | sort -n | head -n1 | cut -d' ' -f2))
echo "Found cufftdx.hpp include directory at $cufftdx_hpp_dir"

# Now get the remaining include path from the include in this file
remaining_include_path=$(grep -oP '#include "\K[^"]+' $cufftdx_hpp_dir/cufftdx.hpp | xargs -I {} dirname {})
echo "Found remaining include path at $remaining_include_path"

# We'll record the appropriate include path which will be accessed in the build process
# The includes then look like <cufftdx/include/detail/system_checks.hpp>
header_path=${cufftdx_hpp_dir#./} 
echo "CPPFLAGS += -I../include/nvidia-srcs/${header_path}/" > ${using_version}.mk


# Get the path for the database of assembly we need to modify
database_path=$(find $cufftdx_hpp_dir -name database -print -quit)
if [ -z "$database_path" ]; then
    echo "Database path not found."
    exit 1
else
    echo "Found database path at $database_path"
fi

if [[ "x$DO_STEP_1_TIDY_TO_TIDZ" = "xtrue" ]]; then

    echo "Step 1: Replace tid.y to tid.z in assembly to enable efficient use of fft decomposotion algos"

    echo "Checking for tid.y or tid.z register usage in the assembly. There should be no tid.z and there should be tid.y"
    n_tidz=$(grep -R -c "tid.z" "$database_path" | awk -F: '{s+=$2} END{print s}')
    n_tidy=$(grep -R -c "tid.y" "$database_path" | awk -F: '{s+=$2} END{print s}')

    if [[ $n_tidz -ne 0 ]]; then
        echo "Found $n_tidz instances of tid.z"
        exit 1
    fi
    if [[ $n_tidy -eq 0 ]]; then
        echo "No instances of tid.y found"
        exit 1
    fi

    # Print tid.y and tid.z counts
    echo "Found $n_tidy instances of tid.y"
    echo "Found $n_tidz instances of tid.z"

    # Make a backup for the database if the don't exist
    if [  -d "$database_path.bak" ]; then
        echo "Warning: database backup exists, copy files there?"
        read -p "Do you want to copy files to the backup directory? (y/n) " choice
        if [[ $choice == [Yy] ]]; then
            cp -r "$database_path" "$database_path.bak"
        fi
    else
        cp -r "$database_path" "$database_path.bak"
        echo "Backup of database created at $database_path.bak"
    fi

    temp_file=$(tempfile)

    # Make sure the tempfile is made
    if [ ! -f "$temp_file" ]; then
        echo "Creating temporary file at $temp_file failed"
        exit 1
    fi

    # Make a trap to ensure we clean up
    trap 'rm -f "$temp_file"' EXIT


    # Now we'll swap all the tid.y for tid.z
    ls $database_path/*.inc | while read a ; do 
        awk '{gsub("tid.y","tid.z")}{print$0}' $a > $temp_file ; mv $temp_file $a ; done

    # Now check on tid.y and tid.z
    n_tidy_new=$(grep -R -c "tid.y" "$database_path" | awk -F: '{s+=$2} END{print s}')
    n_tidz_new=$(grep -R -c "tid.z" "$database_path" | awk -F: '{s+=$2} END{print s}')

    # print tid.y and tid.z counts
    echo "Found $n_tidy_new instances of tid.y"
    echo "Found $n_tidz_new instances of tid.z"

    # Remove the exit statement to continue with validation

    if [[ $n_tidz_new -ne $n_tidy ]] ; then
        echo "Error: number of tid.z ($n_tidz_new) does not match original number of tid.y ($n_tidy)"
        echo "Restoring from backup"
        rm -r "$database_path"/*.inc
        cp -r "$database_path.bak"/*.inc "$database_path"
        exit 1
    else
        echo "Success: number of tid.z ($n_tidz_new) matches original number of tid.y ($n_tidy)"
    fi

    if [[ $n_tidy_new -ne 0 ]]; then
        echo "Error: found $n_tidy_new instances of tid.y after replacement"
        echo "Restoring from backup"
        rm -r "$database_path"/*.inc
        cp -r "$database_path.bak"/*.inc "$database_path"
        exit 1
    else
        echo "Success: no instances of tid.y found after replacement"
    fi
else
    echo "Skipping Step 1: Replace tid.y to tid.z in assembly to enable efficient use of fft decomposotion algos"
fi

if [[ "x$DO_STEP_2" == "xtrue" ]]; then
    echo "Step 2: Replace tid.z to tid.y in assembly to enable efficient use of fft decomposotion algos"
    exit
fi

# WE need to modify
/cufftdx/include/cufftdx/include/detail
    - fft_execution.hpp: we need to make this work to default forcing one fft_per_block and ensure the shared memory is not cooped by cufftdx routines
        - shared_to_registers_impl.hpp
        - registers_to_shared_impl.hpp
            - to both add a template parameter <fft_per_block> as the first param