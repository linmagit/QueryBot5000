#########################################################################
# File Name: run.sh
# Author: Lin Ma
# mail: malin1993ml@gmail.com
# Created Time: 10/08/17
#########################################################################
#!/bin/bash

trap onexit 1 2 3 15
function onexit() {
    local exit_status=${1:-$?}
    pkill -f hstore.tag
    exit $exit_status
}

# ---------------------------------------------------------------------

# remove the log file
if [ -f run.log ] ; then
    rm run.log
fi

PROJECT_ARRAY=( "tiramisu:tiramisu-combined-results"
        "oli:oli-combined-results"
        "admission:admission-combined-results" )

for PAIR in "${PROJECT_ARRAY[@]}"; do
    PROJECT="${PAIR%%:*}"
    DATA_PATH="${PAIR##*:}"
    for RHO in '0.55' '0.65' '0.75' '0.85' '0.95'; do
        cmd="time python3.5 online_clustering.py --project $PROJECT --dir $DATA_PATH
                --rho $RHO"

        echo $cmd
        echo $cmd >> run.log
        START=$(date +%s)

        eval $cmd &

        END=$(date +%s)
        DIFF=$(( $END - $START ))
        echo "Execution time: $DIFF seconds"
        echo -e "Execution time: $DIFF seconds\n" >> run.log

    done # RHO
done # PROJECT

