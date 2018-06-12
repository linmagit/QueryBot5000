#########################################################################
# File Name: run.sh
# Author: Lin Ma
# mail: malin1993ml@gmail.com
# Created Time: 07/09/17
#########################################################################
#!/bin/bash

trap onexit 1 2 3 15
function onexit() {
    local exit_status=${1:-$?}
    pkill -f hstore.tag
    exit $exit_status
}

# ---------------------------------------------------------------------

log_name="run.log"

# remove the log file
if [ -f $log_name ] ; then
    rm $log_name
fi

for AGGREGATE in '60'; do
for HORIZON in '4320'; do
for PROJECT in 'tiramisu'; do
    for METHOD in 'ar' 'kr' 'rnn'; do
        cmd="time python3.5 forecaster/exp_multi_online_continuous.py $PROJECT
            --method $METHOD
            --aggregate $AGGREGATE
            --horizon $HORIZON
            --input_dir online-clusters/
            --cluster_path cluster-coverage/coverage.pickle
            --output_dir prediction-results/"

        echo $cmd
        echo $cmd >> $log_name
        START=$(date +%s)

        eval $cmd

        END=$(date +%s)
        DIFF=$(( $END - $START ))
        echo "Execution time: $DIFF seconds"
        echo -e "Execution time: $DIFF seconds\n" >> $log_name

    done # METHOD
done # PROJECT
done # HORIZON 
done # AGGREGATE

