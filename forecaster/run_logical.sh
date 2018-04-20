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

# remove the log file
if [ -f run.log ] ; then
    rm run.log
fi


for RHO in '0.1' '0.2' '0.3' '0.4'; do
for HORIZON in '60' '1440' '10080'; do
for PROJECT in 'admission'; do
    for METHOD in 'ar'; do
        cmd="time python3.5 exp_multi_online_continuous.py $PROJECT --method $METHOD --aggregate 60
            --horizon $HORIZON
            --input_dir ~/peloton-tf/time-series-clustering/online-clusters-logical/$PROJECT/$RHO/
            --cluster_path ~/peloton-tf/time-series-clustering/cluster-coverage-logical/$PROJECT/$RHO/coverage.pickle
            --output_dir ../prediction-logical-result/$PROJECT/$RHO/"

        echo $cmd
        echo $cmd >> run.log
        START=$(date +%s)

        eval $cmd &

        END=$(date +%s)
        DIFF=$(( $END - $START ))
        echo "Execution time: $DIFF seconds"
        echo -e "Execution time: $DIFF seconds\n" >> run.log

    done # METHOD
done # PROJECT
done # HORIZON 
done # RHO

