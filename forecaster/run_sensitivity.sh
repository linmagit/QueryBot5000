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


for RHO in '0.55' '0.65' '0.75' '0.85' '0.95'; do
for HORIZON in '60' '1440'; do
for PROJECT in 'tiramisu' 'oli' 'admission'; do
    for METHOD in 'ar'; do
        cmd="time python3.5 exp_multi_online_continuous.py $PROJECT --method $METHOD --aggregate 60
            --horizon $HORIZON
            --input_dir ~/peloton-tf/time-series-clustering/online-clusters-sensitivity/$PROJECT/$RHO/
            --cluster_path ~/peloton-tf/time-series-clustering/cluster-coverage-sensitivity/$PROJECT/$RHO/coverage.pickle
            --output_dir ../prediction-sensitivity-result/$PROJECT/$RHO/"

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

