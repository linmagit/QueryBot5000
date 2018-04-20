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

for PROJECT in 'admission'; do
    for RHO in '0.1' '0.2' '0.3' '0.4'; do
        cmd="time python3.5 generate-cluster-coverage.py
                --project $PROJECT
                --assignment online-logical-clustering-results/$PROJECT-$RHO-assignments.pickle
                --output_csv_dir online-clusters-logical/$PROJECT/$RHO/
                --output_dir cluster-coverage-logical/$PROJECT/$RHO/"

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

