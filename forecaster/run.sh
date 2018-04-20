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

#for AGGREGATE in '10' '20' '30' '60' '120'; do
for AGGREGATE in '60'; do
#for AGGREGATE in '1' '5' '10' '30' '60' '120'; do
#for HORIZON in '720'; do
#for HORIZON in '60' '2880' '4320' '10080'; do
for HORIZON in '720' '1440' '7200'; do
#for HORIZON in '60' '720' '1440' '2880' '4320' '7200' '10080'; do
for PROJECT in 'admission'; do
#for PROJECT in 'tiramisu' 'oli' 'admission'; do
    #for METHOD in 'kr'; do
    for METHOD in 'arma' 'ar' 'kr' 'fnn' 'rnn' 'psrnn'; do
        cmd="time python3.5 exp_multi_online_continuous.py $PROJECT
            --method $METHOD
            --aggregate $AGGREGATE
            --horizon $HORIZON"

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

