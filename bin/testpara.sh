COUNTER=0
for dataname in whole rss aoa toa data_4 data_5 data_6
do
    for ncomponents in $(seq 2 2 20)
    do
        for batch in $(seq 5 5 30)
        do
            for early_epoch in $(seq 20 20 100)
            do
                for logsigmamin in $(seq -6 -1) 
                do
                    for logsigmamax in $(seq 0 5)
                    do
                        for nhidden1 in $(seq 20 2 30)
                        do
                            nhidden2=$[$nhidden1-2]
                            nhidden3=$[$nhidden1-4]
                            COUNTER=$[$COUNTER +1]           
                        done
                    done
                done
            done
        done
    done
done
echo $COUNTER
