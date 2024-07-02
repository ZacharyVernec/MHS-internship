#!/bin/bash

#Iterate through each line in the names_mhs.txt file and execute the Python script
for i in $(seq 3 10); do 
    for j in $(seq 3 10); do 
        for k in $(seq 1 10); do
            if [ -s "spectras/$i-$j-$k-conflicts.txt" ]; then
                python QA_tests.py "$i-$j-$k"
            fi
        done
    done
done

# Exit with success
exit 0
