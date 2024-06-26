#!/bin/bash

# Generate names_mhs.txt dynamically
python3 - <<EOF
all_names_MHS = []

for i in range(3, 11):
    for j in range(3, 11):
        for k in range(1, 11):
            all_names_MHS.append(f"{i}-{j}-{k}")

# Save the names to a file
with open('names_mhs.txt', 'w') as file:
    for name in all_names_MHS:
        file.write(name + '\n')

print("Names saved to names_mhs.txt")
EOF

# Ensure the names_mhs.txt file exists
if [ ! -f names_mhs.txt ]; then
    echo "File names_mhs.txt not found!"
    exit 1
fi

# Iterate through each line in the names_mhs.txt file and execute the Python script
while IFS= read -r name; do
    if [ -s "spectras/${name}-conflicts.txt" ]; then
        python3 QA_tests.py "$name"
    fi
done < names_mhs.txt

# Compute average ratios for each qubit size
for qubit_size in {3..10}; do
    results_file="${qubit_size}-qubit-results.txt"
    
    if [ -f "$results_file" ]; then
        total=0
        count=0
        while read -r line; do
            ratio=$(echo $line | awk '{print $2}' | sed 's/%//')
            total=$(echo "$total + $ratio" | bc)
            count=$((count + 1))
        done < "$results_file"

        if [ $count -ne 0 ]; then
            average=$(echo "$total / $count" | bc -l)
            echo "Average ratio: $average" > "$average_file"
        else
            echo "No valid test cases found for qubit size $qubit_size" > "$average_file"
        fi
    else
        echo "No results file found for qubit size $qubit_size" > "$average_file"
    fi
done
