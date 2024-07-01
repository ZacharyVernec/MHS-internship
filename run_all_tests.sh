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

# Remove the names_mhs.txt file
rm names_mhs.txt

# Exit with success
exit 0
