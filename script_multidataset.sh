#!/bin/bash

# This script is responsible to run the 8 datasets at the same time to maximize eficiency
# It works by giving each dataset as a argument for the script

SCRIPT_TO_RUN = "./script_fs_hier_multidataset.py"

ARGS = ("EC-Interpro" "EC-Pfam" "EC-Prints" "EC-Prosite" "GCPR-Interpro" "GCPR-Pfam" "GCPR-Prints" "GCPR-Prosite")

# Change to your terminal, konsole, xterm...
for i in "${!ARGS[@]}"; do
    gnome-terminal -- bash -c "$SCRIPT_TO_RUN ${ARGS[$i]}; exec bash"
done
