#!/bin/bash
cd /home/claude/proj
python3 phaseB.py > results/logB.txt 2>&1
python3 phaseC.py > results/logC.txt 2>&1
python3 best.py > results/logBest.txt 2>&1
python3 phaseE.py > results/logE.txt 2>&1
python3 phaseD.py > results/logD.txt 2>&1
echo done > results/done_all.txt
