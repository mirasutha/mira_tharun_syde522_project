SYDE 522 final project: Letter Recognition (UCI / Kaggle letter-recognition.csv)

paper.tex, neurips_2020.sty, table2.tex, figures/   -> compile with: pdflatex paper.tex (twice)
code/      scripts (run in order: phaseA, phaseB, phaseC, best, phaseE, phaseD, phaseF, build_table, figs)
           common.py reads the CSV from /mnt/user-data/uploads/letter-recognition.csv -- change DATA path if needed.
           (code/*.py expect to be run from a folder that has a results/ and figures/ sibling; ROOT is set in common.py.)
results/   raw sweep results (JSON) and final test predictions (npz)

Protocol: first 16,000 rows = training pool, last 4,000 = test (used once, in phaseD).
Tuning: fixed-seed random 12,000 train / 4,000 validation split of the pool.
