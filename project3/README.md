# Fama-French Factor Model + Green Factor

## Research Question
Does a Green-Minus-Brown factor explain returns beyond Fama-French 3 factors?

## Methodology
**Language:** Python  
**Methods:** Time-series factor regressions, GRS test

## Data
Kenneth French Data Library, Yahoo Finance ETFs (ICLN, QCLN, PBW, XLE, XOP, VDE)

## Key Findings
GMB factor is highly significant for both green and brown ETFs; low correlation with traditional FF3 factors.

## How to Run
```bash
pip install -r requirements.txt
python code/project3_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
