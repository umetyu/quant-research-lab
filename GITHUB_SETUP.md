# GitHub Setup Guide

## 1. Create a new repository

Recommended repository name:

```text
quant-research-lab
```

Description:

```text
Reproducible systematic trading research project with walk-forward validation, ML direction signals, and transaction-cost-aware backtesting.
```

## 2. Upload from terminal

```bash
cd quant-research-lab
git init
git add .
git commit -m "Initial quant research lab"
git branch -M main
git remote add origin https://github.com/<your-username>/quant-research-lab.git
git push -u origin main
```

## 3. What not to upload

Do not upload the original course notebooks if they contain redistribution restrictions.

This repository is intentionally rewritten as an original public project using the same general research themes:
feature engineering, labeling, walk-forward validation, backtesting, and risk metrics.
