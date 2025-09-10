# Code for the paper *How to use the Sharpe ratio*

Generate the figures from the paper as follows: 

```bash
uv venv
uv pip install scikit-learn scipy matplotlib seaborn tqdm cvxpy ray papermill ipykernel ipywidgets
uv run functions.py  # Tests, and the numeric example from the paper
mkdir outputs
for notebook in *.ipynb
do
  uv run papermill "$notebook" outputs/"$notebook"
done
```
