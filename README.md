# Code for the paper *[Sharpe Ratio Inference: A New Standard for Reporting and Decision-Making](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5520741)*

Generate the figures from the paper as follows: 

```bash
uv venv
uv pip install scikit-learn scipy statsmodels matplotlib seaborn tqdm cvxpy ray deprecated papermill ipykernel ipywidgets
uv run functions.py  # Tests, and the numeric example from the paper
mkdir outputs
for notebook in *.ipynb
do
  uv run papermill "$notebook" outputs/"$notebook"
done
```
