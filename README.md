# Feature Development 101

## Files

### Notebooks
- `DataCleaningExploration.ipynb`: 
    - Conducts preliminary cleaning, preprocessing, and data exploration
    - Writes a cleaned data file - `data_cleaned.csv` to the `/data` folder, which is used for subsequent analysis
- `MerchantTypes.ipynb`: 
    - Identifies different types of merchants based on transaction activity
    - Writes assigned merchant categories - `merchant_types.csv` to the `output/` folder
- `Churn.ipynb`: 
    - Defines churn
    - Writes a list of churned merchants - `churned_merchants.csv` to the `output/` folder
- `Modeling.ipynb`: 
    - Predicts churn
    - Contains label generation, feature engineering, modeling, interpretability, and error analysis
    - Writes merchants at risk of churn - `merchant_churn_likelihood.csv` to the `output/` folder

If you'd prefer to just view these notebooks, they're all available under the same names in the `rendered_notebooks/` folder, with the `.html` extension.

### Output Files
- Stored in the `output/` folder
- `merchant_types.csv` - merchant types based on transaction activity
- `churned_merchants.csv` - merchants who have churned as of the end date of the dataset
- `merchant_churn_likelihood.csv` - merchants active over the last three months of the dataset at risk of churning in the three months following the end of the dataset

### Utility Functionss
- Defined in the `lib/` folder
- Meant to make experimentation easier
- **Not** production ready code - suboptimal abstractions, and don't have logging, error handling, or tests

### Data
- Stored in the `data/` folder
- `data/takehome_ds_written - takehome_ds_written.csv` - assignment data file
- `data_cleaned.csv` - cleaned data file


## Focus
Exploration/exposition/modeling as opposed to writing production ready code

## Usage
If you'd like to run the code yourself
```bash
# Install pipenv
pip3 install --user pipenv

# cd into project folder
cd feature_dev_101

# install environment
pipenv install --ignore-pipfile

# activate environment
pipenv shell

# launch jupyter lab
jupyter lab

# deactivate environment when done
deactivate
```
