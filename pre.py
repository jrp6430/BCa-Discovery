from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# pipeline for processing high-throughput sequencing data of groups
# input is a merged dataframe of the raw group data
# output is a dataframe with missing values imputed, log base 10 transformation applied, and uninformative genes removed

def prep_data(df):
    # extract the number of observations in the dataframe for later use
    n = len(df)

    # first, replace NaN values in the raw data with zeros
    filled_df = df.fillna(0)

    # then, use a log base 10 transformation on the dataset
    # however, since we are imputing missing values with zeros, their log value is -inf
    # replace these with zero after applying the transformation
    log_transformed_df = filled_df.apply(np.log10).replace(-np.inf, 0)

    # double check that there are no NaN values in the data after imputation and log scaling
    # if so replace them with zero
    if log_transformed_df.isnull().any().any():
        log_transformed_df.fillna(0, inplace=True)

    # Now, remove uninformative genes, where the value is zero in 80% of observations (NaN or no expression)
    # do this by using a boolean mask looking for values equal to zero, convert it to an integer,
    # then sum for each gene and divide by the total number of observations (n)
    # if the percentage of zeros in a gene is greater than 80%, drop it from the dataframe

    percent_zeros_series = (log_transformed_df == 0).astype(int).sum(axis=0) / n
    uninformative_genes = percent_zeros_series[percent_zeros_series > 0.8].index
    log_transformed_df.drop(uninformative_genes, axis=1, inplace=True)

    # as a final step in preprocessing, restrict the number of gene variables to 10000 on the basis of variance
    final_df = log_transformed_df.drop(log_transformed_df.var().sort_values(ascending=False)[10000:].index, axis=1)

    return final_df


# function to execute optimized random forest technique for feature selection
def optimized_rf_gene_select(df, target, iter=1000):
    # initialize dictionary where number of times each gene has appeared in the RF top 100 is stored
    genes = pd.Series()

    # repeat the random forest algorithm for the desired number of iteration
    for i in range(iter):
        print("Iteration no.", i)
        # instantiate RF classifier, initialize its random state
        # default parameters used: n_estimators=100, bootstrap=True, criterion=gini, max_feats=sqrt
        # then fit it to the gene expression data
        rf = RandomForestClassifier(random_state=i, n_jobs=-1)
        rf.fit(df, target)

        # extract feature importances from the RF model, sort from highest to lowest
        rf_importances = pd.Series(rf.feature_importances_, index=df.columns).sort_values(ascending=False)

        # extract the names of the top 100 genes and put them in a pandas Series, then append it to the genes Series
        genes = pd.concat([genes, pd.Series(rf_importances.iloc[:100].index)])

    # the genes Series will contain all 100,000 genes selected in the 1000 iterations of RF
    # use value_counts to tally up all of them and sort from greatest to least
    tally_genes = genes.value_counts().index.to_list()

    # then transform dataframe to include the top 50, 30, and 10 of the genes from the optimized RF.
    # store these in a dictionary with keys corresponding to top n
    top50_df = df[tally_genes[:50]]
    top30_df = df[tally_genes[:30]]
    top10_df = df[tally_genes[:10]]
    top_rf_dfs = dict()
    top_rf_dfs[50] = top50_df
    top_rf_dfs[30] = top30_df
    top_rf_dfs[10] = top10_df

    return top_rf_dfs
