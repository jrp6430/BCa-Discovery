import pandas as pd
from pre import *
from eval import *
import time


# first, read in TSV file of RNA seq data with pandas function, specify delimiter as tab, genes as columns
df = pd.read_csv(r"E:\Users\joepi\gene_tsv\human_liver.tsv", sep='\t', index_col='genes').T

# randomly generate a series of length N of Male/Female labels, where 1 = Male, 0 = female (has or does not have Y
# chromosome)
n = len(df)
targets = pd.Series(np.random.choice([1, 0], size=n))

# impute NaN with zero, log10 transform, remove genes with low expression, variance across all samples
clean_df = prep_data(df)

# use optimized RF procedure to select the top 10, 30, and 50 genes relevant to the classification task
dfs = optimized_rf_gene_select(clean_df, targets, iter=60)

# score using RF, SVM, and KNN for each cohort of selected genes
top_n_eval(dfs, targets)





