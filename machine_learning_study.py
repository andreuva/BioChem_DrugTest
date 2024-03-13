import pandas as pd
path = 'data/all_metrics.csv'
df = pd.read_csv(path,index_col=0)
print(df)