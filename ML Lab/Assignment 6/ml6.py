
import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# df = pd.read_csv('Market_Basket_Optimisation.csv')
dataset = []
with open('Market_Basket_Optimisation.csv') as file:
  reader = csv.reader(file, delimiter=',')
  for row in reader:
    dataset += [row]

dataset[1:10]

len(dataset)

te = TransactionEncoder()
x = te.fit_transform(dataset)

x

df = pd.DataFrame(x, columns=te.columns_)

len(te.columns_)

df.head()

"""# Apriori Algorithm


"""

# find frequent itemsets
freq_itemset = apriori(df, min_support=0.1, use_colnames=True)

freq_itemset

# finding the rules
rules = association_rules(freq_itemset, metric='confidence', min_threshold=0.25)