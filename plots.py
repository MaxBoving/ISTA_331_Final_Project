'''

Max Boving

10/07/2025

Final Project plots

ISTA 331
'''
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans2

#preproccessing
master = pd.read_csv('insurance.csv')
master['bmi'] = master['bmi'].round(2)
master['charges'] = master['charges'].round(2)
master

#figure 1 : Smoker vs Charges boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=master, x='smoker', y='charges')
plt.title('Health insurance charges for Smokers vs Non-Smokers')
plt.xlabel('Smoker Status')
plt.ylabel('Charges in USD$')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# figure 2 : subplot depicting LBF for BMI vs Charge per female/male colored by smoker
master_encoded = pd.get_dummies(master, columns=['sex', 'smoker'], drop_first=True)
X = master_encoded[['age', 'bmi', 'sex_male', 'smoker_yes']]
y = master_encoded['charges']

X['sex_male'] = X['sex_male'].astype(int)
X['smoker_yes'] = X['smoker_yes'].astype(int)

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

g = sns.lmplot(data=master, x='bmi', y='charges', hue='smoker', col='sex', height=5, aspect=0.8)
g.set_axis_labels("BMI", "Charges in $USD")
plt.show()


# figure 3 : Kmeans visualization 
Xk = master_encoded[['bmi', 'smoker_yes', 'charges']].to_numpy()
X_scaled = StandardScaler().fit_transform(Xk)
# 2 k for smoker - non smoker
centroids, labels = kmeans2(X_scaled, k=2, minit='++', iter=100)  

plot_df = master_encoded.assign(cluster=labels)
# plot 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x='bmi', y='charges', hue='cluster', palette='plasma')
plt.title('K-Means on Scaled Features (visualized in bmi vs charges)')
plt.grid(True)
plt.show()


# figure 4 : Table of Kmeans labels 

# get our summary: mean bmi, smoker, mean charges. frame index is each cluster
summary = plot_df.groupby('cluster')[['bmi','smoker_yes','charges']].mean().round(2)

fig, ax = plt.subplots(figsize=(10,6))
ax.axis('off')
table = ax.table(cellText=summary.values,
                 colLabels=summary.columns,
                 rowLabels=[f'Cluster {i}' for i in summary.index],
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.0)
plt.title("Cluster Summary Statistics")
plt.savefig("cluster_summary.png", dpi=300, bbox_inches='tight')
plt.show()
