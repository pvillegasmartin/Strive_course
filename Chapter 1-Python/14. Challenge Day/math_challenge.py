import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




#PCA Exercise
digits = load_digits()
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
digits_df = pd.DataFrame({"x":proj[:, 0], "y":proj[:, 1], "label":digits.target})
df_numbers = digits_df.groupby('label').mean()
df_numbers['label']=df_numbers.index

#Graph 1 point per label
fig_numbers = plt.figure(figsize=(5,3))
ax = fig_numbers.add_subplot(1,1,1)
plt.xlim(-40,40)
plt.xlabel("x")
plt.ylabel("y")
ylabels = range(-15,25,5)
plt.yticks(ylabels)
for i in df_numbers.index.values.tolist():
    plt.scatter(df_numbers.loc[i:i,'x'],df_numbers.loc[i:i,'y'], marker='o', label=i)
plt.legend()

#IMPROVEMENT
df_dist = pd.DataFrame(columns=['filter','from','dist'])
print(df_dist)
for i in df_numbers.index.values.tolist():
    for j in df_numbers.index.values.tolist():
        df_dist = df_dist.append({'filter':int(i),'from':int(j),'dist':np.sqrt((df_numbers.loc[i,'x']-df_numbers.loc[j,'x'])**2 + (df_numbers.loc[i,'y']-df_numbers.loc[j,'y'])**2)}, ignore_index=True)
        df_dist['filter'] = df_dist['filter'].apply(np.int64)
        df_dist['from'] = df_dist['from'].apply(np.int64)

#STREAMLIT CODE


st.set_page_config(layout="wide")

st.write("""
# PCA
Showing **chart**
""")

st.sidebar.header('Some filters')



col1, col2 = st.columns((2,1.5))
col1.title("Graph")
col1.pyplot(fig_numbers)
col2.title("Similar numbers")
number_studied = col2.selectbox('Select a number',df_numbers['label'].unique())
data_order_numbers = pd.Series(df_dist[(df_dist['filter'] == number_studied) & (df_dist['from'] != number_studied)].sort_values(by='dist', ascending=True)['from'])
col2.write(data_order_numbers.values)