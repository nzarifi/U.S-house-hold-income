# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:40:39 2019

@author: Niloofar-Z
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Niloofar-Z\Desktop\MetroCo\datamining\us-household-income')


df = pd.read_csv("kaggle_income.csv",encoding = "ISO-8859-1")        
df.columns.values

df.describe()
df.isnull().sum()

#keep County 
df.place.value_counts()
df.Place.nunique()
df.County.nunique()
df.City.nunique()

##keep type and Primary
df.Type.value_counts()
df.Primary.value_counts()

#get some idea 
df.groupby(by='State_Name', as_index=False).agg({'County': pd.Series.nunique})
df.groupby('State_Name')['County'].nunique()
df.County.groupby([df.State_Name.str.strip("'")]).nunique()
df.groupby('State_Name')['County'].nunique()
df.groupby(['State_Name', 'County']).count()


df.drop(columns=['id','State_Code','State_Name','City','Place',
                 'Area_Code', 'Zip_Code','ALand','AWater',
                 'Lat', 'Lon'],axis=1, inplace=True)


df=pd.get_dummies(df,columns=['State_ab','County','Type','Primary'],drop_first=True)
df.to_csv("cleaned-income-U.S.csv",index=False)

df = pd.read_csv("cleaned-income-U.S.csv")
df.head()



Mean=df['Mean']
n, bins, patches = plt.hist(x=Mean, bins=20,color='#aa0504',

                            alpha=0.75, rwidth=0.85)

plt.grid(axis='x', alpha=0.2)
plt.grid(axis='y', alpha=0.75)
#plt.xticks(np.arange(0.5, 0.7, 0.005))
plt.xlabel('mean of income',fontsize=12)
plt.ylabel('count',fontsize=12)
plt.title('mean household income of the specified geographic location')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig('foo.png',
            bbox_inches='tight',dpi=600)
plt.show()

Median=df['Median']
n, bins, patches = plt.hist(x=Median,color='b',bins=10, alpha=0.75, rwidth=0.85)

plt.grid(axis='x', alpha=0.2)
plt.grid(axis='y', alpha=0.75)
#plt.xticks(np.arange(0.5, 0.7, 0.005))
plt.xlabel('Median of income',fontsize=12)
plt.ylabel('count',fontsize=12)
plt.title('Median household income of the specified geographic location')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig('foo.png',
            bbox_inches='tight',dpi=600)
plt.show()

##decision Low <30000 , 30000<mid<100000 , else: high

df['target_mean']=df['Mean'].apply(lambda x: "low" if x<=30000 else "mid" if 30000<x<=100000 else "high")
df['target_median']=df['Median'].apply(lambda x: "low" if x<=30000 else "mid" if 30000<x<=100000 else "high")

df.to_csv("cleaned-income.csv",index=False)
#Max value is very different how you explain it?
df['Median'].describe()
df['Mean'].describe()
