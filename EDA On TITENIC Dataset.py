#!/usr/bin/env python
# coding: utf-8

# # Performing EDA on TITENIC Dataset.

# # Problem statement.

# # "Can you perform EDA and predict whether a passenger will survive the Titanic disaster based on various features such as age, gender, passenger class, family size, fare, and embarkation port?"

# In[1]:


# import python libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[108]:


#firstly import dataset.
df = pd.read_csv('train.csv')
df


# In[109]:


df.tail()


# In[110]:


df['Name'].count()


# # Why do EDA?
Model building
Analysis and reporting
Validate assumptions
Handling missing values
feature engineering
detecting outliers
# # Perform Univariate Analysis

# # AGE
conclusions:

1.Age is normally(almost) distributed.
2.20% of the values are missing.
3.There are some outliers.
# In[111]:


# age column, here we gonna to perform some Descriptive statatics. (mean, meadian and mode)

df['Age'].describe()


# In[112]:


df['Age'].plot(kind='hist', bins=30)


# In[113]:


df['Age'].plot(kind='kde') #kde graph-------> its give the distribution of data


# In[114]:


df["Age"].skew() #its very close to the the zero. no more -ve skews and no more +ve skews.


# In[10]:


df['Age'].plot(kind='box') #its is very imortants


# In[115]:


df[df['Age']>65]


# In[116]:


df['Age'].isnull().sum() #its give the missing value.


# In[117]:


len(df['Age']) #total no of passanger in age column


# In[118]:


df['Age'].isnull().sum()/len(df['Age'])*100


# In[15]:


# here around 19% missing value.


# # Fare
conclusions

1.The data is highly(positively) skewed.
2.Fare col actually contains the group fare and not the individual fare(This migth be and issue).
3.We need to create a new col called individual fare.
# In[16]:


df['Fare']


# In[17]:


df['Fare'].describe()


# In[18]:


df['Fare'].skew() #highly positive skews


# In[19]:


df['Fare'].plot(kind='kde')


# In[20]:


df['Fare'].plot(kind='hist')


# In[21]:


df['Fare'].isnull().sum()


# In[22]:


len(df['Fare'])


# In[23]:


df['Fare'].head(20).plot(kind='pie')


# In[24]:


df['Fare'].plot(kind='box')


# In[25]:


df[df['Fare']>265]


# In[26]:


len(df[df['Fare']>265])


# In[119]:


df['Fare'].isnull().sum()


# # Univariate Analysis on Categorical columns

# # servived
conclusions

1.Parch and SibSp cols can be merged to form a new col call family_size.
2.Create a new col called is_alone.
# In[27]:


df['Survived'].value_counts()


# In[28]:


df['Survived'].value_counts().plot(kind='bar')


# In[29]:


df['Survived'].value_counts().plot(kind='pie',autopct="%0.1f%%")


# In[31]:


df['Survived'].value_counts()


# In[32]:


df['Survived'].value_counts().plot(kind='bar')


# In[33]:


df['Survived'].value_counts().plot(kind='pie',autopct='%0.1f%%') #important


# In[34]:


df["Pclass"].value_counts()


# In[35]:


df["Pclass"].value_counts().plot(kind='bar')


# In[36]:


df['Pclass'].value_counts().plot(kind='pie',autopct='%0.1f%%') #important


# In[37]:


df["Sex"].value_counts()


# In[38]:


df["Sex"].value_counts().plot(kind='bar')


# In[39]:


df['Sex'].value_counts().plot(kind='pie',autopct='%0.1f%%') #important


# In[40]:


df['SibSp'].value_counts()


# In[41]:


df['SibSp'].value_counts().plot(kind='bar')


# In[42]:


df['SibSp'].value_counts().plot(kind='pie')


# In[43]:


df['SibSp'].value_counts().plot(kind='pie',autopct='%0.1f%%') #important


# In[44]:


df['Parch'].value_counts()


# In[45]:


df['Parch'].value_counbts().plot(kind='pie',autopct='%0.1f%%') #important


# In[120]:


df['Sex'].isnull().sum()


# In[46]:


# name, tickts , and cabin these all are mixed column.


# In[47]:


df


# # Bivariate Analysis

# 

# In[48]:


# select two column.


# In[49]:


pd.crosstab(df['Survived'],df['Pclass'], normalize='columns')*100 # 0----> died   and 1----> not died


# In[50]:


sns.heatmap(pd.crosstab(df['Survived'],df['Pclass'], normalize='columns')*100)


# In[51]:


# travelling in pclass 3 is more dengeourus as compare to others


# In[52]:


pd.crosstab(df['Survived'],df['Sex'], normalize='columns')*100


# In[53]:


pd.crosstab(df['Survived'],df['Embarked'],normalize='columns')*100


# In[54]:


# here we have two assumption 
# 1. female is more servival as compare to the male passangers
# 2.the passanger pickup the titenic from Cherbourg is higher chance to servive.


# In[55]:


pd.crosstab(df['Sex'],df['Embarked'],normalize='columns')*100


# In[56]:


pd.crosstab(df['Pclass'],df['Embarked'],normalize='columns')*100 #  Cherbourg se
# ameer log chade hai.


# In[ ]:





# In[57]:


# servived vs age


# In[58]:


df[df['Survived']==1]['Age'].plot(kind='kde')


# In[59]:


df[df['Survived'] == 0]['Age'].plot(kind='kde',label='Not Survived')


# In[60]:


df[df['Survived'] == 1]['Age'].plot(kind='kde',label='Survived')
df[df['Survived'] == 0]['Age'].plot(kind='kde',label='Not Survived')

plt.legend()
plt.show()


# In[61]:


df[df['Pclass'] == 1]['Age'].mean()


# In[62]:


df


# # Feature Engineering on 'Fare' column

# In[63]:


df['SibSp'].value_counts()


# In[64]:


df[df['SibSp']==8]


# In[65]:


# there is one more dataset.
df1=pd.read_csv('test.csv')


# In[66]:


df1


# In[67]:


df = pd.concat([df,df1])


# In[68]:


df


# In[69]:


df[df['Ticket'] == 'CA. 2343']


# In[70]:


# we can check the members by ticket number.
df[df['Ticket'] == 'CA 2144']


# In[71]:


df["Ticket"].value_counts()


# In[72]:


df[df['Ticket'] == 'PC 17608']


# # Finding Indivisual column

# In[73]:


df['individual_fare'] = df['Fare']/(df['SibSp'] + df['Parch'] + 1) #try to finding indivisual fare.


# In[74]:


df['individual_fare'].plot(kind='box')


# In[75]:


df[['individual_fare','Fare']].describe()


# In[76]:


# now we have individual fare.
df['Fare']


# In[77]:


df


# In[78]:


# creating a new column name of "family size"


# In[79]:


df['family_size'] = df['SibSp'] + df['Parch'] + 1


# In[80]:


# family_type
# 1 -> alone
# 2-4 -> small
# >5 -> large
# ----------------------------

def transform_family_size(num):

  if num == 1:
    return 'alone'
  elif num>1 and num <5:
    return "small"
  else:
    return "large"


# In[81]:


df['family_size'].apply(transform_family_size)


# In[82]:


df['family_size']=df['family_size'].apply(transform_family_size)


# In[83]:


df


# # Creation of new column which is predict the your servival chances.

# In[84]:


pd.crosstab(df['Survived'],df['family_size'],normalize="columns")*100


# # Creation of new column of surname

# In[85]:


df['Name'].str.split(',').str.get(0)


# In[86]:


df


# In[87]:


df["Name"]


# # Extrect the title of name

# In[88]:


df['title'] = df['Name'].str.split(',').str.get(1).str.strip().str.split(' ').str.get(0)


# In[89]:


temp_df = df[df['title'].isin(['Mr.','Miss.','Mrs.','Master.','ootherr'])]


# In[90]:


pd.crosstab(temp_df['Survived'],temp_df['title'],normalize='columns')*100


# In[91]:


df['title'] = df['title'].str.replace('Rev.','other')
df['title'] = df['title'].str.replace('Dr.','other')
df['title'] = df['title'].str.replace('Col.','other')
df['title'] = df['title'].str.replace('Major.','other')
df['title'] = df['title'].str.replace('Capt.','other')
df['title'] = df['title'].str.replace('the','other')
df['title'] = df['title'].str.replace('Jonkheer.','other')
# ,'Dr.','Col.','Major.','Don.','Capt.','the','Jonkheer.']


# # Analyse the servival rate by cabin

# In[92]:


df['Cabin'].isnull().sum()/len(df['Cabin'])


# In[93]:


df['Cabin'].fillna('M',inplace=True)


# In[94]:


df['Cabin'].value_counts()


# In[95]:


df['deck'] = df['Cabin'].str[0]


# In[96]:


df


# In[97]:


df['deck'].value_counts()


# In[98]:


pd.crosstab(df['deck'],df['Pclass'])


# In[99]:


pd.crosstab(df['deck'],df['Survived'],normalize='index').plot(kind='bar',stacked=True)


# In[100]:


df.corr


# # Multivarient analysis

# In[101]:


sns.pairplot(df1)


# In[102]:


df1


# In[122]:


# over all servive percentage.
servive = sns.barplot(y = "Survived", data=df)


# # Conclusion:
1.About 33% survived the Titanic disaster.
2.Females were given higher priority in the rescue operation than male.so females are more likely to survive.
3.Those who paid for first class are more likely to survive. Like The first class people were given higher priority than the second class than the third class.
4.Embarked - Those who embarked at 'C' have a higher chance at survival.
5.The features such as Age, Siblings Onboard and Parents onboard didn't have major influence on the survival probability.
6.A better way for filling the missing values for Age column is explained by extracting info from Name column.
# # Thank you
