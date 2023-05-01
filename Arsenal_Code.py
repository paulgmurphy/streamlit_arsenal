#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd


# In[61]:


# Import dataset
df = pd.read_csv("df_full_premierleague.csv")
df.head()


# In[3]:


# Details of dataset
df.shape
df.duplicated().sum()


# In[4]:


# Removing columns
df.drop(['Unnamed: 0', 'link_match'], axis=1, inplace=True)
df.head()


# In[5]:


# Creating Arsenal dataframe
df_arsenal = df.loc[(df['home_team'] == 'Arsenal') | (df['away_team'] == 'Arsenal')]
df_arsenal


# In[6]:


# Splitting fulltime score column and removing '-'
df_arsenal[['HomeGoals', 'AwayGoals']] = df_arsenal['result_full'].str.split('-', expand = True)


# In[8]:


# Drop result_full column
df_arsenal.drop('result_full', axis=1, inplace=True)


# In[9]:


# Drop result_ht column
df_arsenal.drop('result_ht', axis=1, inplace=True)


# In[10]:


# Changing HomeGoals and AwayGoals to integers
df_arsenal = df_arsenal.astype({'HomeGoals': int, 'AwayGoals': int})


# In[11]:


# Checking type of columns
df_arsenal.dtypes


# In[62]:


# Just Arsenal away games
df_arsenal_away = df_arsenal.loc[(df['away_team'] == 'Arsenal')]


# In[13]:


# Just Arsenal home games
df_arsenal_home = df_arsenal.loc[(df['home_team'] == 'Arsenal')]


# In[14]:


# New column to see if Arsenal won, drew or lost at home games
df_arsenal_home['GoalDifference'] = df_arsenal_home['HomeGoals'] - df_arsenal_home['AwayGoals']


# In[15]:


# Column for home wins
df_arsenal_home['Win'] = df_arsenal_home['GoalDifference'] > 0
df_arsenal_home['Win'] = df_arsenal_home['Win'].astype(int)


# In[16]:


# Column for home draws
df_arsenal_home['Draw'] = df_arsenal_home['GoalDifference'] == 0
df_arsenal_home['Draw'] = df_arsenal_home['Draw'].astype(int)


# In[17]:


# Column for home losses
df_arsenal_home['Loss'] = df_arsenal_home['GoalDifference'] < 0
df_arsenal_home['Loss'] = df_arsenal_home['Loss'].astype(int)


# In[18]:


# New column to see if Arsenal won, drew or lost at away games
df_arsenal_away['GoalDifference'] = df_arsenal_away['HomeGoals'] - df_arsenal_away['AwayGoals']


# In[19]:


# Column for away wins
df_arsenal_away['Win'] = df_arsenal_away['GoalDifference'] < 0
df_arsenal_away['Win'] = df_arsenal_away['Win'].astype(int)


# In[20]:


# Column for away draws
df_arsenal_away['Draw'] = df_arsenal_away['GoalDifference'] == 0
df_arsenal_away['Draw'] = df_arsenal_away['Draw'].astype(int)


# In[21]:


# Column for away losses
df_arsenal_away['Loss'] = df_arsenal_away['GoalDifference'] > 0
df_arsenal_away['Loss'] = df_arsenal_away['Loss'].astype(int)


# In[22]:


# Joining the two dataframes into one again
dataframes = [df_arsenal_home, df_arsenal_away]

df_arsenal = pd.concat(dataframes)
df_arsenal


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


plt.figure(figsize=(7,7))
sns.countplot(x="season", hue ="Win", data=df_arsenal)
plt.xlabel('Season')
plt.ylabel('Total');
plt.title('Wins and Non-wins')
plt.legend(labels = ['Non-win', 'Win'])
# In the last 2 seasons Aresenal have not won many games. Why?


# In[63]:


# Finding the number of Wins, Loss, Draws per season by groupby
Totals = df_arsenal.groupby(['season'])[['Win', 'Draw', 'Loss']].sum()
Totals


# In[29]:


# I've researched and the dataset is incomplete. For the sake of the following graph I will include the final results of 
# the season in the columns of wins, draws and losses but not for the stats during the games.

Totals.drop(['20/21'],inplace = True)

Totals


# In[30]:


Totals.plot.bar(stacked=True)


# In[32]:


# Finding out if there is a major difference in their home results 
home_season = df_arsenal_home.groupby(['season'])[['Win', 'Draw', 'Loss']].sum()
home_season.drop(['20/21'],inplace = True)
home_season


# In[33]:


# Graph of home win record by season
home_graph = home_season['Win'] / (home_season['Draw'] + home_season['Loss'] + home_season['Win'])
plt.figure(figsize=(7,7))
plt.plot(home_graph)
plt.xlabel('Season')
plt.ylabel('Ratio of Wins')
plt.title('Home win record by season');


# In[34]:


# Finding out if there is a major difference in their away results 
away_season = df_arsenal_away.groupby(['season'])[['Win', 'Draw', 'Loss']].sum()
away_season.drop(['20/21'],inplace = True)
away_season


# In[35]:


# Graph of away win record by season
away_graph = away_season['Win'] / (away_season['Draw'] + away_season['Loss'] + away_season['Win'])
plt.figure(figsize=(7,7))
plt.plot(away_graph)
plt.xlabel('Season')
plt.ylabel('Ratio of Wins')
plt.title('Away win record by season');


# In[36]:


# Create target and data for Home games
df_arsenal_home
datahome = df_arsenal_home.drop(['season', 'date', 'home_team', 'away_team', 'HomeGoals', 'AwayGoals', 'GoalDifference', 'Win',
                                 'Draw', 'Loss'], axis=1)
targethome = df_arsenal_home['Win']
datahome


# In[37]:


# There are a lot of columns I don't want. Like goals scored at half-time average for the season
# I just want the direct data from every game
datahome = datahome.iloc[:,0:24]
datahome


# In[38]:


# Training
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(datahome, targethome, test_size=0.25)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)


# In[39]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)


# In[40]:


plt.figure(figsize=(5,5))
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()


# In[41]:


# Confusion Matrix
from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1.0)
clf.fit(X_train, y_train)
y_pred_lr = clf.predict(X_test)
cm_lr = pd.crosstab(y_test, y_pred_lr, rownames=['Classe réelle'], colnames=['Classe prédite'])
cm_lr


# In[42]:


# Logistic Regression score
clf.score(X_test, y_test)


# In[43]:


# Logistic Regression report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_lr))


# In[44]:


# It looks like clearances are the most important feature.


# In[46]:


# Checking to see if the averages of clearances have changed over the years in home games
clearances_home = df_arsenal_home.groupby(['season'])[['home_clearances', 'away_clearances']].mean()
clearances_home.drop(['20/21'],inplace = True)
clearances_home


# In[47]:


# Plotting home game clearances
plt.figure(figsize=(10,10))
clearances_home.plot()
plt.title('Arsenal home games')


# In[48]:


# Checking to see if the averages of clearances has changed over the years in away games
clearances_away = df_arsenal_away.groupby(['season'])[['home_clearances', 'away_clearances']].mean()
clearances_away.drop(['20/21'],inplace = True)
clearances_away


# In[49]:


# Plotting away game clearances
plt.figure(figsize=(10,10))
clearances_away.plot()
plt.title('Arsenal away games')


# In[50]:


# Changing column names for clarity
clearances_home.rename(columns = {'home_clearances':'Arsenal1','away_clearances':'Opposition1'}, inplace = True)
clearances_home
clearances_away.rename(columns = {'home_clearances':'Opposition2','away_clearances':'Arsenal2'}, inplace = True)
clearances_away


# In[51]:


# Putting the two datasets together
clearances = pd.merge(clearances_home, clearances_away, how='outer', on='season')
clearances


# In[52]:


# Cleaning the dataframe to make a graph
clearances['Arsenal'] = (clearances['Arsenal1'] + clearances['Arsenal2']) / 2
clearances['Opposition'] = (clearances['Opposition1'] + clearances['Opposition2']) / 2
clearances.drop(['Arsenal1','Arsenal2','Opposition1','Opposition2'],axis=1,inplace=True)
clearances


# In[53]:


# Plotting clearances for all Arsenal games
plt.figure(figsize=(10,10))
clearances.plot()
plt.title('Arsenal and Opposition clearances per game (mean)')


# In[54]:


# Just for Arsenal
clearances.drop(['Opposition'],axis=1,inplace=True)
clearances


# In[55]:


# Plotting clearances for just Arsenal
plt.figure(figsize=(10,10))
clearances.plot()
plt.title('Arsenal clearances per game (mean)')


# In[64]:


#Method 2 — Obtain importances from a tree-based model

# Training

X_train2, X_test2, y_train2, y_test2 = train_test_split(datahome, targethome, test_size=0.25)

ss = StandardScaler()
X_train_scaled2 = ss.fit_transform(X_train2)
X_test_scaled2 = ss.transform(X_test2)

from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train_scaled2, y_train2)
importances = pd.DataFrame(data={
    'Attribute': X_train2.columns,
    'Importance': model.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)


# In[65]:


plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()


# In[66]:


# Testing score of XGB
model = XGBClassifier()
model.fit(X_train2, y_train2)


# In[67]:


y_pred_xgb = model.predict(X_test2)
cm_xgb = pd.crosstab(y_test2, y_pred_xgb, rownames=['Classe réelle'], colnames=['Classe prédite'])
cm_xgb


# In[68]:


# XGB score
clf.score(X_test2, y_test2)


# In[69]:


# XGB report
print(classification_report(y_test2, y_pred_xgb))


# In[70]:


# It looks like opposition corners are an important feature


# In[71]:


# Checking to see if the averages of corners have changed over the years in home games
corners_home = df_arsenal_home.groupby(['season'])[['away_corners', 'home_corners']].mean()
corners_home.drop(['20/21'],inplace = True)
corners_home


# In[72]:


# Plotting oposition corners in home games
plt.figure(figsize=(10,10))
corners_home.plot()
plt.title('Arsenal home games, opposition corners (mean)')


# In[73]:


# Checking to see if the averages of opposition corners have changed over the years in away games
corners_away = df_arsenal_away.groupby(['season'])[['home_corners', 'away_corners']].mean()
corners_away.drop(['20/21'],inplace = True)
corners_away


# In[74]:


# Plotting oposition corners in away games
plt.figure(figsize=(10,10))
corners_away.plot()
plt.title('Arsenal away games, opposition corners (mean)')


# In[75]:


# Combining the two dataframes together
corners = pd.merge(corners_away, corners_home, how='outer', on='season')
corners


# In[76]:


# Making it easier to understand. Columns changed to Arsenal and Opposition averages for each season
corners['Opposition Corners'] = (corners['home_corners_x'] + corners['away_corners_y']) / 2
corners.drop(['home_corners_x','away_corners_y'],axis=1,inplace=True)
corners['Arsenal Corners'] = (corners['home_corners_y'] + corners['away_corners_x']) / 2
corners.drop(['home_corners_y','away_corners_x'],axis=1,inplace=True)
corners


# In[77]:


# Plotting opposition corners
plt.figure(figsize=(10,10))
corners.plot()
plt.title('Corners (mean)')


# In[78]:


# Just Opposition
corners.drop(['Arsenal Corners'],axis=1,inplace=True)
corners


# In[79]:


# Plotting opposition corners
plt.figure(figsize=(10,10))
corners.plot()
plt.title('Opposition corners per game (mean)')


# In[100]:


# Looking at opposition shots on goal
shots = df_arsenal_home.groupby(['season'])[['away_shots', 'away_shots_on_target']].mean()
shots.drop(['20/21'],inplace = True)
shots
shots2 = df_arsenal_away.groupby(['season'])[['home_shots', 'home_shots_on_target']].mean()
shots2.drop(['20/21'],inplace = True)
shots2
combined_shots = pd.merge(shots, shots2, how='outer', on='season')
combined_shots


# In[101]:


# Just the opposition shots
combined_shots['Opposition Shots'] = (combined_shots['away_shots'] + combined_shots['home_shots']) / 2
opposition_shots = combined_shots.drop(['away_shots','home_shots','away_shots_on_target','home_shots_on_target'],axis=1,inplace=True)
opposition_shots


# In[102]:


# Plotting opposition shots
plt.figure(figsize=(10,10))
combined_shots.plot()
plt.title('Opposition shots per game (mean)')


# In[98]:


combined_shots
# Just their shots on target
combined_shots['Opposition Shots on Target'] = (combined_shots['away_shots_on_target'] + combined_shots['home_shots_on_target']) / 2
combined_shots.drop(['away_shots','home_shots','away_shots_on_target','home_shots_on_target'],axis=1,inplace=True)
combined_shots


# In[103]:


plt.figure(figsize=(10,10))
combined_shots.plot.bar()
plt.title('Opposition shots on target per game (mean)')


# In[ ]:




