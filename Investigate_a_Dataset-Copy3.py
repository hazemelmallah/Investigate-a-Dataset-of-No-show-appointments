#!/usr/bin/env python
# coding: utf-8

# # Project: No show appointment for patients Data Analysis
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# This dataset collects information
# from 100k medical appointments in
# Brazil and is focused on the question
# of whether or not patients show up
# for their appointment. A number of
# characteristics about the patient are
# included in each row.
# ● ‘ScheduledDay’ tells us on
# what day the patient set up their
# appointment.
# ● ‘Neighborhood’ indicates the
# location of the hospital.
# ● ‘Scholarship’ indicates
# whether or not the patient is
# enrolled in Brasilian welfare
# program Bolsa Família.
# ● Be careful about the encoding
# of the last column: it says ‘No’ if
# the patient showed up to their
# appointment, and ‘Yes’ if they
# did not show up.
#  
# 
# 
# ### Question(s) for Analysis
# What factors are
# important for us to
# know in order to
# predict if a patient will
# show up for their
# scheduled
# appointment?
# 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties
# 

# In[3]:


#load data
df=pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# In[4]:


#exploring shape of data
df.shape


# In[5]:


#explore the mean and counts and max and min 
df.describe()


# max age is 115 & min age is -1 & mean is 37

# In[6]:


#check if there are any missing value
df.info()


# No missing values

# In[7]:


#check for the rows which have age with -1
drp=df[df['Age']<0]
drp


# 
# ### Data Cleaning
#  

# In[8]:


# removing the age of -1
df.drop(index=99832,inplace=True)


# In[9]:


df.duplicated(['PatientId','No-show']).sum()


# there is patients with same status of show with same patient id

# In[10]:


df.drop_duplicates(['PatientId','No-show'], inplace=True)


# removing duplicated patient id  with status of no show & show

# In[11]:


df.dropna()
df.shape


# In[12]:


df.describe()


# In[13]:


#cleaning unnecessary data by remove columns
df.drop(['PatientId', 'AppointmentID','ScheduledDay','AppointmentDay'], axis = 1, inplace = True)


# In[ ]:





# In[14]:


#edit column name
df.rename(columns = {'Hipertension' : 'hypertension'}, inplace = True)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# Now we are going to explore and visualize data after we wrangling and cleaning it 
# 
# ## General Exploratory

# In[15]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df.hist(figsize=(9,8));


# In[16]:


df.info()


# In[17]:


#edit column name and counts no of show& no show
df.rename(columns = {'No-show' : 'No_show'}, inplace = True)
showing=df.No_show=='No'
no_showing=df.No_show=='Yes'
df.count(),df[showing].count(),df[no_showing].count()


# number of showing patients is 4 times than no showing patients

# In[18]:


# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.
#relation between average age and no show
def attending(df,column_name,att,absent):
    
  #df[column_name][showing].mean()
  #df[column_name][no_showing].mean()
  df[column_name][showing].hist(alpha=0.5, bins=10, label='show')
  df[column_name][no_showing].hist(alpha=0.5, bins=10, label='no_show')
  plt.title('relation between showing & average age')
  plt.ylabel('showing & no showing');
  plt.xlabel('Average Age')
  plt.legend();
    
attending(df,'Age',showing,no_showing)


# no of shpow patients are more than no_show patients in all ages

# In[19]:


#relation between chronic diseases and average age according with no show
df[showing].groupby(['Diabetes','hypertension']).Age.mean().plot(kind= 'bar', color='green', label= 'showing');
df[no_showing].groupby(['Diabetes','hypertension']).Age.mean().plot(kind= 'bar', color='red', label= 'no_showing');
plt.title('relation between chronic disease & average age')
plt.xlabel('Diabetes & hypertension')
plt.ylabel('Average Age')
#df.Age[Diabetes][hypertension].hist(alpha=0.5, bins=15, label='showing')
#df.Age[Diabetes][hypertension].hist(alpha=0.5, bins=15, label='no_showing')
plt.legend();


# chronic diseases existing don't affect attedance of patients

# In[20]:


df[showing].mean(),df[no_showing].mean()


# average age of showing patients is 37 & average age of no_showing patients is 34

# In[21]:


#how the gender affect no show?
plt.figure(figsize=[12,8])
df['Gender'][showing].value_counts().plot(kind='pie', label='show')
plt.title('perentage between male and female of attending')
plt.ylabel('number of patients')
plt.xlabel('Gender')
plt.legend();


# percentage of gender attend

# In[22]:


def attending(df,column_name,att,absent):
    
  #df[column_name][showing].mean()
  #df[column_name][no_showing].mean()
  df[column_name][showing].hist(alpha=0.5, bins=10, label='show')
  df[column_name][no_showing].hist(alpha=0.5, bins=10, label='no_show')
  plt.title('relation between showing & average age')
  plt.ylabel('showing & no showing');
  plt.xlabel('Average Age')
  plt.legend();
    
attending(df,'SMS_received',showing,no_showing)


# people who didn't receive SmS are more attend than who received SmS

# <a id='conclusions'></a>
# ## Conclusions
# 
# -we can see that the female show more than men
# -we can see that the show people is more than no show
# -we can see that the less the age the more show is
# -we can see that according to chronic disease the show is more than no show
# -age affect no of show as 0-10 the most number of show patients rather than the above 65 is the least no of patients show
# ## Limitations
# there is not a correlation relation between show & age & chronic diseases & sex
# 
# ## Submitting your Project 
# 

# In[23]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

