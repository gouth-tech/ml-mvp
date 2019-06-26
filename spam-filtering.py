#!/usr/bin/env python
# coding: utf-8

# #  Spam classification with Naive Bayes and Support Vector Machines.

# - Libraries
# - Exploring the Dataset
# - Distribution spam and non-spam plots
# - Text Analytics
# - Feature Engineering
# - Predictive analysis (**Multinomial Naive Bayes and Support Vector Machines**)
# - Conclusion
# 

# ## Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm

import warnings
warnings.filterwarnings("ignore")




# ## Exploring the Dataset

# In[68]:


data = pd.read_csv('spam.csv', encoding='latin-1')
data.head


# ## Distribution spam/non-spam plots

# In[3]:


count_Class=pd.value_counts(data["v1"], sort= True)
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
#plt.show()


# In[4]:


count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
#plt.show()


# ## Text Analytics

# In[7]:


count1 = Counter(" ".join(data[data['v1']=='ham']["v2"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter(" ".join(data[data['v1']=='spam']["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})


# In[8]:


df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
#plt.show()


# In[9]:


df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
#plt.show()


# In[28]:


f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
np.shape(X)


# In[11]:


data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
#print([np.shape(X_train), np.shape(X_test)])


# In[12]:


list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1 


# In[13]:


matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)


# In[14]:


best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]


# In[15]:


models[models['Test Precision']==1].head(n=5)


# In[16]:


best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
models.iloc[best_index, :]


# In[17]:


m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# In[18]:


list_C = np.arange(500, 2000, 100) #100000
score_train = np.zeros(len(list_C))
score_test = np.zeros(len(list_C))
recall_test = np.zeros(len(list_C))
precision_test= np.zeros(len(list_C))
count = 0
for C in list_C:
    svc = svm.SVC(C=C)
    svc.fit(X_train, y_train)
    score_train[count] = svc.score(X_train, y_train)
    score_test[count]= svc.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
    count = count + 1 


# In[19]:


matrix = np.matrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)


# In[20]:


best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]


# In[21]:


models[models['Test Precision']==1].head(n=5)


# In[22]:


best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
svc = svm.SVC(C=list_C[best_index])
svc.fit(X_train, y_train)
models.iloc[best_index, :]


# In[24]:


m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# In[78]:
text = """

All India Certification Programs in AIML and Blockchain by IIIT Hyderabad and TalentSprint
Spam
x
IIIT Hyderabad and TalentSprint <newsletters@analyticsindiamag.com>
Wed, Jun 5, 12:27 PM
to me

Why is this message in spam? It is similar to messages that were identified as spam in the past.
Report not spam
 
Greetings from IIIT Hyderabad and TalentSprint.

Artificial Intelligence, Machine Learning, Blockchain and other deep technologies are increasingly impacting the life of a tech professional.

IIIT Hyderabad, a leading Computer Science institution in India is offering executive programs for working professionals in these deep tech areas.


Analytics India IIIT-H 
IIIT Hyderabad in association with TalentSprint launched Executive Programs on AI and Machine Learning and one on Blockchain and Digital Ledger Technologies, in 2018. Over 1600 professionals have so far built expertise in these technologies across Eleven Cohorts since the launch.

The programs’ pracademic design (deep academic knowledge combined with industry best practices) and hybrid format (combining in-person classes and online classes cum support) have been warmly embraced by working professionals.

These two programs are best suited for tech professionals across India. This high-touch, high-tech program will be delivered over 18 weeks in a hybrid format. There will be 3 campus visits of 3 days each to IIIT Hyderabad, and live interactive online classes in other weeks.


 news 
Gear up to enter a new orbit of forthcoming revolution. With new-age skills, build a successful career at IIIT Hyderabad.

Sincerely,
IIIT Hyderabad | TalentSprint

 
 
This email was sent from IIIT Hyderabad and TalentSprint to gouthammn96@gmail.com.

Analytics India Magazine | #280 , 2nd floor, 5th Main, 15 A cross, Sector 6 , HSR layout, Bangalore, Karnataka, 560037, India

Click here to unsubscribe | Update your profile


"""

Y = [text]
f = feature_extraction.text.CountVectorizer(stop_words = 'english')
f.fit(data["v2"]) # fitting

X = f.transform(Y) # mapping
res=svc.predict(X)
if res == [0]:
    print("Ham")
else:
    print("Spam")

