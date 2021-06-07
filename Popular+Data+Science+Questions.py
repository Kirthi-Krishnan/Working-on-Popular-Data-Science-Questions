#!/usr/bin/env python
# coding: utf-8

# # Popular Data Science Questions
# Our goal in this project is to use [Data Science Stack Exchange](https://datascience.stackexchange.com/) to determine what content should a data science education company create, based on interest by subject.
# 
# ## Stack Exchange
# **What kind of questions are welcome on this site?**
# On DSSE's help center's [section on questions](https://datascience.stackexchange.com/help/asking) , we can read that we should:
# * Avoid subjective questions.
# * Ask practical questions about Data Science — there are adequate sites for theoretical questions.
# * Ask specific questions.
# * Make questions relevant to others.
# 
# All of these characteristics, if employed, should be helpful attributes to our goal.
# 
# In the help center we also learned that in addition to the sites mentioned in the Learn section, there are other two sites that are relevant:
# * [Open Data](https://opendata.stackexchange.com/help/on-topic) (Dataset requests)
# * [Computational Science](https://scicomp.stackexchange.com/help/on-topic) (Software packages and algorithms in applied mathematics)
# 
# ** What, other than questions, does DSSE's [home](https://datascience.stackexchange.com/) subdivide into?** 
# 
# On the home page we can see that we have four sections:
# * [Questions](https://datascience.stackexchange.com/questions) — a list of all questions asked;
# * [Tags](https://datascience.stackexchange.com/tags) — a list of tags (keywords or labels that categorize questions);
# * [Users](https://datascience.stackexchange.com/users) — a list of users;
# * [Unanswered](https://datascience.stackexchange.com/unanswered) — a list of unanswered questions;
# 
# The tagging system used by Stack Exchange looks just like what we need to solve this problem as it allow us to quantify how many questions are asked about each subject.
# 
# Something else we can learn from exploring the help center, is that Stack Exchange's sites are heavily moderated by the community; this gives us some confidence in using the tagging system to derive conclusions.
# 
# ** What information is available in each post?**
# 
# Looking, just as an example, at [this](https://datascience.stackexchange.com/questions/19141/linear-model-to-generate-probability-of-each-possible-output?rq=1) question, some of the information we see is:
# 
# * For both questions and answers:
#     * The posts's score;
#     * The posts's title;
#     * The posts's author;
#     * The posts's body;
# * For questions only:
#     * How many users have it on their "
#     * The last time the question as active;
#     * How many times the question was viewed;
#     * Related questions;
#     * The question's tags;

# ## Stack Exchange Data Explorer
# Perusing the table names, a few stand out as relevant for our goal:
# * Posts
# * PostTags
# * Tags
# * TagSynonyms
# 
# Running a few exploratory queries, leads us to focus our efforts on Posts table. For examples, the Tags table looked very promising as it tells us how many times each tag was used, but there's no way to tell just from this if the interest in these tags is recent or a thing from the past.
# 
# Id |    TagName	      |  Count | ExcerptPostId | WikiPostId
# ---|------------------|--------|---------------|------------
# 2  |  machine-learning|  6919  |    4909	   |     4908
# 46 |  python	      |  3907  |	5523	   |     5522
# 81 |  neural-network  |	 2923  |	8885	   |     8884
# 194|  deep-learning	  |  2786  |	8956	   |     8955
# 77 |  classification  |	 1899  |	4911	   |     4910
# 324|  keras	          |  1736  |	9251	   |     9250
# 128|  scikit-learn	  |  1303  |    5896	   |     5895
# 321|  tensorflow	  |  1224  |	9183	   |     9182
# 47 |  nlp	          |  1162  |	147	       |     146
# 24 |  r	              |  1114  |	49	       |     48
#  

# ## Getting the Data
# To get the relevant data we run the following query.
# 
# SELECT Id, CreationDate,
#        Score, ViewCount, Tags,
#        AnswerCount, FavoriteCount
#   FROM posts
#  WHERE PostTypeId = 1 AND YEAR(CreationDate) = 2019;
#  
# Id	 |CreationDate	|Score|ViewCount|Tags	                               |AnsCount|FavCount
# -----|--------------|-----|---------|--------------------------------------|--------|---------
# 52137|17/05/19 21:54|   0 |	114	    |python,keras,prediction,evaluation	   |   0	|
# 52142|17/05/19 23:11|   1 |	160	    |deep-learning,overfitting,regularize  |   3	|   1
# 52144|18/05/19 1:08	|   1 | 3982    |machine-learning,neural-network	   |   2	|   1
# 52155|18/05/19 11:03|   1 |	46	    |machine-learning,deep-learning,cnn	   |   1	|
# 52157|18/05/19 13:1 |  16 |	8395	|machine-learning,cost-function	       |   2	|   7
# 
# ## Exploring the Data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

questions = pd.read_csv("2019_questions.csv", parse_dates=["CreationDate"])

questions.info()


# We see that only FavoriteCount has missing values. A missing value on this column probably means that the question was is not present in any users' favorite list, so we can replace the missing values with zero.
# 
# The types seem adequate for every column, however, after we fill in the missing values on FavoriteCount, there is no reason to store the values as floats.
# 
# Since the object dtype is a catch-all type, let's see what types the objects in questions["Tags"] are.

# In[2]:


questions["Tags"].apply(lambda value: type(value)).unique()


# We see that every value in this column is a string. On Stack Exchange, each question can only have a maximum of five tags (source), so one way to deal with this column is to create five columns in questions called Tag1, Tag2, Tag3, Tag4, and Tag5 and populate the columns with the tags in each row.
# 
# However, since doesn't help is relating tags from one question to another, we'll just keep them as a list.

# In[3]:


questions.fillna(value={"FavoriteCount": 0}, inplace=True)
questions["FavoriteCount"] = questions["FavoriteCount"].astype(int)
questions.dtypes


# In[4]:


questions["Tags"] = questions["Tags"].str.replace("^<|>$", "").str.split("><")
questions.sample(3)


# ## Most Used and Most Viewed
# We'll begin by counting how many times each tag was used

# In[5]:


tag_count = dict()

for tags in questions["Tags"]:
    for tag in tags:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1

tag_count = pd.DataFrame.from_dict(tag_count, orient="index")
tag_count.rename(columns={0: "Count"}, inplace=True)
tag_count.head(10)


# Let's now sort this dataframe by Count and visualize the top 20 results.

# In[6]:


most_used = tag_count.sort_values(by="Count").tail(20)
most_used


# In[7]:


most_used.plot(kind="barh", figsize=(16,8))


# Some tags are very, very broad and are unlikely to be useful; e.g.: python, dataset, r. Before we investigate the tags a little deeper, let's repeat the same process for views.
# 
# We'll use pandas's pandas.DataFrame.iterrows().

# In[8]:


tag_view_count = dict()

for index, row in questions.iterrows():
    for tag in row['Tags']:
        if tag in tag_view_count:
            tag_view_count[tag] += row['ViewCount']
        else:
            tag_view_count[tag] = row['ViewCount']
            
tag_view_count = pd.DataFrame.from_dict(tag_view_count, orient="index")
tag_view_count.rename(columns={0: "ViewCount"}, inplace=True)

most_viewed = tag_view_count.sort_values(by="ViewCount").tail(20)

most_viewed.plot(kind="barh", figsize=(16,8))


# Here,
# * most_used is a dataframe that counts how many times each of the top 20 tags was used.
# * most_viewed is a dataframe that counts how many times each of the top 20 tags was viewed.
# 
# Looking at the results from the last exercise, we see that most top tags are present in both dataframes.
# 
# Let's see what tags are in most_used, but not in most_viewed. We can identify them by the missing values in ViewCount below.

# In[9]:


in_used = pd.merge(most_used, most_viewed, how="left", left_index=True, right_index=True)
print(in_used)


# Similarly, let's see what tags are in the latter, but not the former:

# In[10]:


pd.merge(most_used, most_viewed, how="right", left_index=True, right_index=True)
print(pd.merge)


# ## Relations Between Tags
# One way of trying to gauge how pairs of tags are related to each other, is to count how many times each pair appears together. Let's do this.
# 
# We'll begin by creating a list of all tags.

# In[11]:


all_tags = list(tag_count.index)


# We'll now create a dataframe where each row will represent a tag, and each column as well. 

# In[12]:


associations = pd.DataFrame(index=all_tags, columns=all_tags)
associations.iloc[0:4,0:4]


# We will now fill this dataframe with zeroes and then, for each lists of tags in questions["Tags"], we will increment the intervening tags by one. The end result will be a dataframe that for each pair of tags, it tells us how many times they were used together.

# In[13]:


associations.fillna(0, inplace=True)

for tags in questions["Tags"]:
    associations.loc[tags, tags] += 1


# This dataframe is quite large. Let's focus our attention on the most used tags. We'll add some colors to make it easier to talk about the dataframe. 

# In[14]:


relations_most_used = associations.loc[most_used.index, most_used.index]

def style_cells(x):
    helper_df = pd.DataFrame('', index=x.index, columns=x.columns)
    helper_df.loc["time-series", "r"] = "background-color: yellow"
    helper_df.loc["r", "time-series"] = "background-color: yellow"
    for k in range(helper_df.shape[0]):
        helper_df.iloc[k,k] = "color: blue"
    
    return helper_df

relations_most_used.style.apply(style_cells, axis=None)


# The cells highlighted in yellow tell us that time-series was used together with r 22 times. The values in blue tell us how many times each of the tags was used. We saw earlier that machine-learning was used 2693 times and we confirm it in this dataframe.
# 
# It's hard for a human to understand what is going on in this dataframe. Let's create a heatmap. But before we do it, let's get rid of the values in blue, otherwise the colors will be too skewed.

# In[15]:


for i in range(relations_most_used.shape[0]):
    relations_most_used.iloc[i,i] = pd.np.NaN


# In[16]:


plt.figure(figsize=(12,8))
sns.heatmap(relations_most_used, cmap="Reds", annot=False)


# The most used tags also seem to have the strongest relationships, as given by the dark concentration in the bottom right corner. However, this could simply be because each of these tags is used a lot, and so end up being used together a lot without possibly even having any strong relation between them.
# 
# A more intuitive manifestation of this phenomenon is the following. A lot of people buy bread, a lot of people buy toilet paper, so they end up being purchased together a lot, but purchasing one of them doesn't increase the chances of purchasing the other.
# 
# Another shortcoming of this attempt is that it only looks at relations between pairs of tags and not between multiple groups of tags. For example, it could be the case that when used together, dataset and scikit-learn have a "strong" relation to pandas, but each by itself doesn't.
# 
# ## Enter Domain Knowledge
# [Keras](https://keras.io/), [scikit-learn](https://scikit-learn.org/stable/), [TensorFlow](https://www.tensorflow.org/) are all Python libraries that allow their users to employ deep learning (a type of neural network).
# 
# Most of the top tags are all intimately related with one central machine learning theme: deep learning. If we want to be very specific, we can suggest the creation of Python content that uses deep learning for classification problems (and other variations of this suggestion).
# 
# At the glance of an eye, someone with sufficient domain knowledge can tell that the most popular topic at the moment, as shown by our analysis, is deep learning.
# 
# ## Just a Fad?
# Let's read in the file into a dataframe called all_q. We'll parse the dates at read-time.

# In[17]:


all_q = pd.read_csv("all_questions.csv", parse_dates=["CreationDate"])


# We can use the same technique as before to clean the tags column.

# In[18]:


all_q["Tags"] = all_q["Tags"].str.replace("^<|>$", "").str.split("><")


# Before deciding which questions should be classified as being deep learning questions, we should decide what tags are deep learning tags.
# 
# The definition of what constitutes a deep learning tag we'll use is: a tag that belongs to the list ["lstm", "cnn", "scikit-learn", "tensorflow", "keras", "neural-network", "deep-learning"].
# 
# This list was obtained by looking at all the tags in most_used and seeing which ones had any relation to deep learning. You can use Google and read the tags descriptions to reach similar results.
# 
# We'll now create a function that assigns 1 to deep learning questions and 0 otherwise; and we use it.

# In[19]:


def class_deep_learning(tags):
    for tag in tags:
        if tag in ["lstm", "cnn", "scikit-learn", "tensorflow",
                   "keras", "neural-network", "deep-learning"]:
            return 1
    return 0


# In[21]:


all_q["DeepLearning"] = all_q["Tags"].apply(class_deep_learning)
all_q.sample(5)


# The data-science-techonology landscape isn't something as dynamic to merit daily, weekly, or even monthly tracking. Let's track it quarterly.
# 
# Since we don't have all the data for the first quarter of 2020, we'll get rid of those dates:

# In[22]:


all_q = all_q[all_q["CreationDate"].dt.year < 2020]


# Let's create a column that identifies the quarter in which a question was asked.

# In[23]:


def fetch_quarter(datetime):
    year = str(datetime.year)[-2:]
    quarter = str(((datetime.month-1) // 3) + 1)
    return "{y}Q{q}".format(y=year, q=quarter)


# In[24]:


all_q["Quarter"] = all_q["CreationDate"].apply(fetch_quarter)
all_q.sample(5)


# For the final stretch, we'll group by quarter and:
# * Count the number of deep learning questions.
# * Count the total number of questions.
# * Compute the ratio between the two numbers above.

# In[25]:


quarterly = all_q.groupby('Quarter').agg({"DeepLearning": ['sum', 'size']})
quarterly.columns = ['DeepLearningQuestions', 'TotalQuestions']
quarterly["DeepLearningRate"] = quarterly["DeepLearningQuestions"]                                /quarterly["TotalQuestions"]


# In[26]:


quarterly.reset_index(inplace=True)
quarterly.sample(5)


# In[27]:


ax1 = quarterly.plot(x="Quarter", y="DeepLearningRate",
                    kind="line", linestyle="-", marker="o", color="orange",
                    figsize=(24,12)
                    )

ax2 = quarterly.plot(x="Quarter", y="TotalQuestions",
                     kind="bar", ax=ax1, secondary_y=True, alpha=0.7, rot=45)

for idx, t in quarterly["TotalQuestions"].iteritems():
    ax2.text(idx, t, str(t), ha="center", va="bottom")
xlims = ax1.get_xlim()

ax1.get_legend().remove()

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles=handles1 + handles2,
           labels=labels1 + labels2,
           loc="upper left", prop={"size": 12})


for ax in (ax1, ax2):
    for where in ("top", "right"):
        ax.spines[where].set_visible(False)
        ax.tick_params(right=False, labelright=False)


# It seems that deep learning questions was a high-growth trend since the start of DSSE and it looks like it is plateauing. There is no evidence to suggest that interest in deep learning is decreasing and so we maintain our previous idea of proposing that we create deep learning content.
