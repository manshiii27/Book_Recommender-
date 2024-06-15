#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books= pd.read_csv('C:\\Users\\Jay Pal\\Downloads\\Books.csv')
users = pd.read_csv('C:\\Users\\Jay Pal\\Downloads\\Users.csv')
ratings = pd.read_csv('C:\\Users\\Jay Pal\\Downloads\\Ratings.csv')


# In[3]:


books.head()


# In[4]:


print(books.shape)


# In[5]:


books.isnull().sum()


# In[6]:


books.duplicated().sum()


# In[9]:


ratings.head()


# In[10]:


print(ratings.shape)


# In[11]:


ratings.isnull().sum()


# In[12]:


ratings.duplicated().sum()


# In[13]:


users.head()


# In[14]:


print(users.shape)


# In[15]:


users.duplicated().sum()


# In[16]:


users.isnull().sum()


# # Popularity Based Recommender System

# In[17]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[18]:


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace= True)
num_rating_df


# In[19]:


avg_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_ratings'},inplace= True)
avg_rating_df


# In[20]:


popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[21]:


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)
popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]


# In[23]:


popular_df


# # Collaborative Filtering Based Recommender System

# In[24]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] >200
padhe_likhe_users = x[x].index


# In[25]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[26]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']
famous_books = y[y].index


# In[27]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[28]:


final_ratings.drop_duplicates()


# In[29]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[30]:


pt.fillna(0,inplace= True)


# In[31]:


pt


# In[32]:


from sklearn.metrics.pairwise import cosine_similarity


# In[33]:


similarity_scores = cosine_similarity(pt)


# In[34]:


similarity_scores.shape


# In[51]:


import numpy as np
import pandas as pd

# Example data setup
# Assuming pt is a pivot table and books is a DataFrame
data = {'Book-Title': ['The Mummies of Urumchi', 'Book1', 'Book2', 'Book3', 'Book4', 'Book5', 'Book6', 'Book7', 'Book8'],
        'Book-Author': ['Author A', 'Author B', 'Author C', 'Author D', 'Author E', 'Author F', 'Author G', 'Author H', 'Author I']}

books = pd.DataFrame(data)
pt = pd.DataFrame(np.random.rand(9, 5), index=data['Book-Title'])

def recommend(book_name):
    if book_name not in pt.index:
        raise ValueError(f"The book '{book_name}' is not in the index.")
    
    # Get the index of the book
    index = np.where(pt.index == book_name)[0][0]
    
    # Calculate similarity scores
    similarity_scores = np.dot(pt.values, pt.values.T)
    
    # Get the similarity scores for the given book
    book_similarity_scores = list(enumerate(similarity_scores[index]))
    
    # Sort the books by similarity score in descending order and exclude the first one (itself)
    similar_items = sorted(book_similarity_scores, key=lambda x: x[1], reverse=True)[1:9]
    
    # Collect the titles of the similar books
    suggestions = [pt.index[i[0]] for i in similar_items]
    
    return suggestions

# Define the book name you want to get recommendations for
book_name = 'The Mummies of Urumchi'

# Example usage
try:
    recommendations = recommend(book_name)
    for book in recommendations:
        print(book)
        item = []
        temp_df = books[books['Book-Title'] == book]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0]))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0]))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]))
        
        data.append(item)
except ValueError as e:
    print(e)


# In[48]:


recommend(book_name)


# In[59]:


pt.index[1]


# In[60]:


import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))


# In[54]:


books.drop_duplicates('Book-Title')


# In[61]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('book.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


# # Thankyou!
