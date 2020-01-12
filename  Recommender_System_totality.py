import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 

# reads dataset 
ds = pd.read_csv("sample-data.csv")

ds.head()

# 1. calculates tf-idf scores for items
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])


# 2. Calculate cosine similarity 
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}

# 3. Saving results in order of similarity to item i

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]
    
print('Saved')

def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

# 4.a Just reads the results out of the dictionary.
def recommend(item_id, num, threshold):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        
        # A condition can be added to only to print items above a similairty threshold (50% etc.)
        if(rec[0]>threshold):
            print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

# 4. Recommend items similar to item_id and num = number of recommendations with a similarity threshold value 

# single recommendations 
recommend(item_id=9, num=5, threshold= 0.3)  # Recommend  5 items similar to item_id 9 with similarity of >30%
print('')
print('')
print('')
recommend(item_id=22, num=10, threshold= 0.5) # Recommend  10 items similar to item_id 22 with similarity of >50%
print('')
print('')
print('')
recommend(item_id=34, num=10, threshold= 0.2)
print('')
print('')
print('')
recommend(item_id=492, num=10, threshold= 0.2)

# Looping through all items (time consuming)
'''
for itemid in range(1,500):
    print('')
    print('')
    recommend(item_id=itemid, num=10, threshold= 0.2)
    print('')
    print('')
'''
