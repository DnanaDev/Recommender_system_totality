# Requirements of Totality  Corp:
<b>Assignment 1 </b><br>(Design Recommendation System Architecture)<br>
Content discovery is a vital part for the Yovo. Users often don't know what they want to watch and need a way to discover content without searching for it. <b>Create a feed personalisation algorithm strategy which enables users to discover the right content. Underlying algorithm must strike an elegant balance between Machine Learning and giving the user control over what content they want to see.</b>
Note:
- Video has certain text attributes: tags, category, title text (context) - Use pseudocode wherever necessary
- Download ​yovo app​ to see actual feed



# Potential solutions to the problem:

### Recommender Systems generally follow one of two methods:
##### 1. Collaborative filtering 
    
This approach <b> will not apply to the Yovo app as there is no option for users to like certain videos in their feed.  
The approach would require the app to keep track of each user and their likes, shares etc. in the form of user matrix. </b>


##### 2. Content Based filtering (Suitable for YOVO app)

This approach utilizes a series of discrete characteristics of an item in order to recommend additional items with similar properties. 

<b>Based on items,which are videos in the case of Yovo including the characteristics - tags, category, title text (context) in this case. 

This method is suitable in the case where metadata is available and no matrix of users ids, preferences is available.</b>


# Practical Example of Content based filtering (Architecture) :

## The Dataset 

The dataset of 500 entries of different items like shoes, shirts etc., along with an item-id and a <b>textual description of the item.</b><br>
The system creates a profile for each item and recommends similar items.

<b>For totality, I imagine this dataset will be replaced with one containing textual discription of the video(videoId)
using the tags or title  attributes.</b>

## Process

#### 1. Extract TF * IDF [(term frequency)*(Inverse document frequency)] Score

The TF*IDF algorithm finds the importance of a word in the tag. This is done for each word in the tag and for each item.<br>
<b>This is implemented using scikit-learns inbuilt TF-IDF vectorizer. </b>

#### 2. Calculating Similarity using Cosine Similarity  
Once we have the vectors for each item, we can use cosine similarity to find items/ words that are similar.<br>
Cosine similarity judges how close the cosine angles are in the vector representation of the items.<br>

<b>This is done using the linear_kernel method of scikit-learn. It takes the tfidf matrix of the items as input and compares them to find items that are similar. </b>
    
#### 3. Store results of cosine similarity 
The results of cosine similarity are stored in result, arranged according to similarity with item i.

#### 4. Recommending Items
The function recommend takes in the item for which a recommendation is to be made and the number of recommmendations to be made and reads out the most similar items from results.
<br>
We input a threshold value to only get recommendation above a certain similarity index.<br>
<b> The items recommended can then be fed into the personalized feed of the user for relevant video recommendations.</b>


```python
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

# Looping through all recommendations(slow)
#for itemid in range(1,501):
#    print('')
#    print('')
#    recommend(item_id=itemid, num=10, threshold= 0.2)
#    print('')
#    print('')

```

    Saved
    Recommending 5 products similar to Baby micro d-luxe cardigan...
    -------
    Recommended: Micro d-luxe cardigan (score:0.37550840843454325)
    
    
    
    Recommending 10 products similar to Cap 2 t-shirt...
    -------
    Recommended: Cap 2 crew (score:0.7049890803634637)
    Recommended: Cap 2 t-shirt (score:0.7041906217524195)
    Recommended: Cap 2 cap sleeve (score:0.6635362790007241)
    Recommended: Cap 2 zip neck (score:0.6225162259563587)
    Recommended: Cap 2 zip neck (score:0.5295225236280288)
    Recommended: Cap 2 v-neck (score:0.5094581638236453)
    
    
    
    Recommending 10 products similar to Delivery shorts...
    -------
    Recommended: Custodian pants (score:0.32366517296967345)
    Recommended: Custodian pants (score:0.32182027336965424)
    Recommended: Custodian pants (score:0.32144324237922)
    Recommended: Thrift shorts (score:0.28426808748740656)
    Recommended: All-wear cargo shorts (score:0.283497236891926)
    Recommended: All-wear shorts (score:0.25662867338464423)
    Recommended: Fezzman shirt (score:0.23970288327831699)
    Recommended: S/s island hopper shirt (score:0.21360431696984364)
    Recommended: L/s island hopper shirt (score:0.2130792598281027)
    
    
    
    Recommending 10 products similar to Freedom to roam t-shirt...
    -------
    Recommended: Freedom to roam t-shirt (score:0.9812622733833937)
    Recommended: Rockpile t-shirt (score:0.25460190581261466)
    Recommended: Iceberg t-shirt (score:0.25196878478551704)
    Recommended: Peregrine t-shirt (score:0.24033174994759707)
    Recommended: Text logo t-shirt (score:0.23862073502004064)
    Recommended: Wind path t-shirt (score:0.22744185912549925)
    Recommended: Live simply deer t-shirt (score:0.22689318214302112)
    Recommended: Trout head t-shirt (score:0.22502819128156626)
    Recommended: Flying fish 2 t-shirt (score:0.2238059283837155)
    Recommended: Tarpon t-shirt (score:0.22049973980630666)


## Insights 

<br><b>
1. The system recommends similar items after computing cosine similarity of items. <br>
2. The recommendations made with lower similarity are of greater interest than the ones with very high similarity as they seem to be duplicate/variants of the same product, hence the extremely high similarity. </b><br>

<b>Assignment 2 </b> <br>(Advanced Learning Problem Set)<br>
In case social media setup, users do not tag data properly as compared to e-commerce. You need to design a feed personalisation strategy for poorly labeled video dataset.

#### Baseline predictions methods

Can be used when the data is sparse, the average tag/most common tags by similar users might be used to fill in data.
