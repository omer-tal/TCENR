import json
import numpy as np
import pickle
import random
import pathlib
from data_preprocess_utils import load_user_graph,filter_user_graph,user_to_uid,load_poi_file,get_glove_dict,load_word_dictionary,clean_review,translate_words_to_index,load_data,poi_to_pid,get_user_poi_reviews,create_train_test_dict,add_record_to_dict,get_user_reviewed,join_reviews_with_split,get_counts,pad_reviews,distance,build_location_graph,random_walk,graph_stats,write_to_pickle,data_to_uid,index_words,get_words_used

# Set parameters value used in data processing
MAX_LIMIT=99999999999
# Set limits to maximum - load all users, locations and reviews
N=MAX_LIMIT
USERS_LIMIT = MAX_LIMIT
POIS_LIMIT = MAX_LIMIT
# Define the test split from the rest of the data
SPLIT = 0.2
# Define the validation split from the reminder after test split
VALIDATION_SPLIT = 0.3
# Number of negative samples per positive one
NEGATIVE = 4
# Limit the data to include only users and locations with reviews in the ranges
MIN_USER_REVIEWS = 10
MAX_USER_REVIEWS = MAX_LIMIT
MIN_POI_REVIEWS = 10
# Limit the data to include only uses with number of friends in the range
MIN_FRIENDS = 10
MAX_FRIENDS = MAX_LIMIT
# The maximum radius for two locations to be considered directly connected
POI_MAX_RADIUS = 0.5 
# Sample ratio for POIs when constructing initial POI graph
POI_SAMPLE_RATIO = 0.1
# Defining input files locations
FILE_NAME = "dataset/review.json"
USER_FILE = "dataset/user.json"
POI_FILE = 'dataset/business.json'
# Glove file location used for the words embedding layer
GLOVE_FILE = "dataset/glove.6B."
# Glove file size
WORD_EMBEDDING_FILE = 50
# Characters to filter out of reviews
FILTERED_CHARS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\''
dictionary = {}

# Load the user graph
user_graph,user_review_count = load_user_graph(USER_FILE,USERS_LIMIT)
print("Loaded graph")
# Filter out users with friends not in given range
filtered_graph = filter_user_graph(user_graph,MIN_FRIENDS,MAX_FRIENDS,user_review_count,1,MAX_USER_REVIEWS)

user_graph = filtered_graph

# Load business locations data
locations,categories,address = load_poi_file(POI_FILE,POIS_LIMIT)
print("Loaded locations")
# Load reviews for users and POIs in user_graph and locations
data,poi_reviews_count,user_review_count = load_data(FILE_NAME,FILTERED_CHARS,N,user_graph,locations)
print("Loaded ",len(data)," reviews")

# Filter the user graph to include required at least 5 friends and number of reviews in given range
filtered_graph = filter_user_graph(user_graph,5,MAX_FRIENDS,user_review_count,MIN_USER_REVIEWS,MAX_USER_REVIEWS)

# Replace locations external id with internal ones, and remove reviews for locations with less than minimum reivews or for users no in filtered graph
new_locations,new_data,dictionary["poi_to_pid"],dictionary["pid_to_poi"],new_categories,new_address = poi_to_pid(locations,data,poi_reviews_count,MIN_POI_REVIEWS,filtered_graph,categories,address)
print("Remained with ",len(new_data)," reviews and ", len(new_locations), " locations")

# Count number of reviews per user after filtering
reviewed_users = {}
for record in new_data:
	uid = record[0]
	if uid not in reviewed_users:
		reviewed_users[uid] = 0
	reviewed_users[uid] += 1

# Filter the graph once more using the latest number of reviews to retrieve latest list of users
filtered_graph2 = filter_user_graph(filtered_graph,1,MAX_FRIENDS,reviewed_users,1,MAX_USER_REVIEWS)

# Transform the users' data to internal ids
filtered_graph2,dictionary['user_to_uid'],dictionary['uid_to_user'] = user_to_uid(filtered_graph2)
# Replace reviews external user id to internal one
new_data = data_to_uid(new_data,dictionary['user_to_uid'])
print("Final size of data ", len(new_data))
reviews = {}
# Get dictionaries of user and poi reviews - for every user/location there is a list of all words used in original order
reviews["user_reviews"] , reviews["poi_reviews"] = get_user_poi_reviews(new_data)

# Get set of unique words used in reviews
all_words = get_words_used(new_data)
# Load the embedding layer from glove file and create an indexed word dictionary from external to internal id. Filter words not used in actual reviews
new_glove,word_dict = load_word_dictionary(GLOVE_FILE,WORD_EMBEDDING_FILE,all_words)
print("loaded dictionary")

# Transform words used in reviews to internal ids
indexed_reviews = {}
indexed_reviews['user_reviews'] = index_words(reviews["user_reviews"],word_dict)
indexed_reviews['poi_reviews'] = index_words(reviews["poi_reviews"],word_dict)

# Print number of users, POIs and reviews
print(len(indexed_reviews["user_reviews"]),len(indexed_reviews["poi_reviews"]))
print(len(filtered_graph2),len(new_locations))
print(max(list(indexed_reviews["user_reviews"].keys())),max(list(indexed_reviews["poi_reviews"].keys())))
print(max(list(filtered_graph2.keys())),max(list(new_locations.keys())))

# Split data to training,test and validation sets, with negative sampling
train,test,validation = join_reviews_with_split(new_data,SPLIT,VALIDATION_SPLIT,NEGATIVE)
print("Length of train, test, validation:", len(train['user']), len(test['user']), len(validation['user']))

# Set all users and POIs reviews to include the same amount of words
user_words_perc = 3000
poi_words_perc = 3000
# Filter or pad reviews so every user and poi will have the same number of words
indexed_reviews['user_reviews'] = pad_reviews(indexed_reviews['user_reviews'],user_words_perc)
indexed_reviews['poi_reviews'] = pad_reviews(indexed_reviews['poi_reviews'],poi_words_perc)
# Build location graph where directly connected locations are up to 2*POI_MAX_RADIUS far from each other
print("Building graph")
poi_graph = build_location_graph(new_locations,POI_SAMPLE_RATIO,POI_MAX_RADIUS)
# Print the location graph statistics
graph_stats(poi_graph)
# Perform random walk on user and location graph to generate the grpahs to be used in the model
user_sampled = random_walk(filtered_graph2,0.2,15,20,3)
graph_stats(user_sampled)
poi_sampled = random_walk(poi_graph,0.2,15,30,3)
graph_stats(poi_sampled)

# Write all data structures to disk
WORK_DIR = "preprocessed/combined_model/" + str(int(len(new_data)/1000)) + "k_" + str(user_words_perc) + "/"
pathlib.Path(WORK_DIR).mkdir(parents=False, exist_ok=True) 

write_to_pickle(train,WORK_DIR + "train.pkl")
write_to_pickle(test,WORK_DIR + "test.pkl")
write_to_pickle(validation,WORK_DIR + "validation.pkl")
write_to_pickle(indexed_reviews,WORK_DIR + "reviews.pkl")
write_to_pickle(new_glove,WORK_DIR + "glove_file.pkl")
write_to_pickle(validation,WORK_DIR + "validation.pkl")
write_to_pickle(word_dict,WORK_DIR + "word_dictionary.pkl")
write_to_pickle(user_sampled,WORK_DIR + "user_graph.pkl")
write_to_pickle(poi_sampled,WORK_DIR + "poi_graph.pkl")
write_to_pickle(dictionary,WORK_DIR + "user_poi_dictionary.pkl")
write_to_pickle(new_categories, WORK_DIR + "poi_categories.pkl")
write_to_pickle(new_address,WORK_DIR + "poi_address.pkl")

