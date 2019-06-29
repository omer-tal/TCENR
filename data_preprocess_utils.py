import json
import numpy as np
import pickle
import random
import pathlib
import operator
from math import sin, cos, sqrt, atan2, radians
from nltk.corpus import stopwords

TO_SPACE = '.,\t\n'
FILTERED_CHARS = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~\''

def load_user_graph(user_graph_file,limit):
    """
    Loads a user graph from file, and translating it to ids
    """
    user_graph = {}
    user_review_count = {}
    file = open(user_graph_file,encoding='utf-8') 
    # Load the file line by line
    for line in file:
        # Load as json file
        user = json.loads(line)
        user_id = user["user_id"]
        user_review_count[user_id] = int(user["review_count"])
        # Loading the user's friend ids as list
        friends_list = list(user["friends"])
        # Filter users with no friends
        if len(friends_list)>0:
            # Add user to the user graph
            if user_id not in user_graph:
                user_graph[user_id] = set()
            # Add friends to graph
            for friend_id in friends_list:
                # Add new edge
                user_graph[user_id].add(friend_id)
                # Add the friend to the user graph
                if friend_id not in user_graph:
                    user_graph[friend_id] = set()
                # Add the reverse edge
                user_graph[friend_id].add(user_id)
        if len(user_graph)>=limit:
            break
    # Convert set of friends to list
    for user,friends in user_graph.items():
        user_graph[user] = list(friends)
    return user_graph,user_review_count
	
def filter_user_graph(user_graph,min_friends,max_friends,user_review_count,min_reviews,max_reviews):
    """
    Filtering a user graph to include only users with a required amount of friends and reviews
    Connections are removed until only users who fulfills the given requirements are reamined
    """
    change_done = False
    removed={}
    i=1
    # Go until there is iteration with no change in the graph
    curr_graph = user_graph
    while (not change_done):
        new_graph = {}
        # Mark nodes with less edges than the minimum for removal
        for user_id,friends in curr_graph.items():
            if len(friends)>=min_friends and len(friends)<=max_friends and user_id in user_review_count and \
            user_review_count[user_id]>=min_reviews and user_review_count[user_id]<=max_reviews:
                new_graph[user_id] = friends
        # If there are no nodes to remove, we are done
        if len(new_graph)==len(curr_graph):
            change_done = True
        else:
            print("Iteration {} with {} remaining nodes".format(i,len(new_graph)))
            i+=1
            # Update graph to exclude edges to deleted nodes
            for user_id,friends in new_graph.items():
                new_friend_list = []
                for friend in friends:
                    if friend in new_graph:
                        new_friend_list.append(friend)
                new_graph[user_id] = new_friend_list
        curr_graph = new_graph
    return curr_graph
	
def user_to_uid(user_graph):
    """
    Translate user graph external ids to internal ids. Returns the dictionary used for the translation
    """
    dictionary = {}
    reverse_dict = {}
    new_graph = {}
    # Internal index starts at 0
    internal_id = 0
    # Go over the users graph
    for user_id,friends in user_graph.items():
        # Add a new user to the dictionary
        if user_id not in dictionary:
            dictionary[user_id] = internal_id
            reverse_dict[internal_id] = user_id
            internal_id +=1
        # Get internal id for the user
        uid = dictionary[user_id]
        new_graph[uid] = []
        # Translate the edges of the users to internal ids
        for friend_id in friends:
            # Add the friend to the dictionary if first encountered
            if friend_id not in dictionary:
                dictionary[friend_id] = internal_id
                reverse_dict[internal_id] = friend_id
                internal_id += 1
            friend_uid = dictionary[friend_id]
            new_graph[uid].append(friend_uid)
    user_graph = new_graph
    # Return translated graph and dictionary
    return user_graph,dictionary,reverse_dict

def load_poi_file(poi_graph_file,limit):
    """
    Loading the locations of pois from file
    """
    locations = {}
    categories = {}
    address = {}
    file = open(poi_graph_file,encoding='utf-8') 
    i=0
    # Loading locations from file with their coordinates
    for line in file:
        poi = json.loads(line)
        poi_id = poi["business_id"]
        review_count = int(poi["review_count"])
        # Don't add locations with no coordinates or with few reviews
        if (poi["longitude"] != None and poi["latitude"]!=None):
            locations[poi_id] = (float(poi["longitude"]),float(poi["latitude"]))
            i+=1
        categories[poi_id] = poi["categories"]
        city = poi["city"]
        state = poi["state"]
        address[poi_id] = city + "," + state
        if (i==limit):
            print("Stoppig due to limit")
            break
    return locations, categories, address
	
def get_glove_dict(file):
    """
    Load glove dictionary file. 
    Glove is a pre-trained nlp embedding layer from https://nlp.stanford.edu/projects/glove/
    """
    with open(file,encoding='utf-8') as glove:
        return {l[0]: np.asarray(l[1:], dtype="float32") for l in [line.split() for line in glove]}
		
def load_word_dictionary(glove_file,word_embedding,words_set):
    """
    Loads a word semantic dictionary, cleans it and translate every word to intenral id
    Returns the transformed dictionary new_glove and structure to translate internal id to word, words_dict
    """
    glove_to_load = glove_file + str(word_embedding) + "d.txt"
    glove_dictionary = get_glove_dict(glove_to_load)
    # Sorting the words
    raw_words = sorted(list(glove_dictionary.keys()))
    words_dict = {}
    new_glove = {}
    new_glove[0] = np.array([0.0] * word_embedding)
    # Starting index is 1
    counter=1
    # Giving the words indexes by their alphabetic order, after filtering
    for i in range(len(raw_words)):
        #to_filter = False
        # If a word contains a filtered character, remove the whole word from dictionary
        new_word = ""
        old_word = raw_words[i]
        for c in old_word:
            if c not in FILTERED_CHARS:
                new_word += c.lower()
        # If not, add it to a dictionary from word to index and to a new word dictionary with the same values as the old one
        if len(new_word)>0 and new_word in words_set and new_word not in words_dict:
            words_dict[new_word] = counter
            new_glove[counter] = glove_dictionary[old_word]
            counter += 1
    return list(new_glove.values()),words_dict
	
def clean_review(review,to_filter):
    """
    Removes characters from the review string that are in the to_filter string
    The output is a filtered, lower case representation of the review, splitted to words by space
    """
    new_review = ""
    # Go over every character and check weather to filter it
    for c in review:
        if c in TO_SPACE:
            c=' '
        if c not in to_filter:
            # Lower the unfiltered character
            new_review += c.lower()
    # SPlit to words and return
    new_review = new_review.split(" ")
    review_stripped = []
    for word in new_review:
        new_word = word.strip()
        if len(new_word)>0:
            review_stripped.append(new_word)
    return new_review
	
def translate_words_to_index(review,word_dict):
    """
    Translate all words in the review to indexes, based on the translateting data strucutre word_dict
    """
    new_review = []
    for word in review:
        # If the word is not in the dictionary, put 0 in the review
        if word not in word_dict:
            new_review.append(0)
        else:
            new_review.append(word_dict[word])
    return new_review

def get_user_poi_reviews(data):
    """
    Combines all the reviews of a user (and poi) to one list, accessible by the user (poi) internal id
    """
    user_reviews = {}
    poi_reviews = {}
    # Go over all reviews
    for uid,pid,_,review in data:
        # If the user was not processed before, add it        
        if (uid not in user_reviews):
            user_reviews[uid] = []
        # Concat the previous user words with the ones from this review
        user_reviews[uid] += review
        # If the poi was not processed before, add it
        if (pid not in poi_reviews):
            poi_reviews[pid] = []
        # Concat the words previously used to review the place with words from this review
        poi_reviews[pid] += review
    return (user_reviews,poi_reviews)
	
def create_train_test_dict():
    """
    Creating a dictionary represents a training instance
    Has three lists - one for user ids, one for poi ids and one for the review score
    """
    dictionary = {}
    dictionary["user"] = []
    dictionary["poi"] = []
    dictionary["review"] = []
    return dictionary

def add_record_to_dict(dictionary,record):
    """
    Adding a record to the dictionary (train,test)
    """
    dictionary["user"].append(record[0])
    dictionary["poi"].append(record[1])
    dictionary["review"].append(record[2])

def get_user_reviewed(data):
    """
    Get a dictionary with all the reviewed locations of a user and a distinct list of all locations
    """
    user_reviewed_pois = {}
    # Using set for distinctive values
    all_pois = set()
    # Going over all reviews
    for record in data:
        uid = record[0]
        pid = record[1]
        # If the user is not in the dictioary, add an empty list for it
        if uid not in user_reviewed_pois:
            user_reviewed_pois[uid] = []
        # Add the new location to the user's list
        user_reviewed_pois[uid].append(pid)
        all_pois.add(pid)
    # Transform the set to a list and return it along with the user dictionary
    return user_reviewed_pois,list(all_pois)
	
def join_reviews_with_split(data,split,val_split,num_negatives):
    """
    Concatenate all reviews for a user/poi, split to training and test set and perfrom a negative sampling
    data - the input data, as list of tuples
    split - percentage to split by
    num_negatives - number of negative sampled instances per positive one
    """
    # Creating data structures for training and test sets
    train = create_train_test_dict()
    test = create_train_test_dict()
    val = create_train_test_dict()
    # Get the reviewed places per user and all available pois, used for negative sampling
    user_reviewed_pois,all_pois = get_user_reviewed(data)
    # Go over all records
    for record in data:
        uid = int(record[0])
        pid = record[1]
        # Choosing training-test set split
        set_choice = random.random()
        # The record and negative samples will be added to training/test set based on the random split
        if (set_choice>split):
            set_choice_val = random.random()
            if (set_choice_val>val_split):
                split_choice = train
            else:
                split_choice = val
        else:
            split_choice = test
        # Add the positive record to the data structure
        add_record_to_dict(split_choice,record)
        # Negative sampling
        negatives = []
        # Add num_negatives instances
        for i in range(num_negatives):
            is_negative = False
            # Go over until a valid negative sample is found
            while (is_negative==False):
                # Get a random poi
                negative_pid_idx = random.randint(0,len(all_pois)-1)
                negative_pid = all_pois[negative_pid_idx]
                # A random poi is a valid negative sample if the user never reviewed the place, 
                # it's not a negative sample already and the poi is not to be filtered
                if (negative_pid not in user_reviewed_pois[uid] and negative_pid not in negatives):
                    is_negative = True
                    negatives.append(negative_pid)
        # Add records for the negative samples
        for neg_pid in negatives:
            add_record_to_dict(split_choice,(uid,neg_pid,0))
    return train,test,val
	
def get_counts(reviews_dict):
    """
    Return a list with the number of reviews per user/poi
    """
    review_counts = []
    for _,reviews in reviews_dict.items():
        review_counts.append(len(reviews))
    return review_counts
	
def pad_reviews(reviews,size):
    """
    Given a dictionary of reviews per user/poi and a given size, the function returns a padded dictionary
    Padding is either removing words or adding padding value (0)
    """
    padded_reviews={}
    # Go over the reviews of users/items
    for key,revs in reviews.items():
        # Determine the padding size
        to_pad = size - len(revs)
        # If there are more words than the required size, remove the tailing words
        if to_pad<0:
            revs = revs[:size]
        # Otherwise add words so the number of words will reach the required size
        else:
            revs = revs + [0] * to_pad
        # Add the padded review words to the dictionary
        padded_reviews[key] = revs
    return padded_reviews
	
def distance(a, b):
    '''
    Parameter: a, b are two tuples in form of (latitude, longtitude)
    Return: The distance between a and b in kimlometers
    '''
    # approximate radius of earth in km
    R = 6373.0
    
    # Get latitudes and longitudes
    lat1 = radians(a[0])
    lon1 = radians(a[1])
    lat2 = radians(b[0])
    lon2 = radians(b[1])
    
    # Calculate the differences between the locations
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Get the geographical distance between locations in kilometers
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
	
def build_location_graph(locations,sample_ratio,max_radius):
    """
    Construct an initial graph based on businesses geographical locations
    base points are randomly selected using the given sample ratio
    edges are constructed between base points and locations within the max_radius
    """
    graph = {}
    # Sampling base points to start the graph from, based on a given sampling ratio
    base_nodes_size = int(len(locations)*sample_ratio)
    base_points = random.sample(locations.keys(), base_nodes_size)
    print("number of base points = {}".format(len(base_points)))
    # Counter used to calculate the graph's density
    # For each base points, add all nodes connected to it and closer than maximum radius
    for base_node in base_points:
        nodes = []
        # Check all POIs for proximaty
        for candidate_node, coordinates in locations.items():
            # If the location is within a given radius from the base node, it will be added to the graph as a node
            if distance(locations[base_node],coordinates) < max_radius:
                nodes.append(candidate_node)
        # Add edges for all pairwise combinations of nodes connected to the base node
        for i in range(len(nodes)-1):
            for j in range(i+1,len(nodes)):
                # Add a node to the graph if it's not there already
                if nodes[i] not in graph:
                    # Edges are represented as a set to ensure distinct edges
                    graph[nodes[i]] = set()
                # Add a new edge
                graph[nodes[i]].add(nodes[j])
                # The same but backwards
                if nodes[j] not in graph:
                    graph[nodes[j]] = set()
                graph[nodes[j]].add(nodes[i])
    # Transform the distinct set of edges to a list
    for node,nodes in graph.items():
        graph[node] = list(nodes)
    return graph
	
def random_walk(graph,path_portion,path_length,num_samples,window_size):
    """
    Construct a new graph using random walk on a given graph
    Done by taking number of paths using the given path portion
    Every path is constructed with a random user and then his friends until reaching a required path_length
    Then a random walk over the path is done by selecting random node in the window size, until required number of samples is reached
    """
    # Determining how many paths will be constructed
    path_count = int(len(graph) * path_portion)
    sampled_graph = {}
    # How many paths should be constructed - based on the graph size and required portion
    for i in range(path_count):
        path = []
        # Build x paths starting from a random walk and choosing a random friend every time
        for j in range(path_length):
            # When starting a path - selecting a random user to start with
            if len(path) == 0:
                path.append(list(graph.keys())[random.randrange(len(graph))])
            # For an existing path - if the user has friends and wasn't filtered, add one of his friends to the graph randomly
            else:
                if path[len(path)-1] not in graph or len(graph[path[len(path)-1]]) == 0:
                    break
                candidate = graph[path[len(path)-1]]
                path.append(candidate[random.randrange(len(candidate))])
        if len(path) > 1:
            # Sample tuples from the path that have distance smaller than window size
            for k in range(num_samples):
                while True:
                    # sample a random tuple from the path
                    node_pair = random.sample(path, k=2)
                    # If they are closer than the window size add them to the sampled graph
                    # The loop will stop if the tuple is far
                    if abs(path.index(node_pair[0]) - path.index(node_pair[1])) < window_size:
                        break
                    # If a node is not in the graph, add it
                    if node_pair[0] not in sampled_graph:
                        sampled_graph[node_pair[0]] = []
                    # If the edge does not exists already, add it
                    if node_pair[1] not in sampled_graph[node_pair[0]]:
                        sampled_graph[node_pair[0]].append(node_pair[1])
                    # The same but backwards
                    if node_pair[1] not in sampled_graph:
                        sampled_graph[node_pair[1]] = []
                    if node_pair[0] not in sampled_graph[node_pair[1]]:
                        sampled_graph[node_pair[1]].append(node_pair[0])
    return sampled_graph
	
def graph_stats(graph):
    """
    Prints main statistics about the given graph - number of nodes, density and average edges per node
    """
    total_nodes = len(graph)
    total_edges = 0
    for key,values in graph.items():
        total_edges += len(values)
    total_edges /= 2
    print("Total nodes: ", total_nodes, " Total edges: ",int(total_edges))
    print("Graph density: ",(total_edges / ((total_nodes * (total_nodes-1)) / 2)))
    print("Avg edges per node ",total_edges/total_nodes)


def write_to_pickle(data,dir):
    """
    Writing a data structure to disk in pickle format
    """
    print("Writing training to {}".format(dir))
    with open(dir,'wb') as file:
        pickle.dump(data,file)

def load_data(review_file,filtered_chars, limit,user_graph,locations):
    """
    Loads the data from a given file up to limit (top sampling). By default limit is maximum
    Filters reviews if user not in user graph or poi not in locations
    Remained reviews are cleaned and returned in the form of list
    Additional return values - number of reviews per user and poi
    """
    lst = []
    i=0
    # Open the reveiws file
    with open(review_file,encoding='utf-8') as file:
        poi_reviews_count = {}
        user_reivews_count = {}
        # Read rows until reaching a limit
        while (i<limit):
            line = file.readline()
            # If reached the end of file, stop
            if line == '':
                break
            # Each line is a json record
            record = json.loads(line)
            user_id = record["user_id"]
            poi_id = record["business_id"]
            # If the user is not in the graph and the poi not in locations collection, don't load the review
            if (user_id in user_graph and poi_id in locations):
                review = record["text"]
                # Clean the textual review
                cleaned_review = clean_review(review,filtered_chars)
                # If the cleaned review is empty, don't load the review
                if (len(cleaned_review)>0):
                    # Count number of reviews for POIs and users
                    if poi_id not in poi_reviews_count:
                        poi_reviews_count[poi_id] = 0
                    poi_reviews_count[poi_id]+=1
                    if user_id not in user_reivews_count:
                        user_reivews_count[user_id] = 0
                    user_reivews_count[user_id]+=1
                    # Add the review 
                    lst.append((user_id,poi_id,1,cleaned_review))
                    i+=1
    # Return the list of reviews and number of reviews for every poi aand user
    return lst,poi_reviews_count,user_reivews_count
	
def poi_to_pid(locations,data,poi_reviews_count,min_reviews,user_graph,categories,address):
    """
    Transform loaded POIs from external to internal ids in the location collection in loaded reviews
    Also filters locations and reviews with less than minimum reviews and that their users not in the user graph
    """
    dictionary = {}
    reverse_dict = {}
    new_locations = {}
    new_categories = {}
    new_address = {}
    # Internal id starts at 0
    internal_id = 0
    new_data = []
    # Go over all reviews
    for record in data:
        poi = record[1]
        user_id = record[0]
        # If the poi doesn't have enough reviews or the user not in the user grpah, remove it
        if (poi_reviews_count[poi]>=min_reviews and user_id in user_graph):
            # Add poi to translation dictionary from external to internal id
            if poi not in dictionary:
                dictionary[poi] = internal_id
                reverse_dict[internal_id] = poi
                new_locations[internal_id] = locations[poi]
                new_categories[internal_id] = categories[poi]
                new_address[internal_id] = address[poi]
                internal_id +=1
            pid = dictionary[poi]
            # Add the transformed review to new reviews list
            new_data.append((user_id,pid,record[2],record[3]))
    return new_locations,new_data,dictionary,reverse_dict,new_categories,new_address

def get_words_used(data):
    """
    Get set of disticnt words used by users to describe locations
    """
    all_words = set()
    # Go over reviews
    for record in data:
        # Go over every word in the reivew
        for word in record[3]:
            # Add word to set
            all_words.add(word)
    return all_words

def data_to_uid(data,user_to_uid):
    """
    Transform reviews to include internal user id instead of external id
    """
    new_data = []
    # Go over all reviews
    for record in data:
        user_id = record[0]
        # If the user is not in the final collection, remove the review
        if user_id in user_to_uid:
            new_data.append((user_to_uid[user_id],record[1],record[2],record[3]))
    return new_data

def index_words(reviews,word_dict):
    """
    Transform words used in reviews to indexes
    """
    new_reviews = {}
    # Go over all reviews
    for uid,words in reviews.items():
        words_count = []
        # Go over all words in the review
        for word in words:
            # Replace word with an index
            if word in word_dict:
                indexed = word_dict[word]
            # If the word has no index, replace it with blank - 0
            else:
                indexed = 0
            words_count.append(indexed)
        new_reviews[uid] = words_count
    return new_reviews
