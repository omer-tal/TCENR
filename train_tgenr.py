import numpy as np
import pickle
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Flatten,Input, Reshape, Embedding,merge, Dropout, GRU, Bidirectional,AveragePooling1D
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.layers.merge import Add, Dot, Concatenate,Multiply
from keras.optimizers import Adam
import gc
import random
import time
import os
import sys

# Setting the input directory
INPUT_SIZE = 1711
WORDS = sys.argv[1]
PATH = "preprocessed/combined_model/"+ str(INPUT_SIZE) + "k_" + WORDS + "/"
reviews_file="reviews"


# In[2]:


# Loading the preprocessed data
print("Loading training file from {}".format(PATH))
with open(PATH + "train.pkl",'rb') as file:
    train = pickle.load(file)
print("Loading test file from {}".format(PATH))
with open(PATH + "test.pkl",'rb') as file:
    test = pickle.load(file)
print("Loading validation file from {}".format(PATH))
with open(PATH + "validation.pkl",'rb') as file:
    validation = pickle.load(file)
print("Loading reviews file from {}".format(PATH))
with open(PATH + reviews_file +".pkl",'rb') as file:
    reviews = pickle.load(file)
print("Loading user graph file from {}".format(PATH))
with open(PATH + "user_graph.pkl",'rb') as file:
    user_graph = pickle.load(file)
print("Loading POI graph file from {}".format(PATH))
with open(PATH + "poi_graph.pkl",'rb') as file:
    poi_graph = pickle.load(file)
print("Loading word embedding file from {}".format(PATH))
with open(PATH + "glove_file.pkl",'rb') as file:
    dictionary = pickle.load(file)
print("Done loading")

# Printing the sizes of input data
print("Training samples:{} , Test samples:{}, Validation samples:{}".format(len(train['review']),len(test['review']),
                                                                            len(validation['review'])))
print("Sample record:" , train['user'][0], train['poi'][0] , train['review'][0])
print("Review scores are between {} and {}".format(min(train['review']),max(train['review'])))
print("Total users:", len(reviews['user_reviews']), " max user:" ,max(list(reviews['user_reviews'].keys())))
print("Total places:", len(reviews['poi_reviews']), " max poi: ",max(list(reviews['poi_reviews'].keys())))
print("Number of nodes in user graph:",len(user_graph))
print("Number of nodes in poi graph:",len(poi_graph))
total_users = len(reviews['user_reviews'])
total_pois = len(reviews['poi_reviews'])

def get_reviews(data,reviews):
    """
    Transform textual reviews from dictionary by user/poi to correspond every user-poi interaction
    """
    review_list=[]
    # Go over list of users/pois used in user-poi interaction
    for key in data:
        # Retrieve all words ever used by this user/poi and add it to output
        review_list.append(reviews[key])
    return review_list

# Transform textual reviews from dictionary of train and validation data to match the user-location entries sequence
embedded_user_reviews = get_reviews(train['user'],reviews['user_reviews'])
embedded_poi_reviews = get_reviews(train['poi'],reviews['poi_reviews'])
embedded_user_reviews_val = get_reviews(validation['user'],reviews['user_reviews'])
embedded_poi_reviews_val = get_reviews(validation['poi'],reviews['poi_reviews'])

def hot_one_context(node_in_reviews,graph,total_nodes):
    """
    Transform the user/poi graph to binary vector corresponding to each user-poi interaction
    """
    total_reviews = len(node_in_reviews)
    # encoded list has a row for every review and column for every user/poi
    print("creating matrix in size ({},{})".format(total_reviews,total_nodes))
    encoded = np.zeros((total_reviews,total_nodes),dtype=np.byte)
    i=0
    # Iterate over all reviews - node is the user/poi id for current review
    for node in node_in_reviews:
        if node in graph:
            # Go over all connected users/pois
            for connected_node in graph[node]:
                encoded[i][connected_node] = 1
        # Increasing the review counter by 1
        i+=1
    return encoded



# Get the number of users and locations to create the binarized graphs
total_users = max(list(reviews['user_reviews'].keys()))+1
total_pois = max(list(reviews['poi_reviews'].keys()))+1

# Creating binarized grpahs for users and locations in training and validation sets
user_graph_emb = np.array(hot_one_context(train['user'],user_graph,total_users))
user_graph_emb_val = np.array(hot_one_context(validation['user'],user_graph,total_users))
poi_graph_emb = np.array(hot_one_context(train['poi'],poi_graph,total_pois))
poi_graph_emb_val = np.array(hot_one_context(validation['poi'],poi_graph,total_pois))


# # Recommender model

def generate_instances(batch_size,u_input,p_input,r_input,u_reviews,p_reviews,u_graph_emb,p_graph_emb,total_users,total_pois,
                       u_emb_size,p_emb_size,emb_size, training,contextual,textualmodel):
    """
    Instance generator used by generator functions in training and evaluation. 
    It receives input and output vectors, batch size and parameters identifying which model is it
    The ouput is a list in size batch_size of user-poi interactions with user/poi/user words/poi words as input, depending on the model,
    and preference/user context/poi context as output, depending on the model
    """
    # Go as long as the calling function asks to
    while 1:
        # The input and output records will be in dictionary representing the corresponding input/output layer name
        features = {}
        features['user_input'] = []
        features['poi_input'] = []
        features['user_reviews_input'] = []
        features['poi_reviews_input'] = []
        labels = {}
        labels['review_output'] = []
        labels['user_context_output'] = []
        labels['poi_context_output'] = []
        # Add instances matching the batch size
        for j in range(batch_size):
            # Select random instance to use
            i = random.randint(0,len(u_input)-1)
            # If the model is based on context, it uses user and poi as input
            if (contextual>=1):
                features['user_input'].append(u_input[i])
                features['poi_input'].append(p_input[i])
            # If the model is based on text, it uses written reviews by user and poi as input
            if (textualmodel):
                features['user_reviews_input'].append(u_reviews[i])
                features['poi_reviews_input'].append(p_reviews[i])
            # If there is training or evaluation, an output is required
            if (training):
                # The score is always used
                labels['review_output'].append(r_input[i])
                # If the model is based on context, the output includes user and POI graphs
                if (contextual==2):
                    labels['user_context_output'].append(u_graph_emb[i])
                    labels['poi_context_output'].append(p_graph_emb[i])
        # Transform inputs to numpy arrays
        x = {}
        for key,data in features.items():
            if len(data)>0:
                x[key] = np.array(data)
        # Transform output to numpy arrays
        y = {"review_output": np.array(labels['review_output'])}
        if (contextual==2):
            y["user_context_output"] = np.array(labels["user_context_output"])
            y["poi_context_output"] = np.array(labels["poi_context_output"])
        # If training or evaluation, input and output pairs are required
        if (training):
            yield x,y
        # For testing, only input is required
        else:
            yield x

# Setting model layer structure
WORD_EMBEDDING = 50
REVIEWS_HIDDEN_LAYER_SIZE = 32
CONTEXT_LAYERS = [32,16]
CONTEXT_REG_LAYERS = [0,0]
# Number of feature maps for CNN
FILTERS = 3
# Size of window
KERNEL_SIZE = 10
# The window change in every iteration
STRIDE_SIZE = 3
# Contextual embedding layer size
CONTEXT_LATENT_DIMS = 10
# Setting hyperparameter values
BATCH_SIZE = 512
EPOCHS = 50
L_RATE = 0.005
# Verbose variable controls the level of output printing
VERBOSE = 2
GRU_UNITS = list(map(int, sys.argv[2].strip('[]').split(',')))

# Setting the size for the user and poi reviews embedding layer
user_embedded_size = len(embedded_user_reviews[0])
poi_embedded_size = len(embedded_poi_reviews[0])

# Setting the number of training and validation steps to be the input size divided by batch size
train_epoch_steps = (len(train['user'])//BATCH_SIZE)
val_epoch_steps = (len(validation['user'])//BATCH_SIZE)

# Defines the model used
# For TCENR set contextual=2, textualmodel=True and mf=0
contextual = 2
textualmodel = True
mf = 0
RNN = int(sys.argv[3])
print("RNN type {}, layers {}".format(RNN,GRU_UNITS))
POOLING_SIZE = int(sys.argv[4])
print("pooling: {}".format(POOLING_SIZE))
if (RNN==0):
    CONTEXT_LAYERS = GRU_UNITS
    CONTEXT_REG_LAYERS = [0]*len(CONTEXT_LAYERS)
print("MLP layers:{}".format(CONTEXT_LAYERS))

# Define the model name used in saving it to disk
RUN_NUM = sys.argv[5]
    
model_str = str(WORDS) + "_" + str(RNN) + "_" + str(GRU_UNITS) + "_" + str(POOLING_SIZE) + "_" + RUN_NUM
print(model_str)

# Model path is the output directory of the model and checkpoint path is the directory for checkpoints during training
MODEL_PATH = "runs/" +model_str + ".h5"
CHECKPOINT_PATH = "checkpoints/"+ model_str + ".h5"

# Dynamically define input,output loss function, weights and early stop measure based on model
inputs = []
losses = ['binary_crossentropy']
loss_weights = [1.]
contextual_outputs = []
measure = 'acc'

# Defining structure for model
if contextual>=1:
    # Input layers - 2 one-hot encoding for user and poi
    user_input_layer = Input(shape=(1,), name = 'user_input')
    poi_input_layer = Input(shape=(1,), name = 'poi_input')
    
    inputs = inputs + [user_input_layer,poi_input_layer]
    
    # Embedding layers for the user and poi input data. Results in the latent factors for each user/poi
    user_embedding = Embedding(input_dim = total_users
                               , output_dim = CONTEXT_LATENT_DIMS 
                               ,embeddings_initializer='normal', embeddings_regularizer = l2(CONTEXT_REG_LAYERS[0]), 
                               input_length=1,name='user_embedding') (user_input_layer)
    poi_embedding = Embedding(input_dim = total_pois#len(train['poi'])
                              , output_dim = CONTEXT_LATENT_DIMS
                              ,embeddings_initializer='normal', embeddings_regularizer = l2(CONTEXT_REG_LAYERS[0]), 
                              input_length=1,name='poi_embedding') (poi_input_layer)
    # Flattening the embdedding layer
    user_latent_flattened = Flatten(name='user_latent_flattened')(user_embedding)
    poi_latent_flattened = Flatten(name='poi_latent_flattened')(poi_embedding)
    # If contextual==2, add contextual regularization as output
    if (contextual==2):      
        # Softmax output for the contextual graphs, using sigmoid activation function
        user_context_output = Dense(total_users,name='user_context_output', activation='sigmoid',
                                     kernel_initializer='lecun_uniform')(user_latent_flattened)
        poi_context_output = Dense(total_pois,name='poi_context_output', activation='sigmoid',
                                    kernel_initializer='lecun_uniform')(poi_latent_flattened)
        # Add the output, corresponding loss functions and weights for the two in training
        contextual_outputs = contextual_outputs + [user_context_output,poi_context_output]
        losses = losses + ['categorical_crossentropy','categorical_crossentropy']
        # Reduced weights for the categorical crossentropy functions
        loss_weights = loss_weights + [0.1,0.1]
        # Setting the measure name to the accuracy of the preferences
        measure = 'review_output_acc'
    # If this is a MF based model
    if (mf==1):
        # Perform dot product of the two embedded layers
        merged_input = Multiply(name='merged_input')([user_latent_flattened, poi_latent_flattened])
        hidden_layer = merged_input
    # If this model has hidden layers on top of concatentation
    else:
        # Merging the user and poi for the input and reviews
        merged_input = Concatenate(name='merged_input')([user_latent_flattened, poi_latent_flattened])
        hidden_layer = merged_input
        # Defining the hidden layers for user-poi input interaction learning
        for i in range(len(CONTEXT_LAYERS)):
            # Each hidden layer defines to use the ReLU aactivation function
            hidden_layer = Dense(CONTEXT_LAYERS[i],name='hidden_' + str(i), activation='relu', 
                                 kernel_regularizer= l2(CONTEXT_REG_LAYERS[i]))(hidden_layer)
# Setting parameters if textual modeling is used
if textualmodel:
    # The input vectors are the words used in users' and POIs' reviews
    user_reviews_input = Input(shape=(user_embedded_size, ), name = 'user_reviews_input')
    poi_reviews_input = Input(shape=(poi_embedded_size,), name = 'poi_reviews_input')
    inputs = inputs + [user_reviews_input,poi_reviews_input]
    
    # Setting the pretrained embedding layer developed by GloVE
    user_reviews_embedding = Embedding(len(dictionary),WORD_EMBEDDING, weights=[np.array(dictionary)],input_length=user_embedded_size,
                                       trainable=False) (user_reviews_input)
    poi_reviews_embedding = Embedding(len(dictionary),WORD_EMBEDDING,weights=[np.array(dictionary)],input_length=poi_embedded_size,
                                      trainable=False) (poi_reviews_input)
    
    if RNN==1:
        user_gru = Bidirectional(GRU(GRU_UNITS[0],return_sequences=True))(user_reviews_embedding)
        user_conv = GRU(GRU_UNITS[1],return_sequences=True)(user_gru)
        poi_gru = Bidirectional(GRU(GRU_UNITS[0],return_sequences=True))(poi_reviews_embedding)
        poi_conv = GRU(GRU_UNITS[1],return_sequences=True)(poi_gru)
        # Pooling layers for the user and poi reviews
        user_pooling = MaxPooling1D(pool_size=2,name='user_pooling')(user_conv)
        poi_pooling = MaxPooling1D(pool_size=2,name='poi_pooling')(poi_conv)
    elif RNN==2:
        user_gru = Bidirectional(GRU(GRU_UNITS[0],return_sequences=True))(user_reviews_embedding)
        user_conv = Bidirectional(GRU(GRU_UNITS[1],return_sequences=True))(user_gru)
        poi_gru = Bidirectional(GRU(GRU_UNITS[0],return_sequences=True))(poi_reviews_embedding)
        poi_conv = Bidirectional(GRU(GRU_UNITS[1],return_sequences=True))(poi_gru)
        # Pooling layers for the user and poi reviews
        user_pooling = MaxPooling1D(pool_size=2,name='user_pooling')(user_conv)
        poi_pooling = MaxPooling1D(pool_size=2,name='poi_pooling')(poi_conv)
    elif RNN==3:
        user_gru = Bidirectional(GRU(GRU_UNITS[0],return_sequences=True))(user_reviews_embedding)
        user_conv = GRU(GRU_UNITS[1],return_sequences=True)(user_gru)
        poi_gru = Bidirectional(GRU(GRU_UNITS[0],return_sequences=True))(poi_reviews_embedding)
        poi_conv = GRU(GRU_UNITS[1],return_sequences=True)(poi_gru)
        # Pooling layers for the user and poi reviews
        user_pooling = AveragePooling1D(pool_size=2,name='user_pooling')(user_conv)
        poi_pooling = AveragePooling1D(pool_size=2,name='poi_pooling')(poi_conv)
    else:
        # Convulational layers for the user and poi reviews
        user_conv = Conv1D(filters=FILTERS,kernel_size=KERNEL_SIZE,activation="relu",name='user_conv')(user_reviews_embedding)
        poi_conv = Conv1D(filters=FILTERS,kernel_size=KERNEL_SIZE,activation="relu",name='poi_conv')(poi_reviews_embedding)
        # Pooling layers for the user and poi reviews
        user_pooling = MaxPooling1D(pool_size=2,name='user_pooling')(user_conv)
        poi_pooling = MaxPooling1D(pool_size=2,name='poi_pooling')(poi_conv)
    # Flattening the results of the two pooling layers
    user_pooling_flattened = Flatten(name='user_pooling_flattened')(user_pooling)
    poi_pooling_flattened = Flatten(name='poi_pooling_flattened')(poi_pooling)
    
    # Dropout layer to avoid overfitting, where 20% of the network is kept at every iteration
    user_reviews_dropout = Dropout(0.2,name='user_reviews_dropout')(user_pooling_flattened)
    poi_reviews_dropout = Dropout(0.2,name='poi_reviews_dropout')(poi_pooling_flattened)

    # Hidden layers for the flattened vector of the pooling layers
    user_reviews_dense = Dense(REVIEWS_HIDDEN_LAYER_SIZE, activation="relu",name='user_reviews_dense')(user_reviews_dropout)
    poi_reviews_dense = Dense(REVIEWS_HIDDEN_LAYER_SIZE, activation="relu",name='poi_reviews_dense')(poi_reviews_dropout)

    # Merge the user and poi written reviews representation using concat
    merged_reviews = Concatenate(name='merged_reviews')([user_reviews_dense,poi_reviews_dense])

# If the output is based on combining the model
if contextual>=1 and textualmodel:
    # Add an hidden layer to model the merge result of reviews
    hidden_layer_reviews = Dense(CONTEXT_LAYERS[-1],name='hidden_reviews',activation='relu',
                         kernel_initializer='lecun_uniform')(merged_reviews)
    # Combine the two models by concat
    combining_models = Concatenate(name='combining_models')([hidden_layer,hidden_layer_reviews])
    # Add additional layer to learn the combination of the two models
    review_output = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'review_output')(combining_models)
# If the model is based on contextual and textual data, the output is a sigmoid transformation of the last hidden layer
elif contextual>=1:
    review_output = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'review_output')(hidden_layer)
# If the model is textual, combine dot product and concatenation (no factorization machines layer in keras)
elif textualmodel:
    # Add a dense layer on top of the concatentation layer
    reviews_merged_layer = Dense(1)(merged_reviews)
    # Perform dot product on the learned user and poi reviews
    reviews_dot_product = Dot(axes=1,name='reviews_dot_product')([user_reviews_dense, poi_reviews_dense])
    # Add the concatentation and dot product combination and model it using one layer
    reviews_merged_with_dot_product = Add(name='reviews_merged_with_dot_product')([reviews_merged_layer, reviews_dot_product])
    # Output tranformed to the range of [0,1] using sigmoid
    review_output = Dense(1,activation='sigmoid',name='review_output')(reviews_merged_with_dot_product)

model_outputs = [review_output] + contextual_outputs
monitor = 'val_'+measure

# Define the model using Adam to optimize it
model = Model(inputs=inputs,
              outputs=model_outputs)
# Metrics used are accuracy and mse
model.compile(optimizer=Adam(lr=L_RATE), 
              loss=losses, 
              metrics=['accuracy','mse'],
              loss_weights=loss_weights
             )
# Define early stopping when the validation accuracy does not improve by more than 0.001 over 10 epochs
early_stop = EarlyStopping(monitor=monitor, min_delta=0.001, patience=10, verbose=VERBOSE, mode='max')
# Save the model every time the validation accuracy improves
checkpoint = ModelCheckpoint(CHECKPOINT_PATH,monitor=monitor,save_best_only=True, verbose=VERBOSE, 
                             save_weights_only=False, period=1)
# Print the model structure
model.summary()

# Train the model using fit generator. Each step is defined to be in batch size, and each epoch is evaluated using the
# validation set.
history = model.fit_generator(generate_instances(BATCH_SIZE,train['user'],train['poi'],train['review'],embedded_user_reviews,
                                       embedded_poi_reviews,user_graph_emb,poi_graph_emb,total_users,total_pois,
                                                 user_embedded_size,poi_embedded_size,WORD_EMBEDDING,True,contextual,textualmodel),
                              validation_data = generate_instances(BATCH_SIZE,validation['user'],validation['poi'],
                                                                   validation['review'],embedded_user_reviews,
                                                                   embedded_poi_reviews,user_graph_emb_val,poi_graph_emb_val,
                                                                   total_users,total_pois,user_embedded_size,
                                                                   poi_embedded_size,WORD_EMBEDDING,True,contextual,textualmodel),
                              steps_per_epoch=train_epoch_steps , validation_steps=val_epoch_steps,
                              epochs=EPOCHS, verbose=VERBOSE, callbacks = [early_stop,checkpoint])

# Saving the model to disk for later use
model.save(MODEL_PATH)

# Define the data structures for the test set - number of test instances, embedded contextual graphs and transformed written
# reviews
total_test_samples = int(len(test['review']))
user_graph_emb_tst = np.array(hot_one_context(test['user'],user_graph,total_users))
poi_graph_emb_tst = np.array(hot_one_context(test['poi'],poi_graph,total_pois))
embedded_user_reviews_test = get_reviews(test['user'],reviews['user_reviews'])
embedded_poi_reviews_test = get_reviews(test['poi'],reviews['poi_reviews'])

# Evaluate the model to return accuracy and MSE for the test set
results = model.evaluate_generator(generate_instances(BATCH_SIZE,test['user'],test['poi'],test['review'],
                                                      embedded_user_reviews_test,embedded_poi_reviews_test,user_graph_emb_tst,
                                                      poi_graph_emb_tst,total_users,total_pois,user_embedded_size,
                                                      poi_embedded_size,WORD_EMBEDDING,True,contextual,textualmodel),
                                   steps=(total_test_samples/BATCH_SIZE))

# Print evaluation results
for i in range(len(results)):
    print(model.metrics_names[i], ":", results[i])

# # Calculating precision and recall manually
from operator import itemgetter
def precision_recall_accuracy(users, pois, actuals, predictions,k,threshold):
    """
    The function calcualtes the top k precision and recall in addition to standard accuracy
    Predictions are classified to 0,1 by comparing to the given threshold
    """
    # Listing a tuple of (poi,actual,prediction) for each user
    user_predictions = {}
    # Count correct predictions
    correct_predictions = 0
    # Go over all test instances
    for i in range(len(predictions)):
        # Test if it is true positive or true negative
        if ((predictions[i] > threshold and actuals[i]==1) or (predictions[i] <= threshold and actuals[i]==0)):
            correct_predictions+=1
        uid = users[i]
        pid = pois[i]
        # Save the prediction and actual results for each user
        if uid not in user_predictions:
            user_predictions[uid] = []
        user_predictions[uid].append((pid,actuals[i],predictions[i]))
    # Calculate accuracy as all true positive and true negatives divided by total instances
    accuracy = (correct_predictions * 1.0) / (len(predictions) * 1.0)
    # Counters for average precision and recall
    total_users = 0
    precision = 0.0
    recall = 0.0
    i=0
    # Iterate over all users to calculate individual precision and recall 
    for uid,predictions in user_predictions.items():
        # Sorting the POIs by the predicted score
        predictions = sorted(predictions, key=itemgetter(2), reverse=True)
        # Sum the relevant instances for user as those with actual=1
        relevant = sum(actual>threshold for (_, actual,_) in predictions) * 1.0
        # Sum recommended instances for user as those with higher value then threshold in top k
        recommended = sum((predicted > threshold) for (_,_,predicted) in predictions[:k]) * 1.0
        # Sum relevant and recommended instances for user as those with actual=1 and prediction higher than threshold in top k
        both = sum((actual>threshold) and (predicted > threshold) for (_,actual,predicted) in predictions[:k]) * 1.0
        # Don't count users with no relevant or recommended items
        if relevant>0 and recommended>0:
            total_users += 1
            precision += both / recommended
            recall += both / relevant
    # Calculate precision and recall as average across all users with relevant and recommended items
    total_users = total_users * 1.0
    precision = precision / total_users
    recall = recall / total_users
    return precision, recall, accuracy

# Retrieve the model predictions for the test set
predictions = model.predict([np.array(test['user']),np.array(test['poi']),
                             np.array(embedded_user_reviews_test), np.array(embedded_poi_reviews_test)]
                             ,verbose=VERBOSE)

# Calculate precision and recall for top 10 recommendations where the treshold is 0.5
precision, recall, accuracy = precision_recall_accuracy(test['user'],test['poi'],test['review'], predictions[0],10,0.5)
print("precision@10:"+str(precision))
print("recall@10:"+str(recall))
print("accuracy:"+str(accuracy))