To run TCENR:

1. run "data_preprocess.py" to preprocess the data. Its input structure is based on three files from the Yelp dataset: "review.json" for user reviews, "user.json" for user data and "business.json" for item data. In addition, it requires a textual embedding file. We used GloVe.

2. run "train_tgenr.py" to train and evalute the model: python train_tgenr.py NUMBER_OF_WORDS HIDDEN_RNN_LAYERS RNN_TYPE POOLING_PARAMETER RUN_NUM, where RUN_NUM is required to generate different outputs and log files.

For example: 

	python train_tgenr.py 3000 [32,16] 2 2 1 to run tcenr_seq with 2 GRU layers of 32 and 16 cells and pooling size of 2. 

To run with no RNN choose: 

	python train_tgenr.py 3000 [32,16] 0 2 1 where 2 hidden contextual layers will be used with 32 and 16 cells.