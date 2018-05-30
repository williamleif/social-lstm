## Social LSTMs to predict intercommunity-conflict
#### Authors: [William L. Hamilton](https://stanford.edu/~wleif) (wleif@stanford.edu), [Srijan Kumar](https://cs.stanford.edu/) (srijan@cs.stanford.edu)
#### [Project Website](https://snap.stanford.edu/conflict/)
#### [Data (including user/community embeddings)](http://snap.stanford.edu/conflict/conflict_data.zip)

### Overview

This package contains code to replicate the prediction results in the paper [Community Interaction and Conflict on the Web](https://stanford.edu/~wleif/files/conflict-paper-www.pdf) published in the The Web Conference (i.e., WWW) 2018.
The task is trying to predict intercommunity mobilization and conflict on Reddit.com.
In particular, we examine cases where one community (the "source") makes a post that hyperlinks to another community (the "target"), and the goal is predict whether or not this "cross-linking" post will lead to a significant number of source community members "mobilizing" to participate in the target community. 

The primary model is a "socially-primed" LSTM that uses vector embeddings of users and communities to help make this prediction.
In particular, embeddings of users and communities are learned using a "node2vec"-style approach, and we use these embeddings (along with text information from the cross-linking post) to predict whether or not the post will lead to a mobilization.
See the [paper](https://stanford.edu/~wleif/files/conflict-paper-www.pdf) and [project website](https://snap.stanford.edu/conflict) for more details.

If you make use of the code or data associated with this project, please cite the following paper:
  ```
  @inproceedings{kumar2018conflict,
	     author = {Kumar, Srijan and Hamilton, William L. and Jurafsky, Dan and Leskovec, Jure},
	     title = {Community Interaction and Conflict on the Web},
	     booktitle = {The Web Conference (WWW)},
	     year = {2018}
	   }
  ```

### Requirements

The code requires reasonably up-to-date pytorch and sklearn libraries. See requirements.txt for details (or just `pip install requirements.txt`).
Note that you might want to install pytorch from their [website](http://pytorch.org/) to get the latest and greatest version. 

### Using the code

The `social_lstm_model.py` file contains the main model code.
However, before using this code you will need to
  1) Download the necessary data [here](http://snap.stanford.edu/conflict/conflict_data.zip)
  2) Update the "DATA_DIR" value in `constants.py` to point to this unzipped data file.

The command-line arguments for the social LSTM model can be explored with the `help` option.
The default is the "socially-primed" LSTM model described in the paper, with the best-performing hyperparameters as default.
However, you could also set the "lstm_no_social" command-line flag to use a vanilla LSTM etc. 

The code will train and compute validation statistics periodically.
By default it runs for 10 epochs and records the best validation accuracy achieved (i.e., there is no explicit early-stopping but instead checkpoints are used). 

### Notes on replication

The `nonneural_baselines.ipynb` notebook can be used to replicate the exact baseline and ensemble results from the paper.
Note that in the reported results for the LSTM models in the paper, we ran a hyperparameter sweep over learning rates in [0.001, 0.01, 0.1, 0.5], model dimensions [64, 128, 256], dropout parameters [0, 0.2, 0.4], and we considered 2-layer and single-layer LSTMS.
All LSTM models used a batch size of 512.
The results reported in the paper are the test set scores of the best models on the validation sets from this sweep.
In general we found the LSTM results to be reasonably stable across random restarts with a standard deviation of the AUC to around 0.5 (assuming a range of [0,100], as in the paper). 
Finally note that some model variants (e.g., appending the social embeddings) are in the code but not discussed in the paper, as we found these to underperform the presented variant. 

For the Random Forest models we found that increasing the number of trees led to stronger performance and so increased the number of trees to 500 (from the default value of 10). 
We found the other hyperparameters to have minimal impact. 
