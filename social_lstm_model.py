import torch
import random
import argparse
import cPickle as pickle
import torch.nn as nn
import numpy as np

from itertools import ifilter
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

import constants
from embeddings import Embeddings

class SocialLSTM(nn.Module):
    """
    LSTM model for predicting conflict between Reddit communities.
    Can incorporate social embeddings of users and communities/subreddits.
    """

    def _load_glove_embeddings(self):
        print "Loading word embeddings..."
        with open(constants.WORD_EMBEDS) as fp:
            embeddings = np.empty((constants.VOCAB_SIZE, constants.WORD_EMBED_DIM), dtype=np.float32)
            for i, line in enumerate(fp):
                embeddings[i,:] = map(float, line.split()[1:])
        return embeddings

    def _load_user_embeddings(self):
        print "Loading user embeddings..."
        embeds = Embeddings(constants.USER_EMBEDS)
        return embeds._vecs

    def _load_subreddit_embeddings(self):
        print "Loading subreddit embeddings..."
        embeds = Embeddings(constants.SUBREDDIT_EMBEDS)
        return embeds._vecs

    def __init__(self, hidden_dim, batch_size=constants.BATCH_SIZE, prepend_social=True, include_meta=False,
            dropout=None, final_dense=True, include_embeds=False):
        """
        hidden_dim - size of internal LSTM layers
        batch_size - size of minibatches during training
        preprend_social - if True then user/subreddit embeds are prepended.
                          if False then user/subreddit embeds are appended.
                          if None then user/subreddit embeds are not fed to the LSTM.
        include_meta - if True then metadata/linguistic/hand-engineered features are included
        dropout - how much dropout in the LSTM layer connections; if None then single-layer LSTM is used.
        final_dense - whether to include an extra dense Linear+ReLU layer before the softmax (same dimension as LSTM)
        include_embeds - whether to include the user/subreddit layers in the final (i.e, post-lstm) layer(s)
        """
        super(SocialLSTM, self).__init__()
        glove_embeds = self._load_glove_embeddings()
        self.glove_embeds= torch.FloatTensor(glove_embeds)
        self.pad_embed = torch.zeros(1, constants.WORD_EMBED_DIM)
        self.unk_embed = torch.FloatTensor(1,constants.WORD_EMBED_DIM)
        self.unk_embed.normal_(std=1./np.sqrt(constants.WORD_EMBED_DIM))
        self.word_embeds = nn.Parameter(torch.cat([self.glove_embeds, self.pad_embed, self.unk_embed], dim=0), requires_grad=False)
        self.embed_module = torch.nn.Embedding(constants.VOCAB_SIZE+2, constants.WORD_EMBED_DIM)
        self.embed_module.weight = self.word_embeds

        user_embeds = self._load_user_embeddings()
        self.user_embeds = torch.nn.Embedding(constants.NUM_USERS+1, constants.WORD_EMBED_DIM)
        self.user_embeds.weight  = nn.Parameter(torch.cat([torch.FloatTensor(user_embeds),  
            self.pad_embed]), requires_grad=False)

        subreddit_embeds = self._load_subreddit_embeddings()
        self.subreddit_embeds = torch.nn.Embedding(constants.NUM_SUBREDDITS+1, constants.WORD_EMBED_DIM)
        self.subreddit_embeds.weight  = nn.Parameter(torch.cat([torch.FloatTensor(subreddit_embeds), 
            self.pad_embed]), requires_grad=False)

        self.hidden_dim = hidden_dim 
        self.prepend_social = prepend_social

        init_hidden_data = torch.zeros(1 if dropout is None else 2, batch_size, self.hidden_dim)
        #init_hidden_data.normal_(std=1./np.sqrt(self.hidden_dim))
        if constants.CUDA:
            init_hidden_data = init_hidden_data.cuda()
        self.init_hidden = (Variable(init_hidden_data, requires_grad=False),
            Variable(init_hidden_data, requires_grad=False))


        self.rnn = nn.LSTM(input_size=constants.WORD_EMBED_DIM, hidden_size=hidden_dim, 
                num_layers=1 if dropout is None else 2, dropout=0. if dropout is None else dropout)
        
        self.final_dense = final_dense
        self.include_meta = include_meta
        self.include_embeds = include_embeds
        out_layer1_outdim = self.hidden_dim if final_dense else constants.NUM_CLASSES
        if include_meta and include_embeds: 
            self.out_layer1 = nn.Linear(self.hidden_dim+constants.SF_LEN+3*constants.WORD_EMBED_DIM, out_layer1_outdim)
        elif include_embeds:
            self.out_layer1 = nn.Linear(self.hidden_dim+3*constants.WORD_EMBED_DIM, out_layer1_outdim)
        elif include_meta:
            self.out_layer1 = nn.Linear(self.hidden_dim+constants.SF_LEN, out_layer1_outdim)
        else:
            self.out_layer1 = nn.Linear(self.hidden_dim, out_layer1_outdim)
        if self.final_dense:
            self.relu = nn.Tanh()
            self.out_layer2 = nn.Linear(self.hidden_dim, constants.NUM_CLASSES)

    def forward(self, text_inputs, user_inputs, subreddit_inputs, metafeats, lengths):
        text_inputs = self.embed_module(text_inputs)
        user_inputs = self.user_embeds(user_inputs)
        subreddit_inputs = self.subreddit_embeds(subreddit_inputs)
        if self.prepend_social is True:
            inputs = torch.cat([user_inputs, subreddit_inputs, text_inputs], dim=0)
        elif self.prepend_social is False:
            inputs = torch.cat([text_inputs, user_inputs, subreddit_inputs], dim=0)
        else:
            inputs = text_inputs
            lengths = [l-3 for l in lengths]
        inputs  = nn.utils.rnn.pack_padded_sequence(inputs, lengths)
        outputs, h = self.rnn(inputs, self.init_hidden)

        h, lengths = nn.utils.rnn.pad_packed_sequence(outputs)
        h = h.sum(dim=0).squeeze()
        lengths = torch.FloatTensor(lengths)
        if constants.CUDA:
            lengths = lengths.cuda()
        h = h.t().div(Variable(lengths))
        self.h = h
#        self.h = h[0][0].t()
#        h = h[0][0].t()
        
        final_input = h.t()
        if self.include_meta:
            final_input = torch.cat([final_input, metafeats.t()], dim=1)
        if self.include_embeds:
            final_input = torch.cat([final_input, user_inputs.squeeze(), subreddit_inputs[0], subreddit_inputs[1]], dim=1)
        if not self.final_dense:
            weights = self.out_layer1(final_input)
        else:
            weights = self.out_layer2(self.relu(self.out_layer1(final_input)))
        return weights

def load_data(batch_size, max_len):
    print "Loading train/test data..."
    thread_to_sub = {}
    with open(constants.POST_INFO) as fp:
        for line in fp:
            info = line.split()
            source_sub = info[0]
            target_sub = info[1]
            source_post = info[2].split("T")[0].strip()
            target_post = info[6].split("T")[0].strip()
            thread_to_sub[source_post] = source_sub
            thread_to_sub[target_post] = target_sub

    label_map = {}
    source_to_dest_sub = {}
    with open(constants.LABEL_INFO) as fp:
        for line in fp:
            info = line.split("\t")
            source = info[0].split(",")[0].split("\'")[1]
            dest = info[0].split(",")[1].split("\'")[1]
            label_map[source] = 1 if info[1].strip() == "burst" else 0
            try:
                source_to_dest_sub[source] = thread_to_sub[dest]
            except KeyError:
                continue

    with open(constants.SUBREDDIT_IDS) as fp:
        sub_id_map = {sub:i for i, sub in enumerate(fp.readline().split())}

    with open(constants.USER_IDS) as fp:
        user_id_map = {user:i for i, user in enumerate(fp.readline().split())}

    with open(constants.PREPROCESSED_DATA) as fp:
        words, users, subreddits, lengths, labels, ids = [], [], [], [], [], []
        for i, line in enumerate(fp):
            info = line.split("\t")
            if info[1] in label_map and info[1] in source_to_dest_sub:
                title_words = info[-2].split(":")[1].strip().split(",")
                title_words = title_words[:min(len(title_words), constants.MAX_LEN)]
                if len(title_words) == 0 or title_words[0] == '':
                    continue
                words.append(map(int, title_words))

                body_words = info[-1].split(":")[1].strip().split(",")
                body_words = body_words[:min(len(body_words), constants.MAX_LEN-len(title_words))]
                if not (len(body_words) == 0 or body_words[0] == ''):
                    words[-1].extend(map(int, body_words))

                words[-1] = [constants.VOCAB_SIZE+1 if w==-1 else w for w in words[-1]]

                if not info[0] in sub_id_map:
                    source_sub = constants.NUM_SUBREDDITS
                else:
                    source_sub = sub_id_map[info[0]]
                dest_sub = source_to_dest_sub[info[1]]
                if not dest_sub in sub_id_map:
                    dest_sub = constants.NUM_SUBREDDITS
                else:
                    dest_sub = sub_id_map[dest_sub]
                subreddits.append([source_sub, dest_sub])

                users.append([constants.NUM_USERS if not info[3] in user_id_map else user_id_map[info[3]]])
                ids.append(info[1])

                lengths.append(len(words[-1])+3)
                labels.append(label_map[info[1]])

        batches = []
        np.random.seed(0)
        for count, i in enumerate(np.random.permutation(len(words))):
            if count % batch_size == 0:
                batch_words = np.ones((max_len, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
                batch_users = np.ones((1, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
                batch_subs = np.ones((2, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
                batch_lengths = []
                batch_labels = []
                batch_ids = []
            length = min(max_len, len(words[i]))
            batch_words[:length, count % batch_size] = words[i][:length]
            batch_users[:, count % batch_size] = users[i]
            batch_subs[:, count % batch_size] = subreddits[i]
            batch_lengths.append(length)
            batch_labels.append(labels[i])
            batch_ids.append(ids[i])
            if count % batch_size == batch_size - 1:
                order = np.flip(np.argsort(batch_lengths), axis=0)
                batches.append((list(np.array(batch_ids)[order]),
                    torch.LongTensor(batch_words[:,order]), 
                    torch.LongTensor(batch_users[:,order]), 
                    torch.LongTensor(batch_subs[:,order]), 
                    list(np.array(batch_lengths)[order]),
                    torch.FloatTensor(np.array(batch_labels)[order])))
    return batches

def get_embeddings(data):

    embeds = []
    ids = []
    for batch in data:
        id, text, users, subs, lengths, metafeats, labels = batch
        text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
        model(text, users, subs, metafeats, lengths)
        batch_embeds = model.h
        embeds.append(batch_embeds.t().data.cpu().numpy())
        ids.extend(id)
    return ids, np.concatenate(embeds)

def train(model, train_data, val_data, test_data, optimizer, 
        epochs=10, log_every=100, log_file=None, save_embeds=False):
    if not log_file is None:
        lg_str = log_file
        log_file = open(log_file, "w")

    ema_loss = None
    criterion = nn.BCEWithLogitsLoss()
    best_iter = (0., 0,0)
    best_test = 0.
    embeds = None
    for epoch in range(epochs):
        random.shuffle(train_data)
        for i, batch in enumerate(train_data):
            _, text, users, subs, lengths, metafeats, labels = batch
            text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
            optimizer.zero_grad()
            outputs = model(text, users, subs, metafeats, lengths)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            if ema_loss is None:
                ema_loss = loss.data[0]
            else:
                ema_loss = 0.01*loss.data[0] + 0.99*ema_loss

            if i % 10 == 0:
                print epoch, i, ema_loss
                print >>log_file, epoch, i, ema_loss
            if  i % log_every == 0:
                auc = evaluate_auc(model, val_data)
                print "Val AUC", epoch, i, auc
                if not log_file is None:
                    print >>log_file, "Val AUC", epoch, i, auc
                if auc > best_iter[0]:
                    best_iter = (auc, epoch, i)
                    print "New best val!", best_iter
                    best_test = evaluate_auc(model, test_data)
                    if auc > 0.7:
                        ids, embeds = get_embeddings(train_data+val_data+test_data)
    print "Overall best val:", best_iter
    if not log_file is None:
        print >>log_file, "Overall best test:", best_test
        print >>log_file, "Overall best val:", best_iter
        if not embeds is None and save_embeds:
            np.save(open(lg_str+"-embeds.npy", "w"), embeds)
            pickle.dump(ids, open(lg_str+"-ids.pkl", "w"))
    return best_iter[0]

def evaluate_auc(model, test_data):

    predictions = []
    gold_labels = []
    for batch in test_data:
        _, text, users, subs, lengths, metafeats, labels = batch
        if constants.CUDA:
            gold_labels.extend(labels.cpu().numpy().tolist())
        else:
            gold_labels.extend(labels.numpy().tolist())
        text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
        outputs = model(text, users, subs, metafeats, lengths)
        if constants.CUDA:
            predictions.extend(outputs.data.squeeze().cpu().numpy().tolist())
        else:
            predictions.extend(outputs.data.squeeze().numpy().tolist())

    auc = roc_auc_score(gold_labels, predictions)
    return auc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--log_file", type=str, default=None, 
            help="Where to log the model training details.")
    parser.add_argument("--save_embeds", action='store_true',
            help="Whether to save the hidden-state LSTM embeddings that are generated.\
                  They will be stored based on the log_file name used above.")
    parser.add_argument("--dropout", type=float, default=0.2,
            help="Dropout rate for inter-LSTM layers in 2-layer LSTM.")
    parser.add_argument("--single_layer", action='store_true',
            help="Use single-layer LSTM (implies that dropout param is ignored)")
    parser.add_argument("--include_meta", action='store_true',
            help="Include metadata/hand-crafted features in final layer of model.")
    parser.add_argument("--final_dense", action='store_true',
            help="Include an extra Linear+ReLU layer before the softmax.")
    parser.add_argument("--lstm_append_social", action='store_true', 
            help="Append the social embeddings instead of prepending them to LSTM input.")
    parser.add_argument("--lstm_no_social", action='store_true', 
            help="Do not include social embeddings in LSTM input.")
    parser.add_argument("--final_layer_social", action='store_true', 
            help="(Also) include social embeddings in the final layer.")
    args = parser.parse_args()
    dropout = None if args.single_layer else args.dropout
    if args.lstm_append_social and args.lstm_no_social:
        raise Exception("Only one of --lstm_append_social and --lstm_no_social can be True at a time.")
    if args.log_file is None and args.save_embeds:
        raise Exception("A log file must be specified if you want to store the LSTM embeddings of the posts.")
    if args.lstm_append_social or args.lstm_no_social:
        prepend_social = None if args.lstm_no_social else False
    else:
        prepend_social = True


    print "Loading training data"
    # WE HAVE PRE-CONSTRUCTED TRAIN/VAL/TEST DATA USING load_data
    # this avoids re-doing all the pre-processing everytime the code is
    # run. This data is fixed to a batch size of 512.
    train_data = pickle.load(open(constants.TRAIN_DATA))
    val_data = pickle.load(open(constants.VAL_DATA))
    test_data = pickle.load(open(constants.TEST_DATA))

    print len(train_data)*constants.BATCH_SIZE, "training examples", len(val_data)*512, "validation examples"
    print sum([i for batch in train_data for i in batch[-1]]), "positive training", sum([i for batch in val_data for i in batch[-1]]), "positive validation"

    # annoying checks for CUDA switches....
    if constants.CUDA:
        for i in range(len(train_data)):
            batch = train_data[i]
            metafeats = batch[5]
            train_data[i] = (batch[0], 
                    batch[1].cuda(),
                    batch[2].cuda(),
                    batch[3].cuda(),
                    batch[4],
                    metafeats.cuda(),
                    batch[6].cuda())

        for i in range(len(val_data)):
            batch = val_data[i]
            metafeats = batch[5]
            val_data[i] = (batch[0], 
                    batch[1].cuda(),
                    batch[2].cuda(),
                    batch[3].cuda(),
                    batch[4],
                    metafeats.cuda(),
                    batch[6].cuda())

        for i in range(len(test_data)):
            batch = test_data[i]
            metafeats = batch[5]
            test_data[i] = (batch[0], 
                    batch[1].cuda(),
                    batch[2].cuda(),
                    batch[3].cuda(),
                    batch[4],
                    metafeats.cuda(),
                    batch[6].cuda())

    best_auc = (0,"") 
    model = SocialLSTM(args.hidden_dim, prepend_social=prepend_social, dropout=args.dropout, include_embeds=args.final_layer_social, 
            include_meta=args.include_meta, final_dense=args.final_dense)
    if constants.CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(ifilter(lambda p : p.requires_grad, model.parameters()), lr=args.learning_rate)
    auc = train(model, train_data, val_data, test_data, optimizer, epochs=10, log_file=args.log_file, save_embeds=args.save_embeds)
