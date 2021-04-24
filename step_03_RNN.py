import torch
import torch.nn as nn # the neural network package that contains functions for creating the neural network layers
import torch.nn.functional as F
import torch.optim as optim # a package that allows use to use an optimizer in order to update the parameters during training
from torch.utils.data import DataLoader # allows use to process the data in batches
from torch.nn.utils.rnn import pad_sequence # a function that zero-pads the sentences so they can have equal size in a batch
import pandas as pd
import pickle
torch.manual_seed(0) # set a random seed for reproducibility
from tqdm import tqdm

batch_size = 1
is_preprocessing = False #This toggle preprocessing for LSTM model (not preprocessing for text)
is_RNN = True

device = torch.device('cuda:0')
#device = torch.device('cpu')
torch.cuda.empty_cache()

# Load data
# Load result:
with open("data_sample_2x.txt", "rb") as fp:   # Unpickling
    sentences = pickle.load(fp)

print(sentences[22:35])

def add_sentence_boundaries(data):
    """
    Takes the data, where each line is a sentence, appends <s> token at the beginning and </s> at the end of each sentence
    Example input: I live in Helsinki
    Example output: <s> I live in Helsinki </s>
    
    Arguments
    ---------
    data : list
            a list of sentences
    
    Returns
    -------
    res : list
            a list of sentences, where each sentence has <s> at the beginning and </s> at the end
    """
    res = []
    for sent in data:
        sent = '<s> ' + sent.rstrip() + ' </s>'
        res.append(sent)
    
    return res

def create_indices(data):
    """
    This function creates two dictionaries: word2idx and idx2word, containing each unique word in the dataset
    and its corresponding index.
    Remember that the starting index should be 1 and not 0
    
    Arguments
    ---------
    data - list
            a list of sentences, where each sentence starts with <s>
            and ends with </s> token
    
    Returns
    -------
    word2idx - dictionary
                a dictionary, where the keys are the words and the values are the indices
                
    idx2word - dictionary
                a dictionary, where the keys are the indices and the values are the words
    """
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    word2idx = dict()
    idx2word = dict()
    
    data_list = ''
    for sentence in data:
        data_list = data_list + ' ' + sentence

    data_list = data_list[1:]
    data_split = data_list.split(' ')
    data_unique = []
    for word in data_split:
        if word not in data_unique:
            data_unique.append(word)
            if word not in word2idx.keys():
                 word2idx[word] = data_unique.index(word)+1
    
    for key, value in word2idx.items():
        idx2word[value] = key
    
    return word2idx, idx2word

def index_data(data, word2idx):
    """
    This function replaces each word in the data with its corresponding index
    
    Arguments
    ---------
    data - list
            a list of sentences, where each sentence starts with <s>
            and ends with </s> token
    
    word2idx - dict
            a dictionary where the keys are the unique words in the data
            and the values are the unique indices corresponding to the words%
    
    Returns
    -------
    data_indexed - list
                a list of sentences, where each word in the sentence is replaced with its index
    """
    
    data_indexed = []
    # YOUR CODE HERE
    #raise NotImplementedError()
    
    for sentence in data:
        sentence_index = []
        for word in sentence.split(' '):
            sentence_index.append(word2idx[word])
        data_indexed.append(sentence_index)
    

    return data_indexed

def convert_to_tensor(data_indexed):
    """
    This function converts the indexed sentences to LongTensors
    
    Arguments
    ---------
    data_indexed - list
            a list of sentences, where each word in the sentence
            is replaced by its index
    
    Returns
    -------
    tensor_array - list
                a list of sentences, where each sentence
                is a LongTensor
    """
    
    tensor_array = []
    for sent in data_indexed:
        tensor_array.append(torch.LongTensor(sent))    
        
    return tensor_array

def combine_data(input_data, labels_data):
    """
    This function converts the input features and the labels into tuples
    where each tuple corresponds to one sentence in the format (features, labels)
    
    Arguments
    ---------
    input_data - list
            a list of tensors containing the training features
    
    labels_data - list
            a list of tensors containing the training labels
    
    Returns
    -------
    res - list
            a list of tuples, where each tuple corresponds to one sentece pair
            in the format (features, labels)
    """
    
    res = []
    
    for i in range(len(input_data)):
        res.append((input_data[i], labels_data[i]))

    return res

def remove_extra(data, batch_size):
    """
    This function removes the extra data that does not fit in a batch   
    
    Arguments
    ---------
    data - list
            a list of tuples, where each tuple corresponds to a
            sentence in a format (features, labels)
            
    batch_size - integer
                    the size of the batch
    
    
    Returns
    -------
    data - list
            a list of tuples, where each tuple corresponds to a
            sentence in a format (features, labels)
    """
    
    extra = len(data) % batch_size
    if extra != 0:
        data = data[:-extra][:]

    return data


def collate(list_of_samples):
    """
    This function zero-pads the training data in order to process the sentences
    in a batch during training
    
    Arguments
    ---------
    list_of_samples - list
                        a list of tuples, where each tuple corresponds to a
                        sentence in a format (features, labels)
    
    
    Returns
    -------
    pad_input_data - tensor
                        a tensor of input features equal to the batch size,
                        where features are zero-padded to have equal lengths
                        
    input_data_lengths - list
                        a list where each element is the length of the 
                        corresponding sentence
    
    pad_labels_data - tensor
                        a tensor of labels equal to the batch size,
                        where labels are zero-padded to have equal lengths
            
    """
    
    
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    input_data, labels_data = zip(*list_of_samples)

    input_data_lengths = [len(seq) for seq in input_data]
    
    padding_value = 0

    # pad input
    pad_input_data = pad_sequence(input_data, padding_value=padding_value)
    
    # pad labels
    pad_labels_data = pad_sequence(labels_data, padding_value=padding_value)

    return pad_input_data, input_data_lengths, pad_labels_data

def prepare_for_training(data_indexed):
    """
    This function creates the input features and their corresponding labels
    
    Arguments
    ---------
    data_indexed - list
            a list of sentences, where each word in the sentence
            is replaced by its index
    
    
    Returns
    -------
    input_data - list
            a list of indexed sentences, where the last element of each sentence is removed
            
    labels_data - list
            a list of indexed sentences, where the first element of each sentence is removed
    """
    
    input_data = []
    labels_data = []

     # YOUR CODE HERE
    #raise NotImplementedError()
    for data in data_indexed:    
        input_data.append(data[:-1])
        labels_data.append(data[1:])
    
    return input_data, labels_data

def preprocess_data(data):
    """
    This function runs the whole preprocessing pipeline and returns the prepared
    input features and labels, along with the word2idx and idx2word dictionaries
    
    Arguments
    ---------
    data - list
            a list of sentences that need to be prepared for training
    
    
    Returns
    -------
    input_data - list
            a list of tensors, where each tensor is an indexed sentence used as input feature
            
    labels_data - list
            a list of tensors, where each tensor is an indexed sentence used as a true label
    
    word2idx - dictionary
                a dictionary, where the keys are the words and the values are the indices
                
    idx2word - dictionary
                a dictionary, where the keys are the indices and the values are the words
    """
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    #1. Add sentence boundaries    
    res = add_sentence_boundaries(data)
    
    #2. Create index dictionaries (word2idx and idx2word)
    word2idx, idx2word = create_indices(res)    
    
    #3. Index the data in a way that each word is replaced by its index
    indexed_data = index_data(res, word2idx)
    
    #4. Convert the indexed data to a list of tensors, where each tensor is a sentence
    tensor_array = convert_to_tensor(indexed_data)    
    
    #5. Split each sentence to input and labels
    input_data, labels_data = prepare_for_training(tensor_array)
    
    return input_data, labels_data, word2idx, idx2word

class DNN(nn.Module):
    def __init__(self, word2idx, embed_dim, context_dim, num_layers):
        """
        This function initializes the layers of the model
        
        Arguments
        ---------
        word2idx - dictionary
                    a dictionary where the keys are the unique words in the data
                    and the values are the unique indices corresponding to the words
        
        embed_dim - integer
                        the size of the word embeddings

        context_dim - integer
                        the dimension of the hidden size
                        
        num_layers - integer
                        the number of layers in the GRU cell
        """
        super(DNN, self).__init__()
        self.word2idx = word2idx
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        
        # here we initialise weighs of a model
        self.word_embed = nn.Embedding(len(self.word2idx)+1, self.embed_dim) # embedding layer
        if (is_RNN):
            self.rnn = nn.GRU(self.embed_dim, self.context_dim, num_layers=self.num_layers) # GRU cell        
        else:
            self.rnn = nn.LSTM(input_size = self.embed_dim, 
                                hidden_size = self.context_dim,
                                num_layers = self.num_layers,
                                dropout=0.1)
            
        self.dropout = nn.Dropout(0.1) # Dropout        
        self.out = nn.Linear(self.context_dim, len(self.word2idx)+1) # output layer

    
    def forward(self, word, hidden):
        """
        This function implements the forward pass of the model
        
        Arguments
        ---------
        word - tensor
                a tensor containing indices of the words in a batch
                
        hidden - tensor
                    the previous hidden state of the GRU model
        
        Returns
        -------
        output - tensor
                    a tensor of logits from the linear transformation
        
        hidden - tensor
                    the current hidden state of the GRU model
        """ 
        
        # YOUR CODE HERE
        #raise NotImplementedError()
        #1. Replace the indexed word with its embedding vector. 
        #In other words, pass it through the embedding layer
        embeds = self.word_embed(word)
        
        batch_size = word.shape[0]
        #print(batch_size)
        #2. Reshape the embedding vector to a shape of (1, batch_size)
        embeds = embeds.reshape(1, batch_size, self.embed_dim)
        
        #3. Pass the embedding through the GRU cell to get the output 
        #and the hidden tensors. The GRU function takes as input the 
        #work embedding and the previous hidden state.
        output, hidden = self.rnn(embeds, hidden)
        
        #4. Addpy a dropout to the output of the GRU
        if is_RNN:
            output = self.dropout(output)            
        
        #5. Apply the linear transformation to the output of the dropout layer.
        output = self.out(output)
        
        #6. Reshape the output to have a shape (batch_size, vocab_length+1)
        output = output.reshape(batch_size, len(self.word2idx) + 1)
        
        #7. Return the output of the linear transformation and the hidden tensor
        hidden = hidden.to(device)
        return output, hidden
    
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.context_dim).to(device),
                torch.zeros(self.num_layers, sequence_length, self.context_dim).to(device))
    
def predict(model, hidden_size, num_layers, word2idx, idx2word, context, max_len):
    """
    This function predicts the next word, based on the history of the previous words.
    We start with the 'context' and then feed the prediction as the next input.
    
    Arguments
    ---------
    model - object
                an nn object that contains the trained model
                
    hidden_size - integer
                    the size of the hidden layer (the context size)
                    
    num_layers - integer
                    the number of layers in the GRU cell
                
    word2idx - dictionary
                    a dictionary where the keys are the unique words in the data
                    and the values are the unique indices corresponding to the words
                    
    idx2word - dictionary
                a dictionary, where the keys are the indices and the values are the words
                    
    context - string
                the context sentence
    
    max_len - integer
                integer value representing up to how many words to generate
                            
    Returns
    -------
    
    predictions - string
                    a string containing the generated sentence
    """
    
    # index the context
    context_indexed = []
    for word in context.split():
        word_indexed = torch.LongTensor(1)
        word_indexed[:] = word2idx[word]
        context_indexed.append(word_indexed)
    
    with torch.no_grad():
        predictions = []
        # first build the hidden state from the context
        
        if is_RNN:
            hidden = torch.zeros((num_layers, 1, hidden_size), device=device)
        else:
            hidden = model.init_state(1)            
        
        for word in context_indexed:
            predictions.append(idx2word[word.item()])
            word = word.to(device)
            output, hidden = model(word, hidden)
            
        next_input = context_indexed[-1]
        while((len(predictions) < max_len) and (predictions[-1] != '</s>')):
            
            # YOUR CODE HERE
            #raise NotImplementedError()
            #1. Run the forward pass to get the output. Don't forget to include the `hidden` state
            #print("INPUT", next_input)
            next_input = next_input.to(device)
            out, hidden = model.forward(next_input, hidden)
            
            #2. Run the output through a softmax to convert it to a probability distribution (`F.softmax`)
            out = F.softmax(out)
            
            #3. Get the word with the highest probability using the `topk(1)` function
            value, index = out.topk(1)
            
            #4. Convert the index of the predicted word to the actual word using the idx2word dictionary
            word = idx2word[index.item()]
            
            #5. Append the predicted word to the `predictions` array
            predictions.append(word)
            next_input = index
            
    predictions = ' '.join(predictions)
    
    return predictions


def train_model(pairs_batch_train, model, hidden_size, num_layers, loss_function, optimizer, n_epochs):
    """
    This function implements the training of the model

    Arguments
    ---------
    pairs_batch_train - object
                            a DataLoader object that contains the batched data

    model - object
                an object that contains the initialized model
                
    hidden_size - integer
                    the size of the hidden layer (the context size)
    
    num_layers - integer
                        the number of layers in the LSTM cell

    loss_function - object
                        the CrossEntropy loss function

    optimizer - object
                        an Adam object of the optimizer class

    n_epochs - integer
                the number of epochs to train
    """ 
    loss_list = []

    for epoch in tqdm(range(n_epochs)): # iterate over the epochs
        epoch_loss = 0
        model.train() # put the model in training mode
        
        for iteration, batch in enumerate(pairs_batch_train): # at each step take a batch of sentences
            sent_loss = 0
            optimizer.zero_grad() # clear gradients
            
            train_input, train_input_lengths, train_labels = batch # extract the data from the batch
            train_input = train_input.to(device)
            train_labels = train_labels.to(device)
            
            if is_RNN:
                hidden = torch.zeros((num_layers, train_input.size(1), hidden_size)) # initialize the hidden state
                hidden = hidden.to(device)
            else:
                hidden = model.init_state(train_input.size(1))            
            
            for i in range(train_input.size(0)): # iterate over the word in the sentence
                output, hidden = model(train_input[i], hidden) # forward pass               
                    
                labels = torch.LongTensor(train_labels.size(1)) # define a random tensor with batch_size as number of elements
                labels = labels.to(device)
                labels[:] = train_labels[i][:] # put the correct label values in the tensor
                
                sent_loss += loss_function(output, labels) # compute the loss, compare the predictions and the labels

            sent_loss.backward() # compute the backward pass
            optimizer.step() # update the parameters

            epoch_loss += sent_loss

        loss_list.append( epoch_loss / len(pairs_batch_train))
        print('Epoch: {}   Loss: {}'.format(epoch+1, epoch_loss / len(pairs_batch_train))) # print the loss at each epoch
        # Save model every 5 epoch
        
        if(epoch+1)%5 == 0:
            filename = "models/LSTM_00005-" + str(epoch+1) + '.pt'
            torch.save(model.state_dict(), filename)
            list_df = {'loss':loss_list}
            df = pd.DataFrame(list_df)
            df.to_csv('models/LSTM_00005_loss.csv')
                       
    return loss_list    


    

if is_preprocessing:
    train_input, train_labels, word2idx, idx2word = preprocess_data(sentences) # run the preprocessing pipeline
    train_data = combine_data(train_input, train_labels)
    train_data = remove_extra(train_data, batch_size)
    torch.save(train_input, "RNN_data/train_input.pt")
    torch.save(train_labels, "RNN_data/train_labels.pt")
    torch.save(word2idx, "RNN_data/word2idx.pt")
    torch.save(idx2word, "RNN_data/idx2word.pt")
    torch.save(train_data, "RNN_data/train_data.pt")
else:
    train_input = torch.load('RNN_data/train_input.pt')
    train_labels = torch.load('RNN_data/train_labels.pt')
    word2idx = torch.load('RNN_data/word2idx.pt')
    idx2word = torch.load('RNN_data/idx2word.pt')
    train_data = combine_data(train_input, train_labels)
    train_data = remove_extra(train_data, batch_size)
    
pairs_batch_train = DataLoader(dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate,
                    pin_memory=True)

n_epochs = 100 # the number of epochs to train
embed_dim = 300 # the size of the embedding
hidden_size = 450 # the size of the hidden state
num_layers = 1 # the number of layers in the GRU cell
model = DNN(word2idx, embed_dim, hidden_size, num_layers).to(device) # initialize the model
loss_function = nn.CrossEntropyLoss(ignore_index=0) # define the loss function
optimizer = optim.Adam(model.parameters(), lr=0.0005) # define the optimizer


loss_list = train_model(pairs_batch_train, model,hidden_size, num_layers, loss_function, optimizer, n_epochs)

contexts = ['<s> to the moon', '<s> GME', '<s> my wife', '<s> Elon Musk', '<s> the best stock is', '<s> I think the market will']
max_len = 50

for context in contexts:
    predictions = predict(model, hidden_size, num_layers, word2idx, idx2word, context, max_len)
    print(predictions)
    print('\n')