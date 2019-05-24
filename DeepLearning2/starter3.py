import sys
import numpy
import tensorflow as tf
import numpy as np
import random
import os

#USC_EMAIL = 'hveerava@usc.edu'  # TODO(student): Fill to compete on rankings.
USC_EMAIL = 'hveerava@usc.edu'
PASSWORD = 'ebb6580b22c75d1b'  # TODO(student): You will be given a password via email.
TRAIN_TIME_MINUTES = 6

class DatasetReader(object):

    # TODO(student): You must implement this.
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        """Reads file into dataset, while populating term_index and tag_index.

        Args:
            filename: Path of text file containing sentences and tags. Each line is a
                sentence and each term is followed by "/tag". Note: some terms might
                have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
                separates the tag.
            term_index: dictionary to be populated with every unique term (i.e. before
                the last "/") to point to an integer. All integers must be utilized from
                0 to number of unique terms - 1, without any gaps nor repetitions.
            tag_index: same as term_index, but for tags.

        the _index dictionaries are guaranteed to have no gaps when the method is
        called i.e. all integers in [0, len(*_index)-1] will be used as values.
        You must preserve the no-gaps property!

        Return:
            The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
            each parsedLine is a list: [(termId1, tagId1), (termId2, tagId2), ...]
        """
        # term_index = {}
        # tag_index = {}
        unique_term = 0
        unique_tag = 0
        initial_termIndex = len(term_index)
        if len(term_index) == 0:
            unique_term = 0
        else:
            unique_term = len(term_index)
        matrixbag = []

        filetoread = open(filename, mode='r', encoding='utf-8')
        data = filetoread.read()
        sentences = data.splitlines()
        for sentence in sentences:
            # listforbag = []
            for word in sentence.split(" "):
                extractdata = word.rsplit('/', 1)
                word = extractdata[0]
                tag = extractdata[1]
                if word not in term_index:
                    term_index[word] = unique_term
                    unique_term = unique_term + 1
                if tag not in tag_index:
                    tag_index[tag] = unique_tag
                    unique_tag = unique_tag + 1
        # print(term_index)
        for sentence in sentences:
            listforbag = []
            for word in sentence.split(" "):
                extractdata = word.rsplit('/', 1)
                word = extractdata[0]
                tag = extractdata[1]
                if word in term_index:
                    if tag in tag_index:
                        tup = (term_index[word], tag_index[tag])
                        listforbag.append(tup)
                        # listforbag.append(tuple(term_index[word],tag_index[tag]))
                    # listforbag.append(tag_index[tag])
            # print(listforbag)
            matrixbag.append(listforbag)
            # pass
        #print(matrixbag)
        return matrixbag

    # TODO(student): You must implement this.
    @staticmethod
    def BuildMatrices(dataset):
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

        Args:
            dataset: Returned by method ReadFile. It is a list (length N) of lists:
                [sentence1, sentence2, ...], where every sentence is a list:
                [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

        Returns:
            Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
                terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
                    indices in dataset[i].
                tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
                    indices in dataset[i].
                lengths: shape (N) int64 numpy array. Entry i contains the length of
                    sentence in dataset[i].

            T is the maximum length. For example, calling as:
                BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
            i.e. with two sentences, first with length 2 and second with length 4,
            should return the tuple:
            (
                [[1, 4, 0, 0],    # Note: 0 padding.
                 [13, 3, 7, 3]],

                [[2, 10, 0, 0],   # Note: 0 padding.
                 [20, 6, 8, 20]],

                [2, 4]
            )
        """
        # pass
        # building matrices
        # building term_matrix
        term_list = []
        tag_list = []
        maxlen = -999
        for list1 in dataset:
            if len(list1) > maxlen:
                maxlen = len(list1)
        nrows = len(dataset)
        ncols = maxlen
        term_index = []
        # term_index1 = np.zero([nrows,ncols],dtype=int)
        tag_index = []
        length_array = []
        # print(len(matrixbag))
        for list1 in dataset:
            # print(list1)
            length_array.append(len(list1))
            templist = [0] * ncols
            temptaglist = [0] * ncols
            counter = 0
            for x, y in list1:
                templist[counter] = x
                temptaglist[counter] = y
                counter = counter + 1
            # print(templist)
            term_index.append(templist)
            tag_index.append(temptaglist)

        term_index1 = np.array(term_index)
        tag_index1 = np.array(tag_index)
        length_array1 = np.array(length_array)

        built_matrix = (term_index1, tag_index1, length_array1)
        return built_matrix

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.

        NOTE: Please do not change this method. The grader will use an identitical
        copy of this method (if you change this, your offline testing will no longer
        match the grader).

        Args:
            train_filename: .txt path containing training data, one line per sentence.
                The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.

        Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries,
            respectively, from term to integer ID and from tag to integer ID. The int
            IDs are used in the numpy matrices.
            The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
                - train_terms: numpy int matrix.
                - train_tags: numpy int matrix.
                - train_lengths: numpy int vector.
                These 3 are identical to what is returned by BuildMatrices().
            The 4th element is a tuple of 3 elements as above, but the data is
            extracted from test_filename.
        """
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        # term_index = {}
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                    (train_terms, train_tags, train_lengths),
                    (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

    def __init__(self, max_length, num_terms, num_tags):
        """Constructor. You can add code but do not remove any code.

        The arguments are arbitrary: when you are training on your own, PLEASE set
        them to the correct values (e.g. from main()).

        Args:
            max_lengths: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).

        You will be passed these arguments by the grader script.
        """

        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
       
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X') #placeholder with max_length of sentence in the matrix
       # input tags for the text
        self.y = tf.placeholder(tf.int64, [None, self.max_length], 'Y')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')
        self.counter = 1
        self.global_step = tf.Variable(0, trainable=False)
        #init_embeds = tf.random_uniform([num_terms, state_size])
        #self.embeddings = tf.Variable(init_embeds)
        self.session = tf.Session()
        
        #self.session.run(tf.global_variables_initializer())

    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        Specifically, the return matrix B will have the following:
            B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        # returns a binary matrix sequence_mask is used to achieve this
        binary_matrix = tf.sequence_mask(length_vector, self.max_length, dtype=tf.float32)
        #return tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32)
        return binary_matrix

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""

        pass

    # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        pass

    # TODO(student): You must implement this.
    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).

        Please do not change or override self.x nor self.lengths in this function.

        Hint:
            - Use lengths_vector_to_binary_matrix
            - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """
        #state_size = 50#number of neurons
        state_size = 50
        #max_length = 10
        drop_prob = 0.2

        self.embeddings = tf.get_variable('embedding_matrix', [self.num_terms, 50])
        xemb = tf.nn.embedding_lookup(self.embeddings, self.x)
        #xemb = tf.nn.dropout(xemb, drop_prob)
        #self.regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        

        #rnn_cell = tf.keras.layers.SimpleRNNCell(state_size,recurrent_regularizer = tf.keras.regularizers.l2(l=0.01), kernel_regularizer =tf.keras.regularizers.l2(l=0.01), bias_regularizer = tf.keras.regularizers.l2(l=0.01))
        rnn_cell = tf.keras.layers.SimpleRNNCell(state_size)
        #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob = 1- drop_prob)
        states = []
        cur_state = tf.zeros(shape=[1, state_size])
        xrange = range
        for i in xrange(self.max_length):
            cur_state = rnn_cell(xemb[:, i, :], [cur_state])[0]
            states.append(cur_state)
        stacked_states = tf.stack(states, axis=1)
        #print(stacked_states)
        #self.logits = tf.contrib.layers.fully_connected(stacked_states, self.num_tags)
        #self.logits = tf.contrib.layers.dropout(self.logits, keep_prob = 0.5)
        self.logits = tf.layers.dense(stacked_states, self.num_tags)
        #self.logits = tf.contrib.layers.batch_norm(self.logits, is_training=True, updates_collections=None)
       
        #finaloutput = tf.contrib.layers.fully_connected(rnn_outputs,self.num_tags)
        #finaloutput = tf.contrib.layers.fully_connected(stacked_states, self.num_tags)
        self.weights_matrix = self.lengths_vector_to_binary_matrix(self.lengths)
        
        # logits is a 3D matrix
        #print("calculating the loss")
        
        #self.__loss = tf.reduce_mean(cross_entropy_loss)

      
 


    def run_inference(self, terms, lengths):
        #called while testing the data to get the test tag matrix for the given test_data matrix
        """Evaluates self.logits given self.x and self.lengths.
        # get the test terms and test lengths to get the predicted tag set
        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """
        #logits = run(self.logits, {self.x: terms, self.lengths: lengths})
        #return numpy.argmax(logits, axis=2)
        #return numpy.zeros_like(terms)
        logits = self.session.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)

    # TODO(student): You must implement this.
    def build_training(self):
        """Prepares the class for training.

        It is up to you how you implement this function, as long as train_on_batch
        works.

        Hint:
            - Lookup tf.contrib.seq2seq.sequence_loss
            - tf.losses.get_total_loss() should return a valid tensor (without raising
                an exception). Equivalently, tf.losses.get_losses() should return a
                non-empty list.
        """
        cross_entropy_loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.y, self.weights_matrix)
        self.loss = cross_entropy_loss
        #print(cross_entropy_loss)
        #cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
        #self.learn_Rate = tf.placeholder_with_default(numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
        self.learning_rate = 0.01
        
        self.learn_rate = tf.placeholder(tf.float32, shape=(), name="learn_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        optimizer.minimize(cross_entropy_loss,global_step=self.global_step)
        # adding loss
        tf.losses.add_loss(self.loss, loss_collection = tf.GraphKeys.LOSSES)
        self.total_loss = tf.losses.get_total_loss(name= "total_loss_Val")
        #tf.losses.sigmoid_cross_entropy(multi_class_labels=tags, logits=self.logits)
        
        #opt = tf.train.AdamOptimizer(learning_rate=learning_Rate)
        self.train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), optimizer, self.global_step)
        self.session.run(tf.global_variables_initializer())
        #print("entering build training")
        #print(self.total_loss)

    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=0.0001):
        """Performs updates on the model given training training data.

        This will be called with numpy arrays similar to the ones created in
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size. Grader script will not pass this,
                but it is only here so that you can experiment with a "good batch size"
                from your main block.
            learn_rate: float for learning rate. Grader script will not pass this,
                but it is only here so that you can experiment with a "good learn rate"
                from your main block.

        Return:
            boolean. You should return True iff you want the training to continue. If
            you return False (or do not return anyhting) then training will stop after
            the first iteration!
        """
        #if self.counter == 5:
        #    return False
        
        avg_cost = 0
        #decay = 0.9
        #initial_learn_rate = 0.01
        #learning_rate = (1/(self.counter*decay)) * initial_learn_rate
        #increment_global_step = tf.assign(self.global_step, self.global_step + 1)
        learning_rate = self.session.run(tf.train.exponential_decay(0.01,self.global_step,40, 0.70, staircase=True))
        #print(terms.shape[0])
        ##print(self.learning_rate)
        self.counter = self.counter+1
        totalbatch = terms.shape[0]/batch_size
        #learn_rate1 = self.session.run(self.learn_Rate)/5
        #print(learn_rate1)
        #print(len(terms))
        #numpy.random.seed(42)
        random.seed(5)
        tf.set_random_seed(5)
        numpy.random.seed(5)
        os.environ['PYTHONHASHSEED'] = str(5)
        
        
        indices = numpy.random.permutation(terms.shape[0])
        
        for si in range(0, terms.shape[0], batch_size):
            se = min(si + batch_size, terms.shape[0])
            slice_x = terms[indices[si:se]] + 0
            #slice_x = self.SparseDropout(slice_x)
            slice_y = tags[indices[si:se]] + 0
            slice_length = lengths[indices[si:se]] 

            _, c =self.session.run([self.train_op, self.total_loss], {self.x: slice_x, self.y: slice_y, self.lengths: slice_length,self.learn_rate: learning_rate})
            #compute average loss
            #avg_cost += c/totalbatch
     

        # <-- Your implementation goes here.
        # Finally, make sure you uncomment the `return True` below.
        #print("loss value")
        #print(self.loss)
        return True

    def step(self, learning_Rate, batch_size, terms, tags, lengths):


        indices = numpy.random.permutation(terms.shape[0])
        
        for si in range(0, terms.shape[0], batch_size):
            se = min(si + batch_size, terms.shape[0])
            slice_x = terms[indices[si:se]] + 0
            slice_y = tags[indices[si:se]] + 0
            slice_length = lengths[indices[si:se]] + 0
            
            loss_val=self.session.run(self.train_op, {self.x:slice_x, self.y:slice_y, self.lengths:slice_length})

            #print(loss_val)

    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths):
        pass


def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader
    train_filename = "/Users/harika/Desktop/NaturalLanguageProcessing/hmm-training-data/it_isdt_train_tagged.txt"
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data
    
    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    # model is the object of Sequence Model
    #print(model)
    model.build_inference()
    model.build_training()
    # send data in batches

  

    xrange=range
    for j in xrange(10):
        model.train_epoch(train_terms, train_tags, train_lengths)

        print('Finished epoch %i. Evaluating ...' % (j + 1))
        #model.get_test_accuracy()
        model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()


