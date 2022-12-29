#This reader generates trainable batches from the preprocessed training text from the data parser script.

# libraries
import pickle
import random

class Data_Reader:
    def __init__(self, cur_train_index=0, load_list=False):
        self.training_data = pickle.load(open('data/conversations_lenmax22_formersents2_with_former', 'rb'))
        self.data_size = len(self.training_data)
        if load_list:
            self.shuffle_list = pickle.load(open('data/shuffle_index_list', 'rb'))
        else:    
            self.shuffle_list = self.shuffle_index()
        self.train_index = cur_train_index

    # get batch number from data
    def get_batch_num(self, batch_size):
        return self.data_size // batch_size

    #shuffle index from data
    def shuffle_index(self):
        shuffle_index_list = random.sample(range(self.data_size), self.data_size)
        pickle.dump(shuffle_index_list, open('data/shuffle_index_list', 'wb'), True)
        return shuffle_index_list

    #generate batch indeices based on batch numbers
    def generate_batch_index(self, batch_size):
        if self.train_index + batch_size > self.data_size:
            batch_index = self.shuffle_list[self.train_index:self.data_size]
            self.shuffle_list = self.shuffle_index()
            remain_size = batch_size - (self.data_size - self.train_index)
            batch_index += self.shuffle_list[:remain_size]
            self.train_index = remain_size
        else:
            batch_index = self.shuffle_list[self.train_index:self.train_index+batch_size]
            self.train_index += batch_size
        return batch_index

    # generate training batches
    def generate_training_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]   # batch_size of conv_b
        return batch_X, batch_Y

    # generate training batches with previous former batches
    def generate_training_batch_with_former(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]   # batch_size of conv_b
        former = [self.training_data[i][2] for i in batch_index]    # batch_size of former utterance
        return batch_X, batch_Y, former

    # generate testing batches
    def generate_testing_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
        return batch_X