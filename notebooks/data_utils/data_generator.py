import numpy as np
import keras
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size, rg, file_names, y_true, scaler, 
                 MAXR=17200, target_names=['E'],shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.rg = rg
        self.shuffle = shuffle
        self.counter = 0
        self.file_names=file_names
        self.y_true = y_true
        self.scaler = scaler
        self.target_names = target_names
        self.MAXR=MAXR
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size)) - 20

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X = self.data_lpmt[indexes]
        y = self.y_cur[indexes]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        i = self.counter % len(self.file_names)
        self.data_lpmt = np.load(self.file_names[i])
        start = self.rg[i]
        end = self.rg[i+1]
        ys = self.y_true[(self.y_true['evtID'] >= start) 
                            & (self.y_true['evtID'] < end)]
        mask = (ys.R <= self.MAXR)
        self.data_lpmt = self.data_lpmt[mask]
        self.y_cur = ys[mask][self.target_names].values
        #Scaler
        self.scaler(self.data_lpmt, self.y_cur)
        
        self.indexes = np.arange(self.data_lpmt.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.counter += 1

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        pass
    
    
class DataGeneratorBig(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size, rg, file_names, y_true_names, scaler, 
                 MAXR=17200, target_names=['E'],shuffle=True):
        """
        param: rg: list with start, end indexes in each big piece, small pieces
        param: file_names: all learning_files
        param: mask_names: names of masks per each big piece
        param: y_true_names: names of true infos for each big piece
        """
        self.batch_size = batch_size
        self.rg = rg
        self.shuffle = shuffle
        self.counter = 0
        
        self.file_names   = file_names
        self.y_true_names = y_true_names
        
        self.big_pieces_len = len(self.y_true_names)
        self.small_pieces_len = len(self.rg) - 1
        print("Big pieces", self.big_pieces_len)
        print("Small pieces", self.small_pieces_len)
        
        
        self.scaler = scaler
        self.target_names = target_names
        self.MAXR=MAXR
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print("CALL LEN", int(np.floor(len(self.indexes) / self.batch_size)))
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X = self.data_lpmt[indexes]
        y = self.y_cur[indexes]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        i = self.counter % len(self.file_names)
        print("Counter: ", self.counter)
        big_piece_id   = ( i // self.small_pieces_len ) % self.big_pieces_len
        small_piece_id = i % self.small_pieces_len
        print('\n Big id ', big_piece_id, 'small id', small_piece_id) 
        print("\n Load files", self.file_names[i],' ', self.y_true_names[big_piece_id])
        self.data_lpmt = np.load(self.file_names[i])
        
        self.y_true = pd.read_csv(self.y_true_names[big_piece_id]) 
        
        start = self.rg[small_piece_id]
        end = self.rg[small_piece_id+1]
        
        print("Ids from: ", start, " to ", end)
        ys = self.y_true[(self.y_true['evtID'] >= start) 
                            & (self.y_true['evtID'] < end)]
        
        mask = (ys.R <= self.MAXR)
        self.data_lpmt = self.data_lpmt[mask]
        self.y_cur = ys[mask][self.target_names].values
        #Scaler
        self.scaler(self.data_lpmt, self.y_cur)
        
        self.indexes = np.arange(self.data_lpmt.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.counter += 1

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        pass