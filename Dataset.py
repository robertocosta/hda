import numpy as np
class DataSet(object):
    def __init__(self,
                 path,
                 period=25,
                 n_cl=12,
                 seed=None,
                 reload=False,
                 weighted=True):
        import scipy.io as sio
        from tensorflow.python.framework import random_seed
        self.path = path
        self._weighted = weighted
        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        if not reload:
            try:
                l = sio.loadmat (path + '_' + str (period) + '_backup_extended.mat')
            except FileNotFoundError:
                reload = True
        if not reload:
            self._images = l['im']
            self._labels = l['lab'][0]
            self._data_set_idx = l['idx']
            self.lab_tab = l['labtab']
            self._labels_one_hot = l['labOH']
            self._mixed_labels = l['mixedlab']
            self._mixed_labels_ind = np.where (self._mixed_labels)
            self._mixed_labels_n = len (self._mixed_labels_ind)
            self._num_examples = self._images.shape[0]
            return None

        # load mat file
        self._images, self._labels, self._data_set_idx = self.load_mat(path, reload_dataset=False)
        if len(self._data_set_idx.shape)>1:
            self._data_set_idx = self._data_set_idx[0]
        self.lab_tab = np.unique (self._labels)
        self._num_examples = self._images.shape[0]
        self._labels_one_hot = np.zeros ((self._num_examples, np.max((len (self.lab_tab),n_cl))))
        labels = np.zeros(self._labels.shape,dtype=np.int64)
        # labels to integer values
        for i in range (self._num_examples):
            lab = self._labels[i]
            labels[i] = np.where(self.lab_tab == lab)[0][0]
            self._labels_one_hot[i, labels[i]] = 1
        self._labels = labels
        self._prob_foreach_label = np.zeros((len(self.lab_tab)))
        self._id_samples_foreach_label = {}
        for i in range (len(self.lab_tab)):
            self._id_samples_foreach_label.update({i: np.where(labels==i)[0]})
            self._prob_foreach_label[i] = float(len(self._id_samples_foreach_label[i]))/self._num_examples
        data_set = []
        labels = []
        labels_one_hot = []
        transient_blocks = []
        # building blocks of inputs
        for i in range(period//2,self._num_examples-period//2):
            from_ind = i-period//2
            to_ind = i+period//2
            # if period % 2 > 0:
            #     to_ind += 1
            data_set_idx = self._data_set_idx[from_ind:to_ind]
            # add the block only if the dataset is from the same person
            if len(np.unique(data_set_idx))==1:
                data = self._images[from_ind:to_ind]
                # add FFT of module of the acceleration to the block
                acc_module = data[:,-1]
                FFT = np.fft.rfft(acc_module-np.mean(acc_module),2*(period+1)-1)
                FFT=FFT[1:] # take the DC component out
                assert(len(FFT)==period)
                fft_mod = np.reshape(abs(FFT),(period,1))
                fft_mod = fft_mod / np.sqrt(np.sum(fft_mod**2))
                fft_phase = (np.reshape(np.angle(FFT),(period,1))+np.pi)/2/np.pi
                # n_blocks = np.array([6, 3, 2])
                # n_el_per_block = period // n_blocks
                # fft_filt = np.zeros ((len(FFT),len(n_blocks)))
                # for j, el in enumerate(zip(n_blocks,n_el_per_block)):
                #     (block_n, el_per_block) = el
                #     for k in range(block_n):
                #         from_ind = el_per_block * k
                #         to_ind = from_ind + el_per_block
                #         if k==block_n-1: # if it is the last block
                #             to_ind = len(FFT)
                #         fft_filt[from_ind:to_ind,j] = np.mean(fft_mod[from_ind:to_ind])
                data_set.append(np.hstack((data,fft_phase,fft_mod)))
                labels.append(self._labels[i])
                labels_one_hot.append(self._labels_one_hot[i,:])
                # transient_blocks is True iff in that period there is more than one activity
                transient_blocks.append(len(np.unique(self._labels[from_ind:to_ind]))>1)

        self._images = np.array(data_set)
        self._labels = np.array(labels)
        self._labels_one_hot = np.array(labels_one_hot)
        self._mixed_labels = np.array(transient_blocks)
        self._mixed_labels_ind = np.where(self._mixed_labels)
        self._mixed_labels_n = len(self._mixed_labels_ind)
        self._num_examples = self._images.shape[0]

        sio.savemat(path+ '_' + str (period) + '_backup_extended.mat',{'im':self._images,
                                                 'lab':self._labels,
                                                 'idx':self._data_set_idx,
                                                 'labtab':self.lab_tab,
                                                 'labOH':self._labels_one_hot,
                                                 'mixedlab':self._mixed_labels})

    @property
    def x(self):
        shap = (self._images.shape[0], self._images.shape[1], self._images.shape[2], 1)
        tmp = np.reshape (self._images, shap)
        return tmp

    @property
    def y(self):
        return self._labels_one_hot

    @property
    def labels_int(self):
        return list(np.unique (self.cls))

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def cls(self):
        lab = self._labels
        if lab.shape[0]==1:
            lab = lab[0]
        self._labels = lab
        return lab

    @property
    def n_cl(self):
        return self._labels_one_hot.shape[1]

    @property
    def weights(self):
        w = {}
        for i in range (self.n_cl):
            if self._weighted:
                freq = float(len(np.where (self.y[:, i])[0])) / self.num_examples
                if freq>0:
                    w.update ({i: 1/freq})
                else:
                    w.update ({i: 0})
            else:
                w.update ({i: 1})
        return w

    def shuffle_idx(self, batch_size):
        shuffled_idx_dict = {}
        n_samples = np.zeros (len (self.lab_tab), dtype=np.int64)
        for i in range(len (self.lab_tab)):
            ordered_idx = self._id_samples_foreach_label[i]
            perm0 = np.arange (len (ordered_idx))
            np.random.shuffle (perm0)
            shuffled_idx_dict.update ({i: ordered_idx[perm0]})
            n_samples[i] = np.round (batch_size * self._prob_foreach_label[i])
        assert (np.sum (n_samples) == batch_size)
        curr_ind = np.zeros (len (self.lab_tab), dtype=np.int64)
        shuffled_idx = np.zeros (self._num_examples, dtype=np.int64)
        glob_ind = 0
        for i in range(np.floor(self._num_examples/batch_size)):
            current_batch_idx = np.zeros (batch_size, dtype=np.int64)
            current_batch_ind = 0
            for j in range(len (self.lab_tab)):
                current_batch_idx[current_batch_ind:current_batch_ind+n_samples[j]] = \
                    shuffled_idx_dict[j][curr_ind[j]:curr_ind[j]+n_samples[j]]
                current_batch_ind[j] += n_samples[j]
            perm1 = np.arange (batch_size)
            np.random.shuffle (perm1)
            shuffled_idx[glob_ind:glob_ind+batch_size] = current_batch_idx[perm1]
            glob_ind += batch_size
        for j in range (len (self.lab_tab)):
            n = n_samples[j] - curr_ind[j]
            shuffled_idx[glob_ind:glob_ind + n] = shuffled_idx_dict[j][curr_ind[j]:]
            # shuffled_idx[glob_ind:glob_ind + n] = shuffled_idx_dict[j][curr_ind[j]:curr_ind[j] + n]
            glob_ind += n
        self.shuffle_from_ind(shuffled_idx)

    def shuffle_from_ind(self,idx):
        self._images = self._images[idx]
        self._labels = self._labels[idx]
        self._labels_one_hot = self._labels_one_hot[idx]
        self._id_samples_foreach_label = {}
        for i in range (len(self.lab_tab)):
            self._id_samples_foreach_label.update({i: np.where(labels==i)[0]})

    def shuffle_idx_semi_ordered(self,lab,n_iter):
        idx = [i for i in range (self._num_examples)]
        for l in lab:
            j = 0
            idx_label = self._id_samples_foreach_label[l]
            for i in range(len (idx_label),len (idx_label)-n_iter,-1):
                if not self._labels[j] in lab:
                    idx[i] =  self._labels[j]
                    idx[j] = l
        perm0 = np.arange (n_iter*len(lab))
        np.random.shuffle (perm0)
        idx[:len(perm0)] = idx[perm0]
        self.shuffle_from_ind(idx)

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        firs_labels = [1,3]
        first_batches_n = 5
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle_idx(batch_size)
            # putting running (0) and transac (3) in the first positions
            shuffle_idx_semi_ordered (self, firs_labels, batch_size*first_batches_n)
            # shuffled_idx = {}
            # self._shuffled_idx = shuffled_idx
            # # giving mixed labels to the algorithm firstly
            # perm0 = np.arange(self._mixed_labels_n)
            # np.random.shuffle (perm0)
            # self._images[:self._mixed_labels_n] = self.images[self._mixed_labels_ind[perm0]]
            # self._labels = self.labels[self._mixed_labels_ind[perm0]]
            # perm1 = np.arange (self._num_examples-self._mixed_labels_n)
            # np.random.shuffle (perm1)
            # non_mixed_ind = np.where(self._mixed_labels==0)
            # self._images[self._mixed_labels_n:] = self.images[non_mixed_ind[perm1]]
            # self._labels = self.labels[non_mixed_ind[perm1]]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                self.shuffle_idx (batch_size)
                shuffle_idx_semi_ordered (self, first_labels, batch_size * first_batches_n)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def load_mat(self, path, reload_dataset=False, redued_dataset=True):
        import scipy.io as sio
        if not reload_dataset:
            try:
                data_set = sio.loadmat (path + '_backup.mat')
            except FileNotFoundError:
                reload_dataset = True
        if not reload_dataset:
            return data_set['data'], data_set['labels'], data_set['data_set_idx']
        # returns [body_acc,global_acc,body_omega,module_acc]
        data_set = sio.loadmat (path)
        data = []
        labels = []
        data_set_idx = []
        index = 0
        for k in data_set.keys():
            cur_set = data_set[k]
            if type (cur_set) is np.ndarray:
                activity = cur_set[0, 2]
                changes = cur_set[0, 3]
                signal = cur_set[0, 0]
                attitude = cur_set[0, 1]
                for i in range (activity.shape[1]):
                    inf = changes[0, 2*i]
                    sup = changes[0, 2*i + 1]
                    for j in range(inf, sup):
                        d_cosine_matrix = np.reshape (attitude[j, 1:], (3, 3)).T
                        body_acc = signal[j, 1:4]
                        # compute the acceleration in global coordinates
                        global_acc = np.matmul(d_cosine_matrix, np.transpose(body_acc)).T
                        body_omega = signal[j, 5:8]
                        # normalize angular velocity
                        body_omega = body_omega / (np.sqrt(np.sum(body_omega**2))+1e-8)
                        # compute the module of the acceleration
                        module_acc = np.sqrt(np.sum(body_acc**2))
                        # normalize acceleration
                        body_acc = body_acc / module_acc
                        global_acc = global_acc / module_acc
                        data.append(np.hstack((body_acc,global_acc,body_omega,module_acc)))
                        # grouping of transient states
                        if redued_dataset:
                            if activity[0, i][0] in ['TRANSDW', 'TRANSUP', 'TRNSACC', 'TRNSDCC', 'JUMPING', 'FALLING']:
                                labels.append ('TRANSAC')
                            else:
                                labels.append (activity[0, i][0])
                        else:
                            labels.append (activity[0, i][0])
                        data_set_idx.append(index)
            index += 1
        data = np.array(data)
        labels = np.array(labels)
        data_set_idx = np.array(data_set_idx)
        acc_modules = data[:,-1]
        # delta_acc = np.max(acc_modules)-np.min(acc_modules)
        # acc_modules = (acc_modules-np.min(acc_modules))/delta_acc
        data[:, -1] = acc_modules
        sio.savemat(path+'_backup.mat',{'data':data,'labels':labels, 'data_set_idx':data_set_idx})
        return data, labels, data_set_idx
