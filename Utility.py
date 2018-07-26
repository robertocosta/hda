class Parameters (object):
    def __init__(self,
                 NN='FFNN',
                 period=10,
                 n_filters=32,
                 n_dense=128,
                 n_cl=12,
                 batch_size=128,
                 n_epochs=10,
                 learning_rate=0.001,
                 loss_type='categorical_crossentropy'):
        self._NN = NN
        self._period = period
        self._n_filters = n_filters
        self._n_dense = n_dense
        self._loss_type = loss_type
        self._n_cl, self._batch_size, self._n_epochs, self._learning_rate = n_cl, batch_size, n_epochs, learning_rate

    @property
    def str(self):
        s = self._NN+'_t-'+str(self._period)\
            +'_nfilters-'+str(self._n_filters)\
            +'_ndense-'+str(self._n_dense)\
            +'_ncl-'+str(self._n_cl)\
            +'_batchsize-'+str(self._batch_size)\
            +'_nepochs-'+str(self._n_epochs)\
            +'_lr-'+str(self._learning_rate)\
            +'_loss-'+self.loss_type.replace('_','')
        return s

    @property
    def NN(self):
        return self._NN

    @property
    def period(self):
        return self._period

    @property
    def n_filters(self):
        return self._n_filters

    @property
    def n_dense(self):
        return self._n_dense

    @property
    def n_cl(self):
        return self._n_cl

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def loss_type(self):
        return self._loss_type

def labels_to_one_hot(lab,n_cl):
    import numpy as np
    oh = np.zeros((lab.shape[0],n_cl))
    for idx in range (oh.shape[0]):
        oh[idx,lab[idx]]=1
    return oh

def get_hist_csv(h):
    import numpy as np
    csv = ''
    n_epochs = len(h['loss'])
    csv_tab = np.zeros ((n_epochs, len (h.keys())))
    index = 0
    head_csv = ''
    for key in h:
        csv_tab[:, index] = h[key]
        index += 1
        head_csv += key + ','
    csv += head_csv[:-1]+'\n'

    format_str = ''
    for i in range(index):
        format_str += "{" + str(i) + ":f},"
    format_str = format_str[:-1] + '\n'
    for i in range(n_epochs):
        row = csv_tab[i, :]
        csv += format_str.format(*row)
    return csv, csv_tab

def save_results(parameters,
                 commento,
                 nn_summary,
                 y_true,
                 y_pred,
                 hist,
                 datestr,
                 n_cl=6,
                 model_path=None,
                 log_path=None,
                 git=True,
                 model=False):
    import scipy.io as sio
    import os
    import numpy as np
    from sklearn.metrics import classification_report
    if not model_path:
        model_path = 'models_backup/' + commento + '.hdf5'
    if not log_path:
        log_path = 'log/'+commento+'/*'
    y_pred = np.array (y_pred)
    y_true = np.array (y_true)
    l = y_pred.shape[0]
    h = hist
    h.keys()
    y_true_oh = labels_to_one_hot (y_true,n_cl)
    y_pred_oh = labels_to_one_hot (y_pred,n_cl)

    # reports
    cl_rep = classification_report (np.array (y_true_oh), np.array (y_pred_oh))
    csv_hist, hist = get_hist_csv(h)
    labels_vs_true = np.hstack ((np.reshape (y_pred, (l, 1)), np.reshape (y_true, (l, 1))))

    # writing to txt file everything
    out1_name = commento+'_{}.txt'.format(parameters)
    out1 = open('txt/'+out1_name, "w")
    out1.write(str(nn_summary))
    out1.write('\n!~!\n')
    out1.write (str(cl_rep))
    out1.write ('\n!~!\n')
    out1.write(csv_hist+'\n')
    out1.write('predicted_class,true_class\n')
    for i in range(labels_vs_true.shape[0]):
        out1.write('{:02d},{:02d}\n'.format(
            int(labels_vs_true[i, 0]), int(labels_vs_true[i, 1])))
    out1.close()

    # writing to csv file the history
    out2_name = '{}_metrics.csv'.format(commento)
    out2 = open('txt/'+out2_name, "w")
    out2.write(csv_hist)
    out2.close()

    # writing to mat file everything
    mat_file = h
    mat_file.update ({'cl_rep': cl_rep,
                      'csv_hist': csv_hist,
                      'parameters': parameters,
                      'commento': commento,
                      'nn_summary': nn_summary,
                      'datestr': datestr,
                      'log_path': log_path})
    mat_name = commento+'.mat'
    sio.savemat('mat/'+mat_name, mat_file)

    # update files list
    with open("txt/list.txt", "a") as myfile:
        myfile.write(out1_name + '\n')
        myfile.write(out2_name + '\n')
    with open("mat/list.txt", "a") as myfile:
        myfile.write(mat_name + '\n')

    if git:
        out4_name = commento + '_git_push.sh'
        out4 = open(out4_name, "w")
        out4.write('git pull\n')
        out4.write('git add mat/list.txt txt/list.txt txt/'+out1_name +
                   ' txt/'+out2_name+' mat/'+mat_name+'\n')
        if model:
            out4.write('git add '+model_path+'\n')
        out4.write('git commit -m "output '+commento+'..params:'+parameters+'"\n')
        out4.write('git push\n')
        print('git pushed')

    return 'Classification report:\n'+cl_rep + '\nLoss and accuracy during training:\n' + csv_hist, \
           hist[:, 0], hist[:,1], hist[:,2], hist[:,3]