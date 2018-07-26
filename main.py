from Dataset import DataSet
from Models import MLModel
from Utility import Parameters, save_results, create_dirs
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from datetime import datetime
import numpy as np
import socket
import matplotlib.pyplot as plt
if 'dei' in socket.gethostname () or 'floydhub' in socket.gethostname():
    local_machine = False
else:
    local_machine = True
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Users/root/Anaconda3/Library/bin/graphviz'

create_dirs()

datestr = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
p = Parameters(
    NN = 'CNN',
    period = 24,
    n_filters = 64,
    n_dense = 256,
    n_cl = 6,
    batch_size = 256,
    n_epochs = 50,
    learning_rate = 0.0005,
    loss_type = 'mean_squared_error'
)

print(datestr)
titolo = str(p.str)+'_date-'+datestr
print(titolo)

train = DataSet (
    'Dataset/ARS_DLR_DataSet_V2.mat',
    seed=1,
    parameters=p,
    reduced_dataset=True)
test = DataSet(
    'Dataset/ARS_DLR_Benchmark_Data_Set.mat',
    seed=1,
    parameters=p,
    reduced_dataset=True,
    labtab=train.lab_tab)

assert(str(train.lab_tab)==str(test.lab_tab))

model = MLModel(train, test, titolo)

model.init(n_filters = p.n_filters,
           n_dense=p.n_dense,
           learning_rate=p.learning_rate,
           loss_type=p.loss_type)
print(model.summ)
if local_machine:
    plot_model(model.classifier, to_file='img/model_'+p.str+'.png', show_shapes=True, show_layer_names=False)
hist = model.train_classifier(epochs=p.n_epochs, batch_size=p.batch_size)
model.load_best_model()

# evaluation of the results
loss, acc = model.classifier.evaluate(test.x,test.y)
y_true = test.cls
y_pred = np.argmax(model.classifier.predict(test.x), axis=-1)
cm = confusion_matrix(y_true,
                      y_pred,
                      labels=train.labels_int,
                      sample_weight=None)
# print results and save
print(datestr)
print('Parameters:'+p.str)
print ('Test Accuracy : ', acc)
print ('Test Loss : ', loss)
print ('Confusion matrix:\n', cm)
print ('Label table:\n' + str (test.lab_tab))
# if there is matplot.pyplot
if local_machine:
    txt_out, val_loss, val_acc, train_loss, train_acc = save_results (p.str,
                                                                      titolo,
                                                                      model.summ,
                                                                      y_true,
                                                                      y_pred,
                                                                      hist.history,
                                                                      datestr,
                                                                      cm=cm,
                                                                      model_path = model.model_path,
                                                                      log_path=model.log_path,
                                                                      n_cl=p.n_cl,
                                                                      git=False,
                                                                      model=False)
    print (txt_out)
    plt.plot ([i for i in range (p.n_epochs)], train_acc)
    plt.plot ([i for i in range (p.n_epochs)], train_loss)
    plt.plot ([i for i in range (p.n_epochs)], val_acc)
    plt.plot ([i for i in range (p.n_epochs)], val_loss)
    plt.legend(('train acc','train loss','val acc', 'val loss'))
    plt.ylim ((0, 1))
    plt.xlabel ('n epochs')
    plt.xticks ([i for i in range (p.n_epochs)])
    plt.savefig('img/'+titolo+'.png')
    plt.show()
else:
    txt_out, val_loss, val_acc, train_loss, train_acc = save_results (p.str,
                                                                      titolo,
                                                                      model.summ,
                                                                      y_true,
                                                                      y_pred,
                                                                      hist.history,
                                                                      datestr,
                                                                      model_path = model.model_path,
                                                                      log_path=model.log_path,
                                                                      n_cl=p.n_cl,
                                                                      git=True,
                                                                      model=True)
    print (txt_out)
    # bashCommand = './push.sh'
    # process = subprocess.Popen (bashCommand.split (), stdout=subprocess.PIPE)
    # output, error = process.communicate ()
    # print(output)
    # print(error)