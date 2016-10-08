import pickle
import matplotlib.pyplot as plt

epoch_loss_lst, epoch_error_lst,  epoch_error_lst_val, \
ter, LEARNING_RATE, MOMENTUM, OMEGA, INPUT_NOISE_STD, \
n_hidden = pickle.load(open( "ctc_blstm710.pkl", "rb" ))


plt.plot(epoch_error_lst)
plt.plot(epoch_error_lst_val)
plt.show()
