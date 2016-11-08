import pickle
import matplotlib.pyplot as plt

epoch_loss_lst, epoch_loss_lst_val, \
tl, LEARNING_RATE, MOMENTUM, OMEGA, epoch  \
 = pickle.load(open( "savedValsLasAdam.spchcl22.esat.kuleuven.be-2016-11-05.pkl", "rb" ))


plt.plot(epoch_loss_lst)
plt.plot(epoch_loss_lst_val)
plt.show()
