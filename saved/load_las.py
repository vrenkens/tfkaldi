import pickle
import matplotlib.pyplot as plt

epoch_loss_lst, epoch_loss_lst_val, \
tl, LEARNING_RATE, MOMENTUM, OMEGA  \
 = pickle.load(open( "savedValsLastAdam.spchcl22.esat.kuleuven.be2016-10-28 07:19:11.626192.pkl", "rb" ))


plt.plot(epoch_loss_lst)
plt.plot(epoch_loss_lst_val)
plt.show()
