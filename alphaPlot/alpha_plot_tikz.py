import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save


mat_contents = sio.loadmat('alpha.mat')
alpha = mat_contents['alpha']
out = ['<sos>', 'sil', 'hh', 'ih', 'f', 'sil', 'k', 'er', 
       'r', 'ow', 'ow', 'sil', 'sil', 't', 'ah', 'm', 'aa', 
       'hh', 'hh', 'ae', 'v', 'er', 'r', 'r', 'n', 'n', 'sil', 
       'f', 'er', 'er', 'm', 'iy', 'iy', 'iy', 'iy', 'iy', 'iy', 
       'iy', 'iy', 'sil', 'sil', 't', 'uw', 'sil', '<eos>']

dat = plt.matshow(alpha, aspect='auto', interpolation='nearest')
plt.xticks(range(len(out)), out, size='small')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.colorbar(dat)
#fig.savefig('test2png.png', dpi=100)
tikz_save('alpha.tex')

