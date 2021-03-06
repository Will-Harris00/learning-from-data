import os
import numpy as np
import matplotlib.pyplot as plt

glioma_count = len(os.listdir('cleaned/Training/glioma')) + len(os.listdir('cleaned/Testing/glioma'))
meningioma_count = len(os.listdir('cleaned/Training/meningioma')) + len(os.listdir('cleaned/Testing/meningioma'))
pituitary_count = len(os.listdir('cleaned/Training/pituitary')) + len(os.listdir('cleaned/Testing/pituitary'))
no_tumor_count = len(os.listdir('cleaned/Training/notumor')) + len(os.listdir('cleaned/Testing/notumor'))

tumor = ('glioma','meningioma', 'pituitary', 'no tumor')

x_pos = np.arange(len(tumor))
sample_size = [glioma_count, meningioma_count, pituitary_count, no_tumor_count]

plt.bar(tumor, sample_size, color=['mediumvioletred', 'orange', 'slategrey', 'dodgerblue', 'crimson'])
plt.xticks(x_pos, tumor)
plt.title('Number of occurrences of each kind of brain tumor in the dataset')
plt.ylabel('Number of samples')
plt.xlabel('Type of tumor')
for i in range(len(tumor)):
        plt.text(i, sample_size[i], sample_size[i], ha = 'center')
plt.show()
