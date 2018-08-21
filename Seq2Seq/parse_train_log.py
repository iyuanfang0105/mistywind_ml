import numpy as np
import matplotlib.pyplot as plt

train_history = 'log/train_history.npy'
history = np.load(train_history).item()

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
ax0.plot(history['acc'], label='train_acc')
ax0.plot(history['val_acc'], label='val_acc')
ax0.legend()

ax1.plot(history['loss'], label='train_loss')
ax1.plot(history['val_loss'], label='val_loss')

plt.show()

print()