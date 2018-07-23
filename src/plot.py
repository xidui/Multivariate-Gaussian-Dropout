import matplotlib.pyplot as plt


def load_data(filename):
    with open('result/{0}'.format(filename)) as f:
        train_loss = []
        val_loss = []
        for line in f.readlines():
            elements = line.split()
            train_loss.append(float(elements[5]))
            val_loss.append(float(elements[9]))
        return train_loss, val_loss


train_loss, val_loss = load_data(
    "layer:7|conv_1:20:{'rate': 0.1, 'type': 'bernoulli'}|conv_2:50:None|fc1:500:None|fc2:200:None|.txt")


plt.clf()
font = {'size': 16, 'weight': 'bold'}
plt.plot(train_loss[:20], label='training cross-entropy loss')
plt.plot(val_loss[:20], label='validation cross-entropy loss')
plt.legend(loc='best')
plt.xlabel('Epochs', **font)
plt.ylabel('Train/Validation Cross-entropy Loss', **font)
plt.title('conv 1, bernoulli 0.1, accuracy:70%')
plt.savefig('6_cross_entropy_loss_16.pdf')
