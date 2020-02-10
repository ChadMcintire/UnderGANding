import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10

print("x_train shape[0]", np.shape(x_train[0][0])) 
print("y_train shape[0]", np.shape(y_train[0]))

print("x_train 0", x_train[0][0],"\n" )
print("y_train 0", y_train[0],"\n" )
print("x_test 0", x_test[0]  ,"\n" )
print("y_test 0", y_test[0]  ,"\n" ) 

print("x_train shape", np.shape(x_train)) 
print("y_train shape", np.shape(y_train))
print("x_test shape ", np.shape(x_test) ) 
print("y_test shape",  np.shape(y_test) ) 


x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

#keras.utils.to_categorical(y, num_classes=None, dtype='float32')
y_train_onehot = to_categorical(y_train, NUM_CLASSES)
y_test_onehot = to_categorical(y_test, NUM_CLASSES)

n_to_show = 10

indices = np.random.choice(range(len(x_test)), n_to_show)
print("indicies", indices)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)


class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
class_val = list(np.unique(y_test))
print(class_val)


res = {class_names[i]: class_val[i] for i in range(len(class_names))}
print("y_test indices ", y_test[indices[0]], "\n\n\n")


print(res)
print(list(res.keys())[list(res.values()).index(y_test[indices[0]])]) 

print("list  of vals", list(res.values()).index(y_test[indices[0]]))
print(list(res.keys()))
print(list(res.keys())[3])


for i, idx in enumerate(indices):
    img_idx_val = list(res.values()).index(y_test[idx])
    img_idx_key = list(res.keys())[img_idx_val] 
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.7, 'act = ' + str(img_idx_key), fontsize=10 , ha='center', transform=ax.transAxes)
    ax.imshow(img)

#plt.show()

