import numpy as np
import struct

def read_images(filename):
    with open(filename, 'rb') as f:
        magic_num = struct.unpack('>i', f.read(4))[0]
        num_images = struct.unpack('>i', f.read(4))[0]
        row_size = struct.unpack('>i', f.read(4))[0]
        col_size = struct.unpack('>i', f.read(4))[0]

        print(magic_num, num_images, row_size, col_size)
        
        vec_size = row_size * col_size
        # Read all data at once and reshape
        image_data = np.frombuffer(f.read(num_images * vec_size), dtype=np.uint8)
        image_data = image_data.reshape(num_images, vec_size)
    
    return image_data, num_images, row_size, col_size, vec_size

def read_labels(filename):
    with open(filename, 'rb') as f:
        magic_num = struct.unpack('>i', f.read(4))[0]
        num_labels = struct.unpack('>i', f.read(4))[0]
        print(magic_num, num_labels)
        label_data = np.frombuffer(f.read(num_labels), dtype=np.uint8)
    return label_data

# === Load Train and Test Data ===
trainv, num_train, row_size, col_size, vec_size = read_images('data/MNIST/train_images.bin')
testv, num_test, _, _, _ = read_images('data/MNIST/test_images.bin')
trainlab = read_labels('data/MNIST/train_labels.bin')
testlab = read_labels('data/MNIST/test_labels.bin')

# === Save to a .npz file (like .mat for Python) ===
np.savez('data_all.npz', 
         num_train=num_train, 
         num_test=num_test, 
         row_size=row_size, 
         col_size=col_size, 
         vec_size=vec_size, 
         trainv=trainv, 
         testv=testv, 
         trainlab=trainlab, 
         testlab=testlab)
