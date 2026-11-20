# Morphological-classification-of-neurons-
Author, Guanglian Li, Rongqing Wu, Haixing Song, Xuqing Zeng, Zhen Wei, Fuyun He in China. 
li_guanglian@stu.gxnu.edu.cn


# data introduction
## the structure of swc-format file.
```
 1 1 0 0 0 10.75 -1
 2 1 -4.89 9.52 0.42 10.75 1
 3 1 4.89 -9.53 -0.42 10.75 1
 4 2 -6.54 -9.66 -2.02 0.1144 1
 5 2 -6.54 -10.77 -2.37 0.1144 4
 6 2 -6.62 -11.91 -2.55 0.1144 5
 7 2 -6.85 -13.02 -2.62 0.1144 6
 8 2 -7.03 -14.16 -2.63 0.1144 7
 9 2 -7.16 -15.3 -2.59 0.1144 8
 10 2 -7.4 -16.41 -2.51 0.1144 9
....    
```
the structure of each line for swc file:
 ['id', 'type', 'x', 'y', 'z', 'width', 'parent'], where,
id: ID of the node
type: 1-soma, 2-axon, 3-dendritic, 4-apical dendrite, 5-branch point, 6-end point
x: X coordinate of the node
y: Y coordinate of the node
z: Z coordinate of the node
width: the radius of the neuron
parent: father id of point, -1 indicates the root node

In swc_data folder:

swc_v0 raw data that crawler from the site  
swc_v1 first version of data repaired by software  
Human neurons provided by the NeuroMorpho.Org database, covering six types: granule, glutamatergic, induced pluripotent stem cell (iPSC)-derived, Pyramidal 7 Allen cell type, sensory receptor, and von Economo neurons



##  Data catalog format
swc is similar to png's directory format  
for example,there are two folders,train, test under swc_v0  
There are 4 folders under train, such as principal cells, interneurons, glial cells, and sensory receptor cells. 
There are several folders under the principal cell, such as pyrimidal, each containing the same class of neurons. 


## Dividing datasets
The different types of swc files are converted into 3D voxel images through Swc_transfer_dataset.py, and then split.py is run. By modifying the file_path('./swc_data/3D_Img_human') parameter, the three datasets are divided into training and testing sets in a 7:3 ratio. This path is the one before the division of the human neuron dataset.


# Operating instructions
## Environment
NVIDIA RTX3090 GPU(24GB)  
python 1.10  


## Structure
The main model proposed is the IRC_Net model, which consists of three code files: Img_input_3D, IRC_Net, and IRC_Net_train_3D. There are also several comparison models, such as C3DNet, EfficientNet3D, Resnet_18, and ShuffleNet.


## Img_input_3D
The Img_input_3D file contains a Dataset class that inherits from torch.utils.data.Dataset, which is used for data reading and preprocessing.

## IRC_Net
The IRC_Net file contains the code required for building the model, namely a deep network architecture based on 3D CNN, incorporating the improved 3D residual blocks, attention mechanism, and depthwise separable convolution layers, which is represented by the function cnn3d_multiscale.


## IRC_Net_train_3D
The IRC_Net_train_3D file contains the code for training or testing data, and the required parameters and data directories are located at the beginning of the file.



## Runing the code
Which model needs to be trained, open the corresponding IRC_Net file,comment off the corresponding part according to 4 or 12 categories and modify the data catalog data_dir and superparameters.
Data_dir the format please refer to the original modification. If you use new data, you need to run Swc_transfer_dataset.py and split.py to pre-divide the training set and test set and place neurons by class. 

## cuda
If there are multiple GPUs on the server, please use os.environ to specify the GPU. Otherwise, errors might occur.
The IRC_Net model uses only one GPU.
