## Training a machine learning model with forces

### 1. prepare data for the training

The training data can calculate by any method, but data must in ASE trajectory format. A convenient way to prepare '.traj' file is using ASE calculator(refer to ASE manual for details), after calculations the trajectory file will generated automatically. Another way is using our learning machine (as show the 'lm.py' script in this directory), but only support SIESTA currently.

### 2. training the machine learning model with data collected in step 1.

Using script 'train.py' to train the model