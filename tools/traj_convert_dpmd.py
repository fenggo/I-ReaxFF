import dpdata
import numpy as np

# dpdata.LabeledSystem('OUTCAR').to('deepmd/npy', 'data', set_size=200)

systems_fe3_0 = dpdata.MultiSystems.from_file(file_name='fe3-0.traj',fmt='ase/structure')
print('\n# the data contains %d systems' % len(systems_fe3_0))
# print(systems_fe3_0)
print(systems_fe3_0.systems)
print('\n---------**************************----------\n')
data_fe3_0    = systems_fe3_0.systems['Fe54']
print('# the data contains %d frames' % len(data_fe3_0))
print(data_fe3_0)

# random choose 20 index for validation_data
index_validation = np.random.choice(50,size=20,replace=False)     
# other indexes are training_data
index_training = list(set(range(50))-set(index_validation))       
data_training = data_fe3_0.sub_system(index_training)
data_validation = data_fe3_0.sub_system(index_validation)
# all training data put into directory:"training_data" 
data_training.to_deepmd_npy('training_data')               
# all validation data put into directory:"validation_data"
data_validation.to_deepmd_npy('validation_data')           
print('\n# the training data contains %d frames' % len(data_training)) 
print('# the validation data contains %d frames \n' % len(data_validation)) 
