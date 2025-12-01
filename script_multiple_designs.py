
from auto_lens_design import main

config_list = ['20250828_2P_Asphere_FOV45_Diag3_W450.yml', '20250828_2P_Asphere_FOV45_Diag3_W525.yml',
               '20250828_2P_Asphere_FOV45_Diag3_W600.yml', '20250828_2P_Asphere_FOV45_Diag3_W675.yml']

num_config = len(config_list)

for i in range(num_config):
    config_name = config_list[i]
    file_name = './configs/' + config_name
    main(file_name)
