import os

#### ATO ####

model_path = "trained_models/karman/ato/"

cmd_infer = "python3 test_scripts/karman_ato.py --gpu '0'"
cmd_infer += " --V0 data/karman/ -t 2000 -b 1999" 
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --enc_model " + model_path + "encoder_model"
cmd_infer += " --cor_model " + model_path + "corrector_model"
cmd_infer += " --dec_model " + model_path + "decoder_model"
cmd_infer += " -l 128 -r 128"
cmd_infer += " -o inferences/karman/ato/"

os.system(cmd_infer)


############# State-of-the-art solvers ###################

model_path = "trained_models/karman/sol/"

cmd_infer = "python3 test_scripts/karman_sol.py --gpu '0'"
cmd_infer += " --V0 data/karman/ -t 2000" 
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "corrector_model"
cmd_infer += " -l 128 -r 32"
cmd_infer += " -o inferences/karman/sol/"

os.system(cmd_infer)

model_path = "trained_models/karman/super_res/"

cmd_infer = "python3 test_scripts/karman_super_res.py --gpu '0'"
cmd_infer += " --V0 inferences/karman/sol/ -t 2000 -b 1999"
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "decoder_model"
cmd_infer += " -l 128"
cmd_infer += " -o inferences/karman/sol/"

os.system(cmd_infer)

model_path = "trained_models/karman/dilresnet/"

cmd_infer = "python3 test_scripts/karman_dilresnet.py --gpu '0'"
cmd_infer += " --V0 data/karman/ -t 2000" 
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "solver_model"
cmd_infer += " -l 128 -r 32"
cmd_infer += " -o inferences/karman/dilresnet/"

os.system(cmd_infer)

model_path = "trained_models/karman/super_res/"

cmd_infer = "python3 test_scripts/karman_super_res.py --gpu '0'"
cmd_infer += " --V0 inferences/karman/dilresnet/ -t 2000 -b 1999"
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "decoder_model"
cmd_infer += " -l 128"
cmd_infer += " -o inferences/karman/dilresnet/"

os.system(cmd_infer)
