import os

#### ATO ####

model_path = "trained_models/forced_turb/ato/"

cmd_infer = "python3 test_scripts/forced_turb_ato.py --gpu '0'"
cmd_infer += " --V0 data/forced_turb/ -n 5 -t 200 -b 199" 
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --enc_model " + model_path + "encoder_model"
cmd_infer += " --cor_model " + model_path + "corrector_model"
cmd_infer += " --dec_model " + model_path + "decoder_model"
cmd_infer += " --nu 0.1 --dt 0.2 -l 128 -r 128"
cmd_infer += " -o inferences/forced_turb/ato/"

os.system(cmd_infer)


############# State-of-the-art solvers ###################

model_path = "trained_models/forced_turb/sol/"

cmd_infer = "python3 test_scripts/forced_turb_sol.py --gpu '0'"
cmd_infer += " --V0 data/forced_turb/ -n 5 -t 200" 
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "corrector_model"
cmd_infer += " --nu 0.1 --dt 0.2 -l 128 -r 32"
cmd_infer += " -o inferences/forced_turb/sol/"

os.system(cmd_infer)

model_path = "trained_models/forced_turb/super_res/"

cmd_infer = "python3 test_scripts/forced_turb_super_res.py --gpu '0'"
cmd_infer += " --V0 inferences/forced_turb_sol/ -n 5 -t 200 -b 199"
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "decoder_model"
cmd_infer += " -l 128 -r 32"
cmd_infer += " -o inferences/forced_turb/sol/"

os.system(cmd_infer)

model_path = "trained_models/forced_turb/dilresnet/"

cmd_infer = "python3 test_scripts/forced_turb_dilresnet.py --gpu '0'"
cmd_infer += " --V0 data/forced_turb/ -n 5 -t 200" 
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "solver_model"
cmd_infer += " -l 128 -r 32"
cmd_infer += " -o inferences/forced_turb/dilresnet/"

os.system(cmd_infer)

model_path = "trained_models/forced_turb/super_res/"

cmd_infer = "python3 test_scripts/forced_turb_super_res.py --gpu '0'"
cmd_infer += " --V0 inferences/forced_turb/dilresnet/ -n 5 -t 200 -b 199"
cmd_infer += " --stats " + model_path + "dataStats.pickle"
cmd_infer += " --model " + model_path + "decoder_model"
cmd_infer += " -l 128 -r 32"
cmd_infer += " -o inferences/forced_turb/dilresnet/"

os.system(cmd_infer)
