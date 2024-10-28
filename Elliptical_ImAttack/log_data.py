import json
import wandb 

attack = "fgm"
wandb.init(project="robust_attack")
wandb.run.name = f"sym_6itper1stlayer_{attack}"
for i in range(1,7):
    eps = str(i/255)[0:-2]
    f = open(f'/root/data/sym_6itper1stlayer_{attack}_{attack}_{str(i)}.txt')
    data = json.load(f)
    wandb.log({f"loss_{attack}":data['loss'], "eps":float(eps)})
    wandb.log({f"acc1_{attack}":data['acc1'], "eps":float(eps)})
    wandb.log({f"acc5_{attack}":data['acc5'], "eps":float(eps)})