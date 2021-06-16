from utils import *

cudnn.benchmark = True
torch.cuda.set_device('cuda:2')
model, model_list, in_inps, in_inps_lab = start("MNIST")

prd = evaluate(model, model_list, 0.85, in_inps[0], "MNIST")

print("Pred: %s Label: %s" % (prd, in_inps_lab[0].item()))


