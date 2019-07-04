import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import subprocess

import torch
import torch.optim as optim
from dataloader import get_data_loader
from tensorboardX import SummaryWriter
from agent.cagent import CBCAgent
from utils import print_over_same_line
from evaluate import evaluate_model

snapshot_dir = "./snaps"
tensorboard_dir="./tensorboard"
# train.py --weighted --snap -history=3 -bsize=8 -lr=5e-4 -name=july2_h3w -val_episodes=10 -val_frames=300
# arg parse ----------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cont', '-c', action = "store_true", dest="continute_training", default = False, help ='continue training')
parser.add_argument('--snap', '-s', action = "store_true", dest="save_snaps", default = True, help='save snapshots every 5 epochs')
parser.add_argument('-name', type=str, dest="name", default="debug", help='name of the run')
parser.add_argument('-lr', type=float, dest="learning_rate", default=5e-4, help='learning rate')
parser.add_argument('-bsize', type=int, dest="batch_size", default=16, help='batch size')
parser.add_argument('-epochs', type=int, dest="num_epochs", default=200, help='number of epochs')
# specific args for this training ------------------------------------------------------------------------------------------------------------------
parser.add_argument("--weighted", action = "store_true", dest="weighted", default = False, help ='apply weights to loss function')
parser.add_argument('-val_episodes', type=int, dest="val_episodes", default=10, help='run x validation episodes')
parser.add_argument('-val_frames', type=int, dest="val_frames", default=300, help='run for x frames')
parser.add_argument('-history', type=int, dest="history", default=1, help='number of previous frames to stack')
args = parser.parse_args()
print("settings:")
print("continute flag: {}\t save snaps :{}".format(args.continute_training,args.save_snaps))
print("name :{}\t\t learning rate :{}".format(args.name,args.learning_rate))
print("batch size :{}\t\t epochs :{}".format(args.batch_size,args.num_epochs))
print("val eps :{}\t val frame :{}".format(args.val_episodes,args.val_frames))
print("weighted :{}\t\t history :{}".format(args.weighted,args.history))
# loaders ------------------------------------------------------------------------------------------------------------------------------------------
train_loader = get_data_loader(batch_size=args.batch_size, train=True, history=args.history, validation_episodes=10)
val_loader   = get_data_loader(batch_size=args.batch_size, train=False, history=args.history, validation_episodes=10)

# setting up training device, agent, optimizer, weights --------------------------------------------------------------------------------------------
print("initializing agent, cuda, loss, optim")
device = torch.device('cuda')
agent = CBCAgent(device=device, history=args.history)
class_weights = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
if args.weighted:
    class_weights = torch.Tensor([0.50829944,   1.20620843,   1.        ,   0.54104019,    1.065929  ,   0.96403628,  84.07272727, 0.001, 0.001]).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
optimizer = optim.Adam(agent.net.parameters(), lr=args.learning_rate)

# if flag --cont is set, continute training from a previous snapshot ----------------------------------------------------------------------------------
if(args.continute_training):
    print("continue flag set")
    try:
        load_path = os.path.join(snapshot_dir,args.name)
        agent.net.load_state_dict(torch.load(load_path+"_model"))
        optimizer.load_state_dict(torch.load(load_path+"_optimizer"))
    except FileNotFoundError:
        print("snapshot file(s) not found")

# tensorboard --logdir=./tensorboard --port=6006
# alias tb="tensorboard --logdir=./tensorboard --port=6006 --reload_interval 5 & sleep 3 ; firefox 127.0.0.1:6006 & fg 1"
print("starting tensorboard")
writer = SummaryWriter(os.path.join(tensorboard_dir,args.name))

print("starting carla in server mode\n...")
my_env = os.environ.copy()
my_env["SDL_VIDEODRIVER"] = "offscreen"
FNULL = open(os.devnull, 'w')
subprocess.Popen(['server/./CarlaUE4.sh', '-benchmark', '-fps=20', '-carla-server', '-windowed', '-ResX=16', 'ResY=9'], stdout=FNULL, stderr=FNULL, env=my_env)
print("done")

print("training ...")
# lowest loss : save  best snapshots of the network
lowest_loss = 20

for epoch in range(1,args.num_epochs+1):    
    print("epoch {}/{}".format(epoch,args.num_epochs))
    loss_t = loss_v = 0
    
    # training episodes
    for idx, (labels, frames) in enumerate(train_loader) :
        print_over_same_line("training batch {}/{}".format(idx, len(train_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        agent.net.train()
        optimizer.zero_grad()
        pred  = agent.net(frames,'')
        loss = loss_fn(pred, labels.squeeze())
        loss.backward()
        loss_t += loss.item()
        optimizer.step()
        writer.add_scalar("iteration_training_loss", loss.item(), (epoch-1)*len(train_loader)+idx)

    # validation episodes
    for idx, (labels, frames) in enumerate(val_loader) :
        print_over_same_line("validation batch {}/{}".format(idx, len(val_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        agent.net.eval()
        pred = agent.predict(frames)
        loss = loss_fn(pred, labels.squeeze())
        loss_v += loss.item()
    
    # running 10 validation episodes with the current model
    acv, acp, aco, aiol, aior = evaluate_model(episodes=args.val_episodes, frames=args.val_frames, model=agent, device=device, 
                                               history=args.history, save_images=False, weather=1, vehicles=20, pedestians=40)
    writer.add_scalar("avg collision vehicle", sum(acv)/len(acv), epoch)
    writer.add_scalar("avg collision pedestrian", sum(acp)/len(acp), epoch)
    writer.add_scalar("avg collision other", sum(aco)/len(aco), epoch)
    writer.add_scalar("avg intersection otherlane", sum(aiol)/len(aiol), epoch)
    writer.add_scalar("avg intersection offroad", sum(aior)/len(aior), epoch)
    # saving current val loss for a shitty way of saving 'good' models
    current_val_loss = loss_v/len(val_loader)
    writer.add_scalar("epoch training loss", loss_t/len(train_loader), epoch)
    writer.add_scalar("epoch validation loss", current_val_loss, epoch)

    # saving model snapshots
    if args.save_snaps :
        save_path = os.path.join(snapshot_dir,args.name)
        torch.save(optimizer.state_dict(), save_path+"_optimizer")
        if  current_val_loss < lowest_loss or epoch%5==0 or current_val_loss <1:
            if current_val_loss < lowest_loss:
                lowest_loss = current_val_loss
            torch.save(agent.net.state_dict(), save_path+"_model_{}".format(epoch))
            print("saved snapshot at epoch {}".format(epoch))
            
writer.close()