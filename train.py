import sys
import os
import argparse
import numpy as np
import subprocess

import torch
import torch.optim as optim
from dataloader import get_data_loader
from tensorboardX import SummaryWriter
from agent.cagent import CBCAgent
from utils import print_over_same_line
from evaluate import evaluate_model
from dagger_train import dagger

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
parser.add_argument("--dagger", action = "store_true", dest="dagger", default = False, help ='perform dagger after each epoch')
parser.add_argument('-dagger_frames', type=int, dest="dagger_frames", default=1000, help='number of dagger frames to train')
parser.add_argument('-val_episodes', type=int, dest="val_episodes", default=10, help='run x validation episodes')
parser.add_argument('-val_frames', type=int, dest="val_frames", default=300, help='run for x frames')
parser.add_argument('-history', type=int, dest="history", default=1, help='number of previous frames to stack')
args = parser.parse_args()
print("settings:")
print("continute flag : {}\t save snaps : {}".format(args.continute_training,args.save_snaps))
print("name : {}\t\t learning rate : {}".format(args.name,args.learning_rate))
print("batch size : {}\t\t epochs : {}".format(args.batch_size,args.num_epochs))
print("val eps : {}\t val frame : {}".format(args.val_episodes,args.val_frames))
print("weighted : {}\t\t history : {}".format(args.weighted,args.history))
print("dagger : {} \t\t dagger frames : {}".format(args.dagger,args.dagger_frames))
# loaders ------------------------------------------------------------------------------------------------------------------------------------------
train_loader = get_data_loader(batch_size=args.batch_size, train=True, history=args.history, validation_episodes=10)
val_loader   = get_data_loader(batch_size=args.batch_size, train=False, history=args.history, validation_episodes=10)

# setting up training device, agent, optimizer, weights --------------------------------------------------------------------------------------------
print("initializing agent, cuda, loss, optim")
device = torch.device('cuda')
agent = CBCAgent(device=device, history=args.history, name='efficient-double')
class_weights = torch.Tensor([1, 1, 1])
if args.weighted:
    class_weights = torch.Tensor([  1.        ,   0.99284543, 319.17272727]).to(device)
classification_loss = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
regression_loss = torch.nn.MSELoss(reduction='sum')
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
    reg_loss_t = reg_loss_v = 0
    cls_loss_t = cls_loss_v = 0
    
    # training episodes
    for idx, (steer, labels, frames) in enumerate(train_loader) :
        print_over_same_line("training batch {}/{}".format(idx, len(train_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        steer = steer.to(device)
        agent.net.train()
        optimizer.zero_grad()
        pred_cls, pred_reg  = agent.net(frames)
        print (labels.squeeze(1).shape)
        print (steer.squeeze().shape)
        loss_cls = classification_loss(pred_cls, labels.squeeze(1))
        loss_reg = regression_loss(pred_reg, steer.squeeze())
        loss_cls.backward(retain_graph=True)
        loss_reg.backward()
        reg_loss_t += loss_reg.item()
        cls_loss_t += loss_cls.item()
        optimizer.step()
        writer.add_scalar("iteration/classification", loss_cls.item(), (epoch-1)*len(train_loader)+idx)
        writer.add_scalar("iteration/regression", loss_reg.item(), (epoch-1)*len(train_loader)+idx)
        break

    if args.dagger:
        reg_loss_dagger, cls_loss_dagger = dagger(frames=args.dagger_frames, model=agent, device=device, optimizer=optimizer, closs=classification_loss,
                                                  rloss=regression_loss, history=args.history, weather=1, vehicles=30, pedestians=30)
        writer.add_scalar("training/dagger_regression", reg_loss_dagger/args.dagger_frames, epoch)
        writer.add_scalar("training/dagger_classification", cls_loss_dagger/args.dagger_frames, epoch)

    # validation episodes
    for idx, (steer, labels, frames) in enumerate(val_loader) :
        print_over_same_line("validation batch {}/{}".format(idx, len(val_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        steer = steer.to(device)
        agent.net.eval()
        pred_cls, pred_reg  = agent.predict(frames)
        loss_cls = classification_loss(pred_cls, labels.squeeze(1))
        loss_reg = regression_loss(pred_reg, steer.squeeze())
        reg_loss_v += loss_reg.item()
        cls_loss_v += loss_cls.item()
        break
    
    
    # running 10 validation episodes with the current model
    acv, acp, aco, aiol, aior = evaluate_model(episodes=args.val_episodes, frames=args.val_frames, model=agent, device=device, 
                                               history=args.history, save_images=False, weather=1, vehicles=30, pedestians=30)
    writer.add_scalar("carla/vehicle_collision", sum(acv)/len(acv), epoch)
    writer.add_scalar("carla/pedestrian_collision", sum(acp)/len(acp), epoch)
    writer.add_scalar("carla/other_collision", sum(aco)/len(aco), epoch)
    writer.add_scalar("carla/otherlane_intersection", sum(aiol)/len(aiol), epoch)
    writer.add_scalar("carla/offroad_intersection", sum(aior)/len(aior), epoch)
    # saving current val loss for a shitty way of saving 'good' models
    current_val_loss = (reg_loss_v + cls_loss_v)/len(val_loader)
    writer.add_scalar("training/regression", reg_loss_t/len(train_loader), epoch)
    writer.add_scalar("training/classification", cls_loss_t/len(train_loader), epoch)
    writer.add_scalar("validation/regression", reg_loss_v, epoch)
    writer.add_scalar("validation/classification", cls_loss_v, epoch)

    # saving model snapshots
    if args.save_snaps :
        save_path = os.path.join(snapshot_dir,args.name)
        torch.save(optimizer.state_dict(), save_path+"_optimizer")
        if  current_val_loss < lowest_loss or epoch%5==0:
            if current_val_loss < lowest_loss:
                lowest_loss = current_val_loss
            agent.save(save_path+"_model_{}".format(epoch))
            print("saved snapshot at epoch {}".format(epoch))
            
writer.close()