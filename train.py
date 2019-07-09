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

STATUS_TRAINING = 0
STATUS_RECORDING_DAGGER = 1
STATUS_TRAINING_DAGGER = 2
STATUS_VALIDATING = 3
STATUS_SIMULATING = 4

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
print("name : {}\t learning rate : {}".format(args.name,args.learning_rate))
print("batch size : {}\t\t epochs : {}".format(args.batch_size,args.num_epochs))
print("val eps : {}\t\t val frame : {}".format(args.val_episodes,args.val_frames))
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
# if flag --cont is set, continute training from a previous snapshot -------------------------------------------------------------------------------
if(args.continute_training):
    print("continue flag set")
    try:
        load_path = os.path.join(snapshot_dir,args.name)
        agent.net.load_state_dict(torch.load(load_path+"_model"))
        optimizer.load_state_dict(torch.load(load_path+"_optimizer"))
    except FileNotFoundError:
        print("snapshot file(s) not found")
# starting tensorboard and carla in server mode -------------------- -------------------------------------------------------------------------------
print("starting tensorboard")
writer = SummaryWriter(os.path.join(tensorboard_dir,args.name))
print("starting carla in server mode\n...")
my_env = os.environ.copy()
my_env["SDL_VIDEODRIVER"] = "offscreen"
FNULL = open(os.devnull, 'w')
subprocess.Popen(['server/./CarlaUE4.sh', '-benchmark', '-fps=20', '-carla-server', '-windowed', '-ResX=16', 'ResY=9'], stdout=FNULL, stderr=FNULL, env=my_env)
print("done")
#train.py --snap -name=dnet_h3w_svdag -bsize=16 -val_episodes=10 -val_frames=400 -history=3 --weighted --dagger -dagger_frames=150
print("training ...")
# lowest loss : save  best snapshots of the network
lowest_loss = 20
# dagger episodes ran so far
dagger_episode_index = 0
for epoch in range(1,args.num_epochs+1):    
    print("epoch {}/{}".format(epoch,args.num_epochs))
    reg_loss_t = reg_loss_v = reg_loss_d = 0
    cls_loss_t = cls_loss_v = cls_loss_d = 0
    writer.add_scalar("status", STATUS_TRAINING, epoch+STATUS_TRAINING)
    # training episodes ----------------------------------------------------------------------------------------------------------------------------
    for idx, (steer, labels, frames) in enumerate(train_loader) :
        print_over_same_line("training batch {}/{}".format(idx, len(train_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        steer = steer.to(device)
        agent.net.train()
        optimizer.zero_grad()
        pred_cls, pred_reg  = agent.net(frames)
        loss_cls = classification_loss(pred_cls, labels.squeeze(1))
        loss_reg = regression_loss(pred_reg, steer)
        loss_cls.backward(retain_graph=True)
        loss_reg.backward()
        reg_loss_t += loss_reg.item()
        cls_loss_t += loss_cls.item()
        optimizer.step()
        writer.add_scalar("iteration/trn_classification", loss_cls.item(), (epoch-1)*len(train_loader)+idx)
        writer.add_scalar("iteration/trn_regression", loss_reg.item(), (epoch-1)*len(train_loader)+idx)
    writer.add_scalar("training/regression", reg_loss_t/len(train_loader), epoch)
    writer.add_scalar("training/classification", cls_loss_t/len(train_loader), epoch)
    # dagger episodes ------------------------------------------------------------------------------------------------------------------------------
    if args.dagger:
        writer.add_scalar("status", STATUS_RECORDING_DAGGER, epoch+STATUS_RECORDING_DAGGER)
        next_loc = 0
        dg_episodes = dagger(frames=args.dagger_frames, model=agent, device=device, history=args.history, weather=1, vehicles=30, pedestians=30, 
                            DG_next_location=next_loc, DG_next_episode=dagger_episode_index, DG_threshold=0.15)
        dagger_episode_index += dg_episodes
        # dagger loader
        writer.add_scalar("status", STATUS_TRAINING_DAGGER, epoch+STATUS_TRAINING_DAGGER)
        daggr_loader = get_data_loader(batch_size=args.batch_size, train=False, history=args.history, dagger=True)
        for idx, (steer, labels, frames) in enumerate(daggr_loader) :
            print_over_same_line("dagger batch {}/{}".format(idx, len(daggr_loader)))
            labels = labels.to(device)
            frames = frames.to(device)
            steer = steer.to(device)
            agent.net.train()
            optimizer.zero_grad()
            pred_cls, pred_reg  = agent.net(frames)
            loss_cls = classification_loss(pred_cls, labels.squeeze(1))
            loss_reg = regression_loss(pred_reg, steer)
            loss_cls.backward(retain_graph=True)
            loss_reg.backward()
            reg_loss_d += loss_reg.item()
            cls_loss_d += loss_cls.item()
            optimizer.step()
            writer.add_scalar("iteration/dgr_classification", loss_cls.item(), (epoch-1)*len(daggr_loader)+idx)
            writer.add_scalar("iteration/dgr_regression", loss_reg.item(), (epoch-1)*len(daggr_loader)+idx)
        writer.add_scalar("dagger/dagger_episode_count", dg_episodes, epoch)
        writer.add_scalar("dagger/dagger_regression", reg_loss_d/args.dagger_frames, epoch)
        writer.add_scalar("dagger/dagger_classification", cls_loss_d/args.dagger_frames, epoch)

    writer.add_scalar("status", STATUS_VALIDATING, epoch+STATUS_VALIDATING)
    # validation episodes --------------------------------------------------------------------------------------------------------------------------
    for idx, (steer, labels, frames) in enumerate(val_loader) :
        print_over_same_line("validation batch {}/{}".format(idx, len(val_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        steer = steer.to(device)
        agent.net.eval()
        pred_cls, pred_reg  = agent.predict(frames)
        loss_cls = classification_loss(pred_cls, labels.squeeze(1))
        loss_reg = regression_loss(pred_reg, steer)
        reg_loss_v += loss_reg.item()
        cls_loss_v += loss_cls.item()
    # saving current val loss for a shitty way of saving 'good' models
    current_val_loss = (reg_loss_v + cls_loss_v)/len(val_loader)
    writer.add_scalar("validation/regression", reg_loss_v/len(val_loader), epoch)
    writer.add_scalar("validation/classification", cls_loss_v/len(val_loader), epoch)
    
    writer.add_scalar("status", STATUS_SIMULATING, epoch+STATUS_SIMULATING)
    # simulation episodes --------------------------------------------------------------------------------------------------------------------------
    acv, acp, aco, aiol, aior = evaluate_model(episodes=args.val_episodes, frames=args.val_frames, model=agent, device=device, 
                                               history=args.history, save_images=False, weather=1, vehicles=30, pedestians=30)
    writer.add_scalar("carla/vehicle_collision", sum(acv)/len(acv), epoch)
    writer.add_scalar("carla/pedestrian_collision", sum(acp)/len(acp), epoch)
    writer.add_scalar("carla/other_collision", sum(aco)/len(aco), epoch)
    writer.add_scalar("carla/otherlane_intersection", sum(aiol)/len(aiol), epoch)
    writer.add_scalar("carla/offroad_intersection", sum(aior)/len(aior), epoch)

    # saving model snapshots
    if args.save_snaps :
        save_path = os.path.join(snapshot_dir,args.name)
        torch.save(optimizer.state_dict(), save_path+"_optimizer")
        if current_val_loss < lowest_loss or epoch%2==0:
            if current_val_loss < lowest_loss:
                lowest_loss = current_val_loss
            agent.save(save_path+"_model_{}".format(epoch))
            print("saved snapshot at epoch {}".format(epoch))            
writer.close()