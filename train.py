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
from utils import batch_accuracy
import random

STATUS_TRAINING = 0
STATUS_RECORDING_DAGGER = 1
STATUS_TRAINING_DAGGER = 2
STATUS_VALIDATING = 3
STATUS_SIMULATING = 4

# CUDA_VISIBLE_DEVICES=0 python3 train.py --snap -name=dnet_h3w_16th_HD -bsize=32 -val_episodes=10 -val_frames=500 -history=3 --weighted --dagger -dagger_frames=1600 -carla_port=200
# latest counts : [44556. 25988.     0.]
# latest weights: [0.5832660023341413,  1, 0.1]
snapshot_dir = "./snaps"
tensorboard_dir="./tensorboard"
# arg parse ----------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cont', '-c', action = "store_true", dest="continute_training", default = False, help ='continue training')
parser.add_argument('-se', type=int, dest="start_epoch", default=1, help='continue from epoch x')
parser.add_argument('--snap', '-s', action = "store_true", dest="save_snaps", default = True, help='save snapshots every 5 epochs')
parser.add_argument('-name', type=str, dest="name", default="debug", help='name of the run')
parser.add_argument('-lr', type=float, dest="learning_rate", default=5e-4, help='learning rate')
parser.add_argument('-bsize', type=int, dest="batch_size", default=16, help='batch size')
parser.add_argument('-epochs', type=int, dest="num_epochs", default=200, help='number of epochs')
# specific args for this training ------------------------------------------------------------------------------------------------------------------
parser.add_argument("--weighted", action = "store_true", dest="weighted", default = False, help ='apply weights to loss function')
parser.add_argument("--dagger", action = "store_true", dest="dagger", default = False, help ='perform dagger after each epoch')
parser.add_argument("--overwrite", action = "store_true", dest="overwrite_dagger", default = False, help ='overwrite the old dagger dataset')
parser.add_argument('-dagger_frames', type=int, dest="dagger_frames", default=1000, help='number of dagger frames to train')
parser.add_argument('-carla_port', type=int, dest="carla_port", default=2000, help='port to connect to for carla')
parser.add_argument('-val_episodes', type=int, dest="val_episodes", default=10, help='run x validation episodes')
parser.add_argument('-val_frames', type=int, dest="val_frames", default=300, help='run for x frames')
parser.add_argument('-history', type=int, dest="history", default=1, help='number of previous frames to stack')
args = parser.parse_args()
print("settings:")
print("continute flag : {}\t save snaps : {}".format(args.continute_training,args.save_snaps))
print("starting epoch : {}\t carla port : {}".format(args.start_epoch,args.carla_port))
print("name : {}\t learning rate : {}".format(args.name,args.learning_rate))
print("batch size : {}\t\t epochs : {}".format(args.batch_size,args.num_epochs))
print("val eps : {}\t\t val frame : {}".format(args.val_episodes,args.val_frames))
print("weighted : {}\t\t history : {}".format(args.weighted,args.history))
print("dagger : {} \t\t dagger frames : {}".format(args.dagger,args.dagger_frames))
# loaders ------------------------------------------------------------------------------------------------------------------------------------------
train_loader = get_data_loader(batch_size=args.batch_size, train=True, history=args.history, validation_episodes=10)
val_loader   = get_data_loader(batch_size=args.batch_size, train=False, history=args.history, validation_episodes=10)
# setting up training device, agent, optimizer, weights --------------------------------------------------------------------------------------------
print("initializing agent, cuda, loss, optimizer")
device = torch.device('cuda')
agent = CBCAgent(device=device, history=args.history, name='efficient-double-large')
class_weights = torch.Tensor([1, 1, 1])
if args.weighted:
    class_weights = torch.Tensor([0.5832660023341413,  1, 0.1]).to(device)
classification_loss = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
regression_loss = torch.nn.MSELoss(reduction='sum')
# setting up optimizer + special parameters
optimizer = optim.Adam(agent.net.parameters(), lr=args.learning_rate)
l2_weight = torch.nn.Parameter(torch.Tensor([-2.0]).to(device))
ce_weight = torch.nn.Parameter(torch.Tensor([0]).to(device))
optimizer.add_param_group({"params": ce_weight})
optimizer.add_param_group({"params": l2_weight})
# if flag --cont is set, continute training from a previous snapshot -------------------------------------------------------------------------------
if(args.continute_training):
    print("continue flag set")
    try:
        load_path = os.path.join(snapshot_dir,args.name)
        agent.net.load_state_dict(torch.load(load_path +"_model_{}".format(args.start_epoch-1)))
        optimizer.load_state_dict(torch.load(load_path+"_optimizer"))
    except FileNotFoundError:
        print("snapshot file(s) not found")
# starting tensorboard and carla in server mode ----------------------------------------------------------------------------------------------------
print("starting tensorboard")
writer = SummaryWriter(os.path.join(tensorboard_dir,args.name))
print("starting carla in server mode\n...")
my_env = os.environ.copy()
my_env["DISPLAY"] = ""
FNULL = open(os.devnull, 'w')
subprocess.Popen(['server/./CarlaUE4.sh', '-benchmark', '-fps=20', '-carla-server', '-windowed', '/Game/Maps/Town02', '-opengl',
                 '-ResX=16', '-ResY=9','-world-port={}'.format(args.carla_port)], stdout=FNULL, stderr=FNULL, env=my_env)
print("done")
print("training ...")
# lowest loss : save  best snapshots of the network
lowest_loss = 20
# dagger init
dagger_episode_index = 0
dagger_next_loc = 0
dagger_loss = torch.nn.CrossEntropyLoss().to(device)
if args.overwrite_dagger :
    print("removing old dagger dataset")
    os.system("rm -f /tmp/dagger_dataset.hdf5")
# start training
for epoch in range(args.start_epoch, args.num_epochs+1):   
    print("epoch {}/{}".format(epoch,args.num_epochs))
    reg_loss_t = reg_loss_v = reg_loss_d = 0
    cls_loss_t = cls_loss_v = cls_loss_d = 0
    training_accuracy_cls = training_accuracy_reg = 0
    validation_accuracy_cls = validation_accuracy_reg = 0
    dagger_accuracy_cls = dagger_accuracy_reg = 0
    writer.add_scalar("status", STATUS_TRAINING, epoch+STATUS_TRAINING)
    agent.net.train()
    # training episodes ----------------------------------------------------------------------------------------------------------------------------
    for idx, (steer, labels, frames) in enumerate(train_loader) :
        print_over_same_line("training batch {}/{}".format(idx+1, len(train_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        steer = steer.to(device)
        optimizer.zero_grad()
        pred_cls, pred_reg  = agent.net(frames)
        loss_cls = classification_loss(pred_cls, labels.squeeze(1))
        loss_reg = regression_loss(pred_reg, steer)
        (loss_cls*torch.exp(-ce_weight)+ ce_weight + loss_reg*torch.exp(-l2_weight)+l2_weight).backward()
        reg_loss_t += loss_reg.item()
        cls_loss_t += loss_cls.item()
        optimizer.step()
        cls_accuracy, reg_accuracy = batch_accuracy(pred_cls,pred_reg,labels.squeeze(1),steer)
        training_accuracy_cls +=cls_accuracy
        training_accuracy_reg +=reg_accuracy
        writer.add_scalar("iteration/trn_classification", loss_cls.item(), (epoch-1)*len(train_loader)+idx)
        writer.add_scalar("iteration/trn_regression", loss_reg.item(), (epoch-1)*len(train_loader)+idx)
    training_accuracy_cls /= len(train_loader)
    training_accuracy_reg /= len(train_loader)
    writer.add_scalar("training/regression", reg_loss_t/len(train_loader), epoch)
    writer.add_scalar("training/classification", cls_loss_t/len(train_loader), epoch)
    writer.add_scalar("training/cls_accuracy", training_accuracy_cls, epoch)
    writer.add_scalar("training/reg_accuracy", training_accuracy_reg, epoch)
    # dagger episodes ------------------------------------------------------------------------------------------------------------------------------
    if args.dagger:
        writer.add_scalar("status", STATUS_RECORDING_DAGGER, epoch+STATUS_RECORDING_DAGGER)
        dg_episodes, skipped_frames = dagger(frames=args.dagger_frames, model=agent, device=device, history=args.history, weather=1, carla_port=args.carla_port,
                                vehicles=50, pedestians=30, DG_next_location=dagger_next_loc, DG_next_episode=dagger_episode_index, DG_threshold=0.08)
        dagger_episode_index +=dg_episodes
        #TODO figure out a good system 
        dagger_next_loc = (dagger_next_loc+random.randint(6, 10))%140
        # dagger loader
        writer.add_scalar("status", STATUS_TRAINING_DAGGER, epoch+STATUS_TRAINING_DAGGER)
        daggr_loader = get_data_loader(batch_size=args.batch_size, train=True, history=args.history, dagger=True)
        agent.net.train()
        for idx, (steer, labels, frames) in enumerate(daggr_loader) :
            print_over_same_line("dagger batch {}/{}".format(idx+1, len(daggr_loader)))
            labels = labels.to(device)
            frames = frames.to(device)
            steer = steer.to(device)
            optimizer.zero_grad()
            pred_cls, pred_reg  = agent.net(frames)
            loss_cls = dagger_loss(pred_cls, labels.squeeze(1))
            loss_reg = regression_loss(pred_reg, steer)
            (loss_cls*torch.exp(-ce_weight)+ce_weight + loss_reg*torch.exp(-l2_weight)+l2_weight).backward()
            reg_loss_d += loss_reg.item()
            cls_loss_d += loss_cls.item()
            optimizer.step()
            cls_accuracy, reg_accuracy = batch_accuracy(pred_cls,pred_reg,labels.squeeze(1),steer)
            dagger_accuracy_cls +=cls_accuracy
            dagger_accuracy_reg +=reg_accuracy
            writer.add_scalar("iteration/dgr_classification", loss_cls.item(), (epoch-1)*len(daggr_loader)+idx)
            writer.add_scalar("iteration/dgr_regression", loss_reg.item(), (epoch-1)*len(daggr_loader)+idx)
        dagger_accuracy_cls /= len(daggr_loader)
        dagger_accuracy_reg /= len(daggr_loader)
        writer.add_scalar("dagger/dagger_episode_count", dg_episodes, epoch)
        writer.add_scalar("dagger/dagger_skipped_frames", skipped_frames, epoch)
        writer.add_scalar("dagger/dagger_regression", reg_loss_d/args.dagger_frames, epoch)
        writer.add_scalar("dagger/dagger_classification", cls_loss_d/args.dagger_frames, epoch)
        writer.add_scalar("dagger/cls_accuracy", dagger_accuracy_cls, epoch)
        writer.add_scalar("dagger/reg_accuracy", dagger_accuracy_reg, epoch)
    writer.add_scalar("status", STATUS_VALIDATING, epoch+STATUS_VALIDATING)
    print("evaluating")
    agent.net.eval()
    # validation episodes --------------------------------------------------------------------------------------------------------------------------
    for idx, (steer, labels, frames) in enumerate(val_loader) :
        print_over_same_line("validation batch {}/{}".format(idx+1, len(val_loader)))
        labels = labels.to(device)
        frames = frames.to(device)
        steer = steer.to(device)
        pred_cls, pred_reg  = agent.predict(frames)
        loss_cls = classification_loss(pred_cls, labels.squeeze(1))
        loss_reg = regression_loss(pred_reg, steer)
        reg_loss_v += loss_reg.item()
        cls_loss_v += loss_cls.item()
        cls_accuracy, reg_accuracy = batch_accuracy(pred_cls,pred_reg,labels.squeeze(1),steer)
        validation_accuracy_cls +=cls_accuracy
        validation_accuracy_reg +=reg_accuracy
    validation_accuracy_cls /= len(val_loader)
    validation_accuracy_reg /= len(val_loader)
    # saving current val loss as a shitty way of saving 'good' models
    current_val_loss = (reg_loss_v + cls_loss_v)/len(val_loader)
    if args.dagger :
        current_val_loss += (reg_loss_d + cls_loss_d)/len(daggr_loader) 
    writer.add_scalar("validation/regression", reg_loss_v/len(val_loader), epoch)
    writer.add_scalar("validation/classification", cls_loss_v/len(val_loader), epoch)
    writer.add_scalar("validation/cls_accuracy", validation_accuracy_cls, epoch)
    writer.add_scalar("validation/reg_accuracy", validation_accuracy_reg, epoch)

    writer.add_scalar("status", STATUS_SIMULATING, epoch+STATUS_SIMULATING)
    writer.add_scalar("training/l2_weight", torch.exp(-l2_weight).item(), epoch)
    writer.add_scalar("training/ce_weight", torch.exp(-ce_weight).item(), epoch)
    # simulation episodes --------------------------------------------------------------------------------------------------------------------------
    acv, acp, aco, aiol, aior, adt = evaluate_model(episodes=args.val_episodes, frames=args.val_frames, model=agent, device=device, carla_port=args.carla_port,
                                               history=args.history, save_images=False, weather=1, vehicles=50, pedestians=30)
    writer.add_scalar("carla/average_distance_traveled", sum(adt)/len(adt), epoch)   
    writer.add_scalar("carla/otherlane_intersection", sum(aiol)/len(aiol), epoch)
    writer.add_scalar("carla/offroad_intersection", sum(aior)/len(aior), epoch)
    writer.add_scalar("carla/other_collision", sum(aco)/len(aco), epoch)
    writer.add_scalar("carla/vehicle_collision", sum(acv)/len(acv), epoch)
    writer.add_scalar("carla/pedestrian_collision", sum(acp)/len(acp), epoch)
    
    
    average_fault =  sum(aiol)/len(aiol) +sum(aior)/len(aior)
    # saving model snapshots
    if args.save_snaps :
        save_path = os.path.join(snapshot_dir,args.name)
        torch.save(optimizer.state_dict(), save_path+"_optimizer")
        if current_val_loss < lowest_loss or epoch%2==0 or average_fault<10:
            if current_val_loss < lowest_loss :
                lowest_loss = current_val_loss
            agent.save(save_path+"_model_{}".format(epoch))
            print("saved snapshot at epoch {}".format(epoch))            
writer.close()