import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from dataloader import get_data_loader
from tensorboardX import SummaryWriter
from agent.cagent import CBCAgent

snapshot_dir = "./snaps"
tensorboard_dir="./tensorboard"

# arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--cont', '-c', action = "store_true", dest="continute_training", default = False, help ='continue training')
parser.add_argument('--snap', '-s', action = "store_true", dest="save_snaps", default = False, help='save snapshots every 5 epochs')
parser.add_argument('-name', type=str, dest="name", default="debug", help='name of the run')
parser.add_argument('-lr', type=float, dest="learning_rate", default=5e-4, help='learning rate')
parser.add_argument('-bsize', type=int, dest="batch_size", default=16, help='batch size')
parser.add_argument('-epochs', type=int, dest="num_epochs", default=2000, help='number of epochs')
parser.add_argument('-val', type=int, dest="val_epoch", default=5, help='run an episode every x epochs')
# specific args for this training
# parser.add_argument("--weighted", action = "store_true", dest="weighted", default = False, help ='apply weights to loss function')
parser.add_argument('-history', type=int, dest="history", default=1, help='number of previous frames to stack')
args = parser.parse_args()

print("settings :\ncontinute flag: {}\t save snaps: {}".format(args.continute_training,args.save_snaps))
print("name : {}\t\t learning rate {}".format(args.name,args.learning_rate))
print("batch size: {}\t\t epochs :{}".format(args.batch_size,args.num_epochs))
print("validate every : {}\t history :{}".format(args.val_epoch,args.history))
# loaders
train_loader = get_data_loader(batch_size=args.batch_size, train=True, history=args.history)
val_loader   = get_data_loader(batch_size=args.batch_size, train=False, history=args.history)

# setting up training device, agent, optimizer
print("initializing agent, cuda, loss, optim")
device = torch.device('cpu')
agent = CBCAgent(device=device, history=args.history)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(agent.net.parameters(), lr=args.learning_rate)

# if flag --c is set, continute training from a previous snapshot
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

print("training ...")
for epoch in range(1,args.num_epochs+1):    
    print("epoch {}/{}".format(epoch,args.num_epochs))
    loss_t = loss_v = 0
    
    for idx, (labels, frames) in enumerate(train_loader) :
        labels.to(device)
        frames.to(device)
        agent.net.train()
        optimizer.zero_grad()
        pred  = agent.net(frames,'')
        loss = loss_fn(pred, labels.squeeze())
        loss.backward()
        loss_t += loss.item()
        optimizer.step()

    for idx, (labels, frames) in enumerate(val_loader) :
        labels.to(device)
        frames.to(device)
        agent.net.eval()
        pred = agent.predict(frames,'')
        loss = loss_fn(pred, labels.squeeze())
        loss_v += loss.item()

    writer.add_scalar("epoch_training_loss", loss_t/len(train_loader), epoch)
    writer.add_scalar("epoch_validation_loss", loss_v/len(val_loader), epoch)
    loss_e_t = loss_e_v = 0

    if args.save_snaps :
        save_path = os.path.join(snapshot_dir,args.name)
        torch.save(agent.net.state_dict(), save_path+"_model")
        torch.save(optimizer.state_dict(), save_path+"_optimizer")
    writer.close()