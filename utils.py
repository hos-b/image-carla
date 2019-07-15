import numpy as np
import sys
import shutil
import torch

def print_over_same_line(text):
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    empty_space = max(0, terminal_width - len(text))
    sys.stdout.write('\r' + text + empty_space * ' ')
    sys.stdout.flush()

# function for the double network
def action_to_label_double(action):
    """
    converts action arrays [steering, throttle, brake] to
    labels. 0 : throttle, 1 : brake, 2: no-op
    """
    # bizarre cases
    if action[3] != 0 or action[4]!= 0 :
        print ("bizarre expert action! handbrake {}, reverse {}".format(action[3], action[4]))
    # normal cases
    steer = action[0] if np.abs(action[0]) > 5e-4 else 0
    throttle = action[1]>0
    brake = action[2]>0

    if throttle :
        return 0, steer
    elif brake :
        return 1, steer
    else :
        return 2, steer
    # throttle = 0
    # brake = 1
    # no-op = 2

def label_to_action_dobule(label, reg) :
    """
    converting doublenet's output to actions
    output : [steering, throttle, brake]
    """
    action = [reg, 0.0, 0.0]
    if label == 0 :
        action = [reg, 1.0, 0.0]
    elif label==1 :
        action = [reg, 0.0, 1.0]
    else :
        action = [reg, 0.0, 0.0]
    return action

def compare_controls(expert, agent, threshold) :
    """
    compare the expert controls with an agent
    returns True if they vary enough
    """
    # steering in opposite directions
    if agent[0]*expert[0] < 0:
        return True
    if np.abs(agent[0]-expert[0])>=threshold :
        return True
    # wrong steering
    if (expert[0]==0 and agent[0]!=0) or (expert[0]!=0 and agent[0]==0):
        return True
    # wrong throttle
    if (expert[1]==0 and agent[1]!=0) or (expert[1]!=0 and agent[1]==0):
        return True
    # wrong brake
    if (expert[2]==0 and agent[2]!=0) or (expert[2]!=0 and agent[2]==0):
        return True
    return False

@torch.no_grad()
def batch_accuracy(agent_cls, agent_reg, expert_cls, expert_reg):
    batch_size = agent_cls.shape[0]
    pred_cls = torch.argmax(agent_cls, dim=1).detach()
    hits = (torch.sum(torch.eq(pred_cls,agent_cls))).item()
    cls_accuracy = hits/batch_size

    pred_reg = agent_reg.clone().detach() 
    reg_accuracy = -(1/4)*(torch.pow((pred_reg-expert_reg),2)) + 1
    reg_accuracy = torch.sum(reg_accuracy).item()
    reg_accuracy = reg_accuracy/batch_size

    return cls_accuracy, reg_accuracy
