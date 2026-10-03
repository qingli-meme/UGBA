
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import imp
import time
import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr


# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=40,
                    help="number of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="prune",
                    choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree','rpi','rpi_gap','rpi_sensitivity','fixed'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--fixed_attach_file', type=str, default='',
                    help='Text file containing explicit node IDs for Phase-2 utility validation')
# RPI selection setting -- only used when --selection_method rpi
parser.add_argument('--rpi_surrogate_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage'],
                    help='Clean surrogate used only for offline RPI scoring')
parser.add_argument('--rpi_mc_samples', type=int, default=4,
                    help='Number of topology samples for RPI; first sample is the clean graph')
parser.add_argument('--rpi_edge_drop', type=float, default=0.10,
                    help='Undirected clean-edge drop probability for robust RPI samples')
parser.add_argument('--rpi_kappa', type=float, default=0.0,
                    help='Required target decision margin used in the minimum propagation cost')
parser.add_argument('--rpi_eps', type=float, default=1e-12,
                    help='Numerical stabilizer in RPI denominator')
parser.add_argument('--rpi_max_candidates', type=int, default=0,
                    help='Optional cap for large graphs; 0 scores all eligible candidates')
parser.add_argument('--rpi_no_class_balance', action='store_true', default=False,
                    help='Disable class-balanced final Bottom-K selection')
parser.add_argument('--rpi_score_dir', type=str, default='./rpi_scores',
                    help='Directory used to save per-node RPI diagnostics')
parser.add_argument('--rpi_verbose_every', type=int, default=100,
                    help='Print RPI scoring progress every N candidates; <=0 disables progress prints')
parser.add_argument('--attack_method', type=str, default='ugba',
                    choices=['ugba','fmrb'],
                    help='Backdoor implementation: original UGBA or focused multi-relay backdoor')
parser.add_argument('--relay_method', type=str, default='pfr',
                    choices=['pfr','gain','low_degree','random'],
                    help='Relay selection rule used by FMRB')
parser.add_argument('--relay_count', type=int, default=-1,
                    help='Number of relays/payloads per victim; <=0 reuses --trigger_size')
parser.add_argument('--relay_eps', type=float, default=1e-12,
                    help='Numerical epsilon in relay scoring')
parser.add_argument('--trigger_mode', type=str, default='ugba',
                    choices=['ugba', 'passband'],
                    help='Original UGBA trigger or Passband-Matched Graph Backdoor')
parser.add_argument('--pgb_m_neighbors', type=int, default=1,
                    help='Existing neighbors each PGB trigger node also connects to')
parser.add_argument('--pgb_pca_r', type=int, default=32)
parser.add_argument('--pgb_lambda_manifold', type=float, default=1.0)
parser.add_argument('--pgb_lambda_delta', type=float, default=1e-3)
parser.add_argument('--pgb_delta_budget', type=float, default=1.0)
parser.add_argument('--pgb_init_delta', type=float, default=0.5)
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


# In[9]:

from sklearn_extra import cluster
from models.backdoor import Backdoor
from models.distributed_backdoor import DistributedRelayBackdoor
from models.passband_backdoor import PassbandBackdoor
from models.construct import model_construct
import heuristic_selection as hs
import rpi_selection as rpis

# from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("#Attach Nodes:{}".format(size))
assert size>0, 'The number of selected trigger nodes must be larger than 0!'
# here is randomly select poison nodes from unlabeled nodes
if(args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    if(args.dataset == 'Pubmed'):
        idx_attach = hs.cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    else:
        idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method in ['rpi', 'rpi_gap', 'rpi_sensitivity']):
    idx_attach = rpis.robust_propagation_selection(
        args=args,
        data=data,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_clean_test=idx_clean_test,
        unlabeled_idx=unlabeled_idx,
        train_edge_index=train_edge_index,
        size=size,
        device=device,
    )
elif(args.selection_method == 'fixed'):
    if not args.fixed_attach_file:
        raise ValueError('--fixed_attach_file is required for selection_method=fixed')
    fixed_nodes = np.loadtxt(args.fixed_attach_file, dtype=int, ndmin=1)
    if len(fixed_nodes) != size:
        raise ValueError('fixed attach file contains {} nodes, expected {}'.format(
            len(fixed_nodes), size))
    eligible_nodes = set(unlabeled_idx.detach().cpu().tolist())
    invalid_nodes = [int(node) for node in fixed_nodes if int(node) not in eligible_nodes]
    if invalid_nodes:
        raise ValueError('fixed attach file contains ineligible nodes: {}'.format(invalid_nodes))
    idx_attach = torch.as_tensor(fixed_nodes, dtype=torch.long, device=device)
print("idx_attach: {}".format(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print(unlabeled_idx)
# In[10]:
# train trigger generator 
if args.trigger_mode == 'passband':
    if args.attack_method != 'ugba':
        raise ValueError('--trigger_mode passband cannot be combined with --attack_method fmrb')
    model = PassbandBackdoor(args,device)
elif args.attack_method == 'fmrb':
    model = DistributedRelayBackdoor(args,device)
else:
    model = Backdoor(args,device)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()

if(args.defense_mode == 'prune'):
    poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'isolate'):
    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))


models = ['GCN','GAT', 'GraphSage']
total_overall_asr = 0
total_overall_ca = 0
for test_model in models:
    args.test_model = test_model
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=5)
    # seeds = [args.seed]
    overall_asr = 0
    overall_ca = 0
    for seed in seeds:
        args.seed = seed
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print(args)
        #%%
        test_model = model_construct(args,args.test_model,data,device).to(device) 
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)

        output = test_model(poison_x,poison_edge_index,poison_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        #%%
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


        if(args.evaluate_mode == '1by1'):
            from torch_geometric.utils  import k_hop_subgraph
            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
            asr = 0
            flip_asr = 0
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            for i, idx in enumerate(idx_atk):
                idx=int(idx)
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                with torch.no_grad():
                    # inject trigger on attack test nodes (idx_atk)'''
                    if args.attack_method == 'fmrb':
                        eligible_relay_mask = (sub_induct_nodeset < data.x.shape[0]).to(device)
                        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                            relabeled_node_idx,
                            poison_x[sub_induct_nodeset],
                            sub_induct_edge_index,
                            sub_induct_edge_weights,
                            device,
                            eligible_relay_mask=eligible_relay_mask,
                        )
                    else:
                        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                            relabeled_node_idx,
                            poison_x[sub_induct_nodeset],
                            sub_induct_edge_index,
                            sub_induct_edge_weights,
                            device,
                        )
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                    # # do pruning in test datas'''
                    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,False)
                    # attack evaluation
                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                    asr += train_attach_rate
                    if(data.y[idx] != args.target_class):
                        flip_asr += train_attach_rate
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                    output = output.cpu()
            asr = asr/(idx_atk.shape[0])
            flip_asr = flip_asr/(flip_idx_atk.shape[0])
            print("Overall ASR: {:.4f}".format(asr))
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
        elif(args.evaluate_mode == 'overall'):
            # %% inject trigger on attack test nodes (idx_atk)'''
            if args.attack_method == 'fmrb':
                eligible_relay_mask = (
                    torch.arange(poison_x.shape[0], device=device) < data.x.shape[0]
                )
                induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                    idx_atk,
                    poison_x,
                    induct_edge_index,
                    induct_edge_weights,
                    device,
                    eligible_relay_mask=eligible_relay_mask,
                )
            else:
                induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                    idx_atk,
                    poison_x,
                    induct_edge_index,
                    induct_edge_weights,
                    device,
                )
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            # do pruning in test datas'''
            if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
            # attack evaluation
            output = test_model(induct_x,induct_edge_index,induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            asr = train_attach_rate
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
            ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
            print("CA: {:.4f}".format(ca))

            induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
            output = output.cpu()

        overall_asr += asr
        overall_ca += clean_acc

        test_model = test_model.cpu()
        
    overall_asr = overall_asr/len(seeds)
    overall_ca = overall_ca/len(seeds)
    print("Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.test_model, args.seed))
    print("Overall Clean Accuracy: {:.4f}".format(overall_ca))

    total_overall_asr += overall_asr
    total_overall_ca += overall_ca
    test_model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
total_overall_asr = total_overall_asr/len(models)
total_overall_ca = total_overall_ca/len(models)
print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))
