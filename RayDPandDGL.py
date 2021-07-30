
import ray
import sys
import argparse
import raydp
from raydp.torch import TorchEstimator
from raydp.utils import random_split
from raydp.spark.dataset import *
from raydp.spark import create_ml_dataset_from_spark
from typing import List, NoReturn, Optional
import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.utils import AverageMeterCollection
from ray.util.sgd.torch import TrainingOperator

import dgl
from dgl.data import RedditDataset
from dgl.nn.pytorch import GATConv
from torch.utils.data import DataLoader
from dgl.dataloading import NodeCollator

print("Current Path: " + os.getcwd())
torch.manual_seed(42)

def run(num_workers, use_gpu, num_epochs, lr, batch_size, n_hidden, n_layers,
        n_heads, fan_out, feat_drop, attn_drop, negative_slope,
        sampling_num_workers,num_cpus_per_worker):
    print("runing~~~~~~~~~~~~~")
    trainer = TorchTrainer(
        training_operator_cls=CustomTrainingOperator,
        num_workers=num_workers,
        use_gpu=use_gpu,
        #backend="nccl",
        num_cpus_per_worker= num_cpus_per_worker,
        config={
            "lr": lr,
            "batch_size": batch_size,
            "n_hidden": n_hidden,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "fan_out": fan_out,
            "feat_drop": feat_drop,
            "attn_drop": attn_drop,
            "negative_slope": negative_slope,
            "sampling_num_workers": sampling_num_workers
        })

    for i in range(num_epochs):
        print("training~~~~~~~~~~~")
        trainer.train()
    validation_results = trainer.validate()
    trainer.shutdown()
    print(validation_results)
    print("success!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RayDP with DGL')
    parser.add_argument('--app_name', type=str, default="RayDP with DGL for abfusion_v2(Full Graph Store)")
    parser.add_argument('--partition_num', type=int, default=90, help='partition number of rdd')
    parser.add_argument('--full_graph_path', type=str,
                        default="file:///mnt/disk1/dataset/gnn_dataset/GraphScope/reddit_edges_reverse.csv",
                        help='The edge file dir of the original full graph')
    parser.add_argument('--full_graph_feat_path', type=str,
                        default="file:///mnt/disk1/dataset/gnn_dataset/GraphScope/reddit_feat.csv",
                        help='The edge file dir of the original full graph')
    parser.add_argument('--num_shard', type=int, default=1,
                        help='num of shard to fetch the dataset stored in Ray ObjectStore')
    parser.add_argument('--write_batch_size', type=int, default=1024,
                        help='batch size when write data into ObjectStore')
    parser.add_argument('--executors_num', type=int, default=3)
    parser.add_argument('--per_executor_core', type=int, default=45)
    parser.add_argument('--per_executor_mem', type=str, default='20GB')
    parser.add_argument('--ray_cluster_passwd', type=str, default='5241590000000000')
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--num-cpus-per-worker", type=int, default=12)
    parser.add_argument("--use-gpu", type=bool, default=False)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--fan-out", type=str, default="10,25")
    parser.add_argument("--feat-drop", type=float, default=0.)
    parser.add_argument("--attn-drop", type=float, default=0.)
    parser.add_argument("--negative-slope", type=float, default=0.2)
    parser.add_argument(
        "--sampling-num-workers",
        type=int,
        default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="The address to use for ray")

    args = parser.parse_args()



    Edage_data_PATH = args.full_graph_path
    Node_data_PATH = args.full_graph_feat_path
    ray.init(address='auto', _redis_password=args.ray_cluster_passwd)
    # After initialize ray cluster, you can use the raydp api to get a spark session
    app_name = "RayDP with DGL for abfusion_v2(Full Graph Store)"
    num_executors = 3
    cores_per_executor = 40
    memory_per_executor = "50GB"
    spark = raydp.init_spark(app_name,
                             num_executors,
                             cores_per_executor,
                             memory_per_executor)

    # # schema for Node_features_data
    long_cols = list(range(0, 1))
    float_cols = list(range(1, 1 + 602))
    byte_cols1 = list(range(1 + 602, 604))
    byte_cols2 = list(range(1 + 603, 605))
    byte_cols3 = list(range(1 + 604, 606))
    byte_cols4 = list(range(1 + 605, 607))

    long_fields = [('node_id') for i in long_cols]
    float_fields = [('feat_%d') % i for i in float_cols]
    byte_fields1 = [('label') for i in byte_cols1]
    byte_fields2 = [('mask[train]') for i in byte_cols2]
    byte_fields3 = [('mask[val]') for i in byte_cols3]
    byte_fields4 = [('mask[test]') for i in byte_cols4]
    schema1 = (long_fields +
              float_fields +
              byte_fields1 +
              byte_fields2 +
              byte_fields3 +
              byte_fields4)

    schema2 = ["src_node_id","dst_node_id"]

    # print("schema type:",type(schema))
    # print("schema::::",schema)
    # # Here we just use a subset of the training data
    Node_features_data = spark.read.format("csv").option("header", "False") \
        .option("inferSchema", "true") \
        .load(Node_data_PATH) \
        .toDF(*schema1) \

    Edage_data = spark.read.format("csv").option("header", "False") \
        .option("inferSchema", "true") \
        .load(Edage_data_PATH) \
        .toDF(*schema2) \

    Node_features_MLdataset = create_ml_dataset_from_spark(Node_features_data, 3, 1024)  ##1097.3MB
    Edage_MLdataset = create_ml_dataset_from_spark(Edage_data, 3, 1024)  ##908.3MB

    # print("Edage_MLdataset的每列数据类型：：：\n",Edage_data.dtypes)
    # print("Edage_data的分区个数：：\n",Edage_data.rdd.getNumPartitions())
    # print("----------------------------")
    # #print("Node_features_data的每列数据类型：：：\n",Node_features_data.dtypes)
    # print("Node_features_data的分区个数：：：\n",Node_features_data.rdd.getNumPartitions())
    # print("features:::\n", schema1)
    # print("edge:::\n", schema2)

    #print("ray nodes:",ray.nodes())
    #Edage_data.show(10)
    Node_features_data.show(8)
    # print("-----------------------------")
    # #Node_features_MLdataset.show(3)
    tmp =Node_features_MLdataset.take(2)
    print(tmp)
    node_num = Node_features_data.count()

    ##-----------------

    # define the model class
    class GAT(nn.Module):
        def __init__(self, in_feats, n_hidden, n_classes, n_layers, n_heads,
                     activation, feat_drop, attn_drop, negative_slope, residual):
            super().__init__()

            self.n_layers = n_layers
            self.activation = activation
            self.n_hidden = n_hidden
            self.n_heads = n_heads
            self.n_classes = n_classes
            self.convs = nn.ModuleList()

            # input layer
            self.convs.append(
                GATConv((in_feats, in_feats), n_hidden, n_heads, feat_drop,
                        attn_drop, negative_slope, residual, self.activation))
            # hidden layer
            for _ in range(1, n_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.convs.append(
                    GATConv((n_hidden * n_heads, n_hidden * n_heads), n_hidden,
                            n_heads, feat_drop, attn_drop, negative_slope,
                            residual, self.activation))
            # output layer
            self.convs.append(
                GATConv((n_hidden * n_heads, n_hidden * n_heads), n_classes,
                        n_heads, feat_drop, attn_drop, negative_slope, residual,
                        None))

        def forward(self, blocks, x):
            h = x
            for i, (layer, block) in enumerate(zip(self.convs, blocks)):
                h_dst = h[:block.number_of_dst_nodes()]
                if i != len(self.convs) - 1:
                    h = layer(block, (h, h_dst)).flatten(1)
                    h = F.dropout(h, p=0.5, training=self.training)
                else:
                    h = layer(block, (h, h_dst))
            h = h.mean(1)
            return h.log_softmax(dim=-1)


    def compute_acc(pred, labels):
        """
        Compute the accuracy of prediction given the labels.
        """
        return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


    # util function to convert pandas series to tensor in a zero copy manner
    def series_to_tensor(ser):
        data = ser.to_numpy()
        dt_type = data.dtype
        if dt_type == np.dtype("float64"):
            return torch.DoubleTensor(data)
        elif dt_type == np.dtype("float32"):
            return torch.FloatTensor(data)
        elif dt_type == np.dtype("int64"):
            return torch.LongTensor(data)
        elif dt_type == np.dtype("int32"):
            return torch.IntTensor(data)
        elif dt_type == np.dtype("int8"):
            return torch.ByteTensor(data)
        elif dt_type == np.dtype("bool"):
            return torch.BoolTensor(data)
        else:
            raise Exception("unrecognized data type!")


    # util function to convert pandas to tensor
    def resolve_pandas_to_tensor(pd_dataframe, split_pandas):
        tensors = []
        print("pd_dataframe ::",pd_dataframe.iloc[:,:])
        print("pd_dataframe type in resolve_pandas_to_tensor ::", type(pd_dataframe)) #DataFrame
        # when deal with data which contains feats
        # we split the pandas into 3 parts, which still share the same memory space with pd_dataframe
        if split_pandas:
            pd_index = pd_dataframe.iloc[:, 0:1]
            pd_feat = pd_dataframe.iloc[:, 1:1 + 602]
            pd_label_masks = pd_dataframe.iloc[:,603:607]

            for col in pd_index.columns:
                tensors.append(series_to_tensor(pd_index[col]))

            # should treat the feats with special way cuz the feats is non-1-dimensional
            tensors.append(torch.Tensor(pd_feat.values))

            for col in pd_label_masks.columns:
                tensors.append(series_to_tensor(pd_label_masks[col]))
        # otherwise, there is no need for split
        else:
             #可以按列取数据，最后转置
             for col in pd_dataframe.columns:
                 tensors.append(series_to_tensor(pd_dataframe[col]))
        return tensors


    # ray data getter function
    def fetch_func(ml_dataset, re_batch_size=None,split_pandas=False):
        if re_batch_size:
            ml_dataset = ml_dataset.batch(re_batch_size)
        # each column represented by a tensor in pd_tensors
        pd_tensors = []
        ml_dataset = ml_dataset.shards()
        print("ml_dataset in fetch_func:::",ml_dataset)
        for ml_tmp in ml_dataset:
            for batch_pd_dataframe in ml_tmp:
                pd_tensors.append(resolve_pandas_to_tensor(batch_pd_dataframe, split_pandas))
        return pd_tensors


    class CustomTrainingOperator(TrainingOperator):
        def setup(self, config):
            # load reddit data
            ##-------------
            # Get the corresponging shard

            node_feature_data = fetch_func(Node_features_MLdataset,node_num,False)
            #edge_data_data = fetch_func(edge_data_shard,True)

            print("len(node_feature_data):",len(node_feature_data))

            print("node_feature_data[0]:\n",node_feature_data[0])

            print("node_feature_data type::",type(node_feature_data))

            ##-------------

            ##须将存储道ObjectStore中的全部数据取出，构造全图，赋值给下面的 g
            # the data should be replaced by the above data, instead of using RedditDataset().
            # g - dgl.graph()##输入边的点对关系后续对其进行特征赋值。
            data = RedditDataset()
            g = data[0]
            g.ndata["features"] = g.ndata["feat"]
            g.ndata["labels"] = g.ndata["label"]
            self.in_feats = g.ndata["features"].shape[1]
            self.n_classes = data.num_classes
            # add self loop,
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            # Create csr/coo/csc formats before launching training processes
            g.create_formats_()
            self.g = g
            train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
            val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
            test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
            self.train_nid = train_nid
            self.val_nid = val_nid
            self.test_nid = test_nid

            # Create sampler
            sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in config["fan_out"].split(",")])
            # Create PyTorch DataLoader for constructing blocks
            collator = NodeCollator(g, train_nid, sampler)
            train_dataloader = DataLoader(
                collator.dataset,
                collate_fn=collator.collate,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=False,
                num_workers=config["sampling_num_workers"])
            # Define model and optimizer, residual is set to True
            model = GAT(self.in_feats, config["n_hidden"], self.n_classes,
                        config["n_layers"], config["n_heads"], F.elu,
                        config["feat_drop"], config["attn_drop"],
                        config["negative_slope"], True)
            self.convs = model.convs
            # Define optimizer.
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            # Register model, optimizer, and loss.
            self.model, self.optimizer = self.register(
                models=model, optimizers=optimizer)
            # Register data loaders.
            self.register_data(train_loader=train_dataloader)

        def train_epoch(self, iterator, info):
            meter_collection = AverageMeterCollection()
            iter_tput = []
            model = self.model
            # for batch_idx,batch in enumerate(iterator):
            for step, (input_nodes, seeds, blocks) in enumerate(iterator):
                tic_step = time.time()
                # do some train
                optimizer = self.optimizer
                device = 0
                if self.use_gpu:
                    blocks = [block.int().to(device) for block in blocks]
                batch_inputs = blocks[0].srcdata["features"]
                batch_labels = blocks[-1].dstdata["labels"]
                batch_pred = model(blocks, batch_inputs)
                loss = F.nll_loss(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % 20 == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = torch.cuda.max_memory_allocated(
                    ) / 1000000 if torch.cuda.is_available() else 0
                    print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                          "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                          "{:.1f} MB".format(info["epoch_idx"] + 1, step,
                                             loss.item(), acc.item(),
                                             np.mean(iter_tput[3:]),
                                             gpu_mem_alloc))
            status = meter_collection.summary()
            return status

        def validate(self, validation_loader, info):
            meter_collection = AverageMeterCollection()
            model = self.model
            n_layers = self.config["n_layers"]
            n_hidden = self.config["n_hidden"]
            n_heads = self.config["n_heads"]
            batch_size = self.config["batch_size"]
            num_workers = self.config["sampling_num_workers"]
            g = self.g
            train_nid = self.train_nid
            val_nid = self.val_nid
            test_nid = self.test_nid
            device = 0
            model.eval()
            with torch.no_grad():
                x = g.ndata["features"]
                for i, layer in enumerate(self.convs):
                    if i < n_layers - 1:
                        y = torch.zeros(
                            g.number_of_nodes(), n_hidden * n_heads
                            if i != len(self.convs) - 1 else self.n_classes)
                    else:
                        y = torch.zeros(
                            g.number_of_nodes(), n_hidden
                            if i != len(self.convs) - 1 else self.n_classes)
                    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                    collator = NodeCollator(g, torch.arange(g.number_of_nodes()),
                                            sampler)
                    dataloader = DataLoader(
                        collator.dataset,
                        collate_fn=collator.collate,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=num_workers)
                    for input_nodes, output_nodes, blocks in dataloader:
                        block = blocks[0]
                        if self.use_gpu:
                            block = block.int().to(device)
                            h = x[input_nodes].to(device)
                            h_dst = x[output_nodes].to(device)
                        else:
                            h = x[input_nodes]
                            h_dst = x[output_nodes]
                        if i != len(self.convs) - 1:
                            h = layer(block, (h, h_dst)).flatten(1)
                        else:
                            h = layer(block, (h, h_dst)).mean(1)
                            h = h.log_softmax(dim=-1)
                        y[output_nodes] = h.cpu()
                    x = y
                pred = y
            labels = g.ndata["labels"]
            _, val_acc, test_acc = compute_acc(pred[train_nid], labels[
                train_nid]), compute_acc(pred[val_nid], labels[val_nid]), \
                                   compute_acc(pred[test_nid], labels[test_nid])

            metrics = {
                "num_samples": pred.size(0),
                "val_acc": val_acc.item(),
                "test_acc": test_acc.item()
            }
            meter_collection.update(metrics, n=metrics.pop("num_samples", 1))
            status = meter_collection.summary()
            return status
    ###----------------

    run(num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        fan_out=args.fan_out,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop,
        negative_slope=args.negative_slope,
        sampling_num_workers=args.sampling_num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker)
