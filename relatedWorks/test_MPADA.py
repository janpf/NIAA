import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

tf1 = tf.compat.v1

import torch

tf1.disable_v2_behavior()

sys.path[0] = "/workspace"
from relatedWorks.datasets import SSPexels
from SSMTIA.utils import mapping

# imported = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], "./resaved/test")
# saver = tf.train.import_meta_graph(imported)
# saver.restore(sess, "/tmp/model.ckpt")
# list(sess.run(y, feed_dict={x: np.random.randint(low=0, high=255, size=(1, 224, 224, 3))})[0])

test_file = "/workspace/dataset_processing/test_set.txt"
out_file = "/workspace/analysis/not_uploaded/MPADA_test_scores.csv"

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


sess = tf1.InteractiveSession()

imported = tf1.saved_model.load(sess, [tf1.saved_model.tag_constants.SERVING], "/relatedNets/MPADA/resaved/test")

logging.info("successfully loaded model")

x = tf1.get_default_graph().get_tensor_by_name("input_2:0")  # dunno why "_2", but when tracing back softmax-logits to all connected inputs, only input_2 is connected :shrug:
y = tf1.get_default_graph().get_tensor_by_name("softmax-logits:0")

logging.info("creating dataloader")
dataset = SSPexels(file_list_path=test_file, mapping=mapping, normalize=False, moveAxis=False)
batch_loader = torch.utils.data.DataLoader(dataset, batch_size=30, drop_last=False, num_workers=8)

out_file = open(out_file, "w")
out_file.write("img;parameter;change;scores\n")

logging.info("testing")

for i, data in enumerate(batch_loader):
    logging.info(f"{i}/{len(batch_loader)}")
    for key in data.keys():
        if key == "file_name":
            continue

        img: np.ndarray = data[key].numpy()
        out = sess.run(y, feed_dict={x: img})
        for p, s in zip(data["file_name"], out):
            if key == "original":
                key = "original;0"
            out_file.write(f"{p};{key};{list(s)}\n")

out_file.close()
