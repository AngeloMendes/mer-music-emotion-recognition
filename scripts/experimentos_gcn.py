import networkx as nx
import numpy as np
import pandas as pd
import sys
import scipy.sparse
import tensorflow as tf
import time

import gcn_tutorial.layers.graph as lg
import gcn_tutorial.utils.sparse as us

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from glob import glob
from tqdm import tqdm


def set_label(G, grau, df):
    # G: grafo - grau:  tipo de rótulo in [grau_arousal, grau_valence]
    for node in G.nodes():
        if ':music' in node:
            label = df[df.musicId == int(node.replace(':music', ''))][grau].to_list()[0]
            G.nodes[node]['label'] = label
            # Y.append(label)
    return G


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def build_gcn(G, run):
    # if True:

    run += 1
    network = str(run)
    node_list = []
    for node in G.nodes():
        node_list.append(node)
        # add lista com tipo do no train/test

    label_codes = {}
    for node in node_list:
        if 'label' in G.nodes[node]:
            label = G.nodes[node]['label']
            if label not in label_codes: label_codes[label] = len(label_codes)
            G.nodes[node]['membership'] = label_codes[label]
        else:
            G.nodes[node]['membership'] = -1

    # adj = nx.adj_matrix(G,nodelist=node_list)
    adj = nx.adjacency_matrix(G, nodelist=node_list)

    # Get important parameters of adjacency matrix
    n_nodes = adj.shape[0]

    # Some preprocessing
    adj_tilde = adj + np.identity(n=adj.shape[0])
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1 / 2)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))

    # Features are just the identity matrix
    # feat_x = np.identity(n=adj.shape[0])

    # get true labels
    y_true = []
    y_true_index = []
    cnt = 0
    for node in node_list:
        if "split" in G.nodes[node] and 'test' == G.nodes[node]['split']:
            y_true.append(label_codes[G.nodes[node]['label']])
            y_true_index.append(cnt)
        cnt += 1

    # Features from two modalities
    L_X = []
    for node in node_list:
        v1 = list(G.nodes[node]['f_acoustic'])
        v2 = list(G.nodes[node]['f_bert'])
        v_final = v1 + v2
        L_X.append(v_final)
    feat_x = np.array(L_X)

    feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))

    # Preparing train data
    memberships = [m for m in nx.get_node_attributes(G, 'membership').values()]
    nb_classes = len(set(memberships))
    targets = np.array([memberships], dtype=np.int32).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]

    labels_to_keep = []

    counter = 0
    for node in node_list:
        if 'label' in G.nodes[node]:
            labels_to_keep.append(counter)
        counter += 1

    y_train = np.zeros(shape=one_hot_targets.shape,
                       dtype=np.float32)

    train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)

    for l in labels_to_keep:
        y_train[l, :] = one_hot_targets[l, :]
        train_mask[l] = True

    # TensorFlow placeholders
    ph = {
        'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
        'x': tf.sparse_placeholder(tf.float32, name="x"),
        'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes)),
        'mask': tf.placeholder(tf.int32)}

    l_sizes = [512, 256, 128, nb_classes]
    print(nb_classes)  # , set(memberships))

    o_fc1 = lg.GraphConvLayer(
        input_dim=feat_x.shape[-1],
        output_dim=l_sizes[0],
        name='fc1_' + network,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=ph['x'], sparse=True)

    o_fc2 = lg.GraphConvLayer(
        input_dim=l_sizes[0],
        output_dim=l_sizes[1],
        name='fc2_' + network,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)

    o_fc3 = lg.GraphConvLayer(
        input_dim=l_sizes[1],
        output_dim=l_sizes[2],
        name='fc3_' + network,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)

    o_fc4 = lg.GraphConvLayer(
        input_dim=l_sizes[2],
        output_dim=l_sizes[3],
        name='fc4_' + network,
        activation=tf.identity)(adj_norm=ph['adj_norm'], x=o_fc3)

    with tf.name_scope('optimizer'):
        loss = masked_softmax_cross_entropy(preds=o_fc4, labels=ph['labels'], mask=ph['mask'])
        accuracy = masked_accuracy(preds=o_fc4, labels=ph['labels'], mask=ph['mask'])
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        opt_op = optimizer.minimize(loss)

    feed_dict_train = {ph['adj_norm']: adj_norm_tuple,
                       ph['x']: feat_x_tuple,
                       ph['labels']: y_train,
                       ph['mask']: train_mask}

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 1000  # 500  # 100 #800 #1000

    t = time.time()
    outputs = {}
    # Train model
    min_train_acc = 0
    for epoch in tqdm(range(epochs), total=epochs):
        _, train_loss, train_acc = sess.run(
            (opt_op, loss, accuracy), feed_dict=feed_dict_train)
        feed_dict_output = {ph['adj_norm']: adj_norm_tuple, ph['x']: feat_x_tuple}

        # print(train_loss,train_acc)
        if train_acc >= min_train_acc:
            min_train_acc = train_acc
            embeddings = sess.run(o_fc3, feed_dict=feed_dict_output)
            preds = sess.run(o_fc4, feed_dict=feed_dict_output)

    y_pred = []
    for i in y_true_index:
        y_pred.append(preds[i].argmax())

    return y_true, y_pred


if __name__ == '__main__':
    start = sys.argv[1]
    end = sys.argv[2]

    folder_path = "../experimentos"
    grafos = glob(f"{folder_path}/grafos/*/*")

    df = pd.read_csv(f"{folder_path}/df_baseline.tsv", sep='\t')
    run = 4
    for grafo in grafos:
        evaluation = []
        G = nx.read_gpickle(f'{grafo}')
        grau = f"grau_{grafo.split('experimentos/')[1].split('/')[1]}"
        index = grafo.split("_")[1].split(".")[0]  # {path}/grafo_1.nx
        # print(grau,index)
        G = set_label(G=G, grau=grau, df=df)
        y_true, y_pred = build_gcn(G=G, run=run)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        evaluation.append([grau, index, f"grafo_{index}", y_true, y_pred, acc, f1])
        run += 1
        scores = pd.DataFrame(columns=['dominio', '_id', 'grafo', 'y_test', 'y_pred', 'acc', 'f1-macro'],
                              data=evaluation)
        scores.to_csv(f"{folder_path}/resultados_gcn/resutados_gcn_{index}.tsv", sep='\t')
