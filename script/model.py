import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
from transformer import transformer_model, gelu

TIME_INTERVAL = 16
ITEM_BH_CLS_CNT = 3

weight_emb_w = [[16, 8], [8, 4]]
weight_emb_b = [0, 0]
print(weight_emb_w, weight_emb_b)
orders = 3
order_indep = False  # True
WEIGHT_EMB_DIM = (sum([w[0] * w[1] for w in weight_emb_w]) + sum(weight_emb_b))  # * orders
INDEP_NUM = 1
if order_indep:
    INDEP_NUM *= orders

print("orders: ", orders)
CALC_MODE = "can"
device = '/cpu:0'
keep_fake_carte_seq = False

def gen_coaction(ad, his_items, dim, mode="can", mask=None):
    weight, bias = [], []
    idx = 0
    weight_orders = []
    bias_orders = []
    for i in range(orders):
        for w, b in zip(weight_emb_w, weight_emb_b):
            weight.append(tf.reshape(ad[:, idx:idx + w[0] * w[1]], [-1, w[0], w[1]]))
            idx += w[0] * w[1]
            if b == 0:
                bias.append(None)
            else:
                bias.append(tf.reshape(ad[:, idx:idx + b], [-1, 1, b]))
                idx += b
        weight_orders.append(weight)
        bias_orders.append(bias)
        if not order_indep:
            break

    if mode == "can":
        out_seq = []
        hh = []
        for i in range(orders):
            hh.append(his_items ** (i + 1))
        for i, h in enumerate(hh):
            if order_indep:
                weight, bias = weight_orders[i], bias_orders[i]
            else:
                weight, bias = weight_orders[0], bias_orders[0]
            for j, (w, b) in enumerate(zip(weight, bias)):
                h = tf.matmul(h, w)
                if b is not None:
                    h = h + b
                if j != len(weight) - 1:
                    h = tf.nn.tanh(h)
                out_seq.append(h)
        out_seq = tf.concat(out_seq, 2)
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
            out_seq = out_seq * mask
    out = tf.reduce_sum(out_seq, 1)
    if keep_fake_carte_seq and mode == "emb":
        return out, out_seq
    return out, None

class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False,use_coaction = False):
        with tf.name_scope('Inputs'):
            self.use_negsampling = use_negsampling
            self.use_coaction = use_coaction
            self.EMBEDDING_DIM = EMBEDDING_DIM
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
            self.item_user_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='item_user_his_batch_ph')
            self.item_user_his_time_ph = tf.placeholder(tf.int32, [None, None], name='item_user_his_time_ph')
            self.item_user_his_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name= 'item_user_his_mid_batch_ph')
            self.item_user_his_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name= 'item_user_his_cat_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.item_user_his_mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.item_user_his_mid_mask = tf.placeholder(tf.float32, [None, None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.seq_len_u_ph = tf.placeholder(tf.int32, [None], name='seq_len_u_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph') #generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
            self.item_user_his_uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.item_user_his_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            self.item_user_his_mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.item_user_his_mid_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            self.item_user_his_cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.item_user_his_cat_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)

            if self.use_coaction :
                ph_dict = {
                    "item": [self.mid_batch_ph, self.mid_his_batch_ph, self.mid_his_batch_embedded],
                    "cate": [self.cat_batch_ph, self.cat_his_batch_ph, self.cat_his_batch_embedded]
                }
                self.mlp_batch_embedded = []
                with tf.name_scope('CAN_Embedding_layer'):
                    self.item_mlp_embeddings_var = tf.get_variable("item_mlp_embedding_var",[n_mid, INDEP_NUM * WEIGHT_EMB_DIM],trainable=True)
                    self.cate_mlp_embeddings_var = tf.get_variable("cate_mlp_embedding_var",[n_cat, INDEP_NUM * WEIGHT_EMB_DIM],trainable=True)

                    self.mlp_batch_embedded.append(tf.nn.embedding_lookup(self.item_mlp_embeddings_var, ph_dict['item'][0]))
                    self.mlp_batch_embedded.append(tf.nn.embedding_lookup(self.cate_mlp_embeddings_var, ph_dict['cate'][0]))

                    self.input_batch_embedded = []
                    self.item_input_embeddings_var = tf.get_variable("item_input_embedding_var",[n_mid, weight_emb_w[0][0] * INDEP_NUM],trainable=True)
                    self.cate_input_embeddings_var = tf.get_variable("cate_input_embedding_var",[n_cat, weight_emb_w[0][0] * INDEP_NUM],trainable=True)
                    self.input_batch_embedded.append(tf.nn.embedding_lookup(self.item_input_embeddings_var, ph_dict['item'][1]))
                    self.input_batch_embedded.append(tf.nn.embedding_lookup(self.cate_input_embeddings_var, ph_dict['cate'][1]))

            self.time_embeddings_var = tf.get_variable("time_embedding_var", [TIME_INTERVAL, EMBEDDING_DIM])
            tf.summary.histogram('time_embedding_var', self.time_embeddings_var)
            self.item_bh_time_embeeded = tf.nn.embedding_lookup(self.time_embeddings_var, self.item_user_his_time_ph)


            self.item_bh_cls_embedding = tf.get_variable("item_cls_embedding", [ITEM_BH_CLS_CNT, EMBEDDING_DIM * 2])

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_user_his_eb = tf.concat([self.item_user_his_mid_batch_embedded, self.item_user_his_cat_batch_embedded], -1)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb*tf.expand_dims(self.mask,-1), 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat([self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb, [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])
            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

        self.cross = []

        if self.use_coaction :
            input_batch = self.input_batch_embedded
            tmp_sum, tmp_seq = [], []
            if INDEP_NUM == 2:
                for i, mlp_batch in enumerate(self.mlp_batch_embedded):
                    for j, input_batch in enumerate(self.input_batch_embedded):
                        # def gen_coaction(ad, his_items, dim, mode="can", mask=None):
                        coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, WEIGHT_EMB_DIM * j:  WEIGHT_EMB_DIM * (j + 1)],input_batch[:, :, weight_emb_w[0][0] * i: weight_emb_w[0][0] * (i + 1)], EMBEDDING_DIM,mode=CALC_MODE, mask=self.mask)
                        tmp_sum.append(coaction_sum)
                        tmp_seq.append(coaction_seq)
            else:
                for i, (mlp_batch, input_batch) in enumerate(zip(self.mlp_batch_embedded, self.input_batch_embedded)):
                    coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, : INDEP_NUM * WEIGHT_EMB_DIM],input_batch[:, :, : weight_emb_w[0][0]], EMBEDDING_DIM,mode=CALC_MODE, mask=self.mask)
                    tmp_sum.append(coaction_sum)
                    tmp_seq.append(coaction_seq)

            self.coaction_sum = tf.concat(tmp_sum, axis=1)
            self.cross.append(self.coaction_sum)
        #结束

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14],
                self.lr: inps[15],
                self.noclk_mid_batch_ph: inps[16],
                self.noclk_cat_batch_ph: inps[17]
            })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14],
                self.lr: inps[15]
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14],
                self.noclk_mid_batch_ph: inps[15],
                self.noclk_cat_batch_ph: inps[16]
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14]
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)




class TriSeqNet(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True,use_coaction = False):
        super(TriSeqNet, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling,use_coaction = use_coaction)

        maxlen = 20
        other_embedding_size = EMBEDDING_DIM*2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, other_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.item_his_eb)[0], 1])
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.item_his_eb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])

        with tf.name_scope('rnn_1'):
            print("self.item_his_eb.get_shape()",
                  self.item_his_eb.get_shape())
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            print("rnn_outputs.get_shape()", rnn_outputs.get_shape())
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputs= self_multi_head_attn(rnn_outputs, num_units=EMBEDDING_DIM*2, num_heads=4, dropout_rate=0, is_training=True) #(B,T,36)
            print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
            multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs, EMBEDDING_DIM*4,activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM*2)
            multihead_attention_outputs = multihead_attention_outputs1 +multihead_attention_outputs

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1)

        with tf.name_scope('Attention_layer_1'):
            attention_outputs, alphas, alphas_scores_no_softmax = din_attention_new(self.item_eb,multihead_attention_outputs, self.position_his_eb,
                                                                                               ATTENTION_SIZE,self.mask,softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

            alphas_sum = tf.reduce_sum(alphas, axis=-1)
            alphas_sum = tf.expand_dims(alphas_sum, -1)

        with tf.name_scope('rnn_2'):
            print("tf.expand_dims(alphas, -1).get_shape()",
                  tf.expand_dims(alphas, -1).get_shape())
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=multihead_attention_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            print("final_state2.get_shape()", final_state2.get_shape())
            tf.summary.histogram('GRU2_Final_State', final_state2)
            rnn_outputs2=tf.reduce_sum(rnn_outputs2, 1)

        inp = tf.concat([inp, alphas_sum,rnn_outputs2, final_state2]+self.cross , 1)

        SEQ_USER_T = 50
        INC_DIM = EMBEDDING_DIM * 2

        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')
        self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=SEQ_USER_T)
        self.item_bh_time_embeeded_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=SEQ_USER_T)
        with tf.name_scope('item_representation'):
            att_mask_input = tf.concat([tf.cast(tf.ones([tf.shape(self.item_bh_t)[0], ITEM_BH_CLS_CNT]), float), tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_t)[0], -1])], 1)
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)

            self.item_bh_time_embeeded_padding = tf.layers.dense(self.item_bh_time_embeeded_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_t, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_embeeded_padding
            item_bh_cls_tile = tf.tile(tf.expand_dims(self.item_bh_cls_embedding, 0), [tf.shape(item_bh_drink_trm_input)[0], 1, 1])
            item_bh_drink_trm_input = tf.concat([item_bh_cls_tile, item_bh_drink_trm_input], axis=1)

            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input,
                                                             hidden_size=INC_DIM,
                                                             attention_mask=item_bh_self_att_mask,
                                                             num_hidden_layers=1,
                                                             num_attention_heads=2,
                                                             intermediate_size=256,
                                                             intermediate_act_fn=gelu,
                                                             hidden_dropout_prob=0.2,
                                                             scope='item_bh_drink_trm',
                                                             attention_probs_dropout_prob=0.2,
                                                             do_return_all_layers=False)

            with tf.name_scope('rnn_2'):
                self.item_bh_drink_trm_output, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_drink_trm_output,
                                             sequence_length=self.seq_len_ph, dtype=tf.float32,
                                             scope="gru2")
                tf.summary.histogram('GRU_outputs', self.item_bh_drink_trm_output)

            print('self.item_bh_drink_trm_output.get_shape()', self.item_bh_drink_trm_output.get_shape())
            self.user_embedded = tf.layers.dense(self.uid_batch_embedded, INC_DIM, name='user_map_2dim')
            i_att, _ = attention_net_v1(self.item_bh_drink_trm_output[:, ITEM_BH_CLS_CNT:, :],
                                             sl=self.item_bh_seq_len_t, dec=self.user_embedded,
                                             num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                             is_training=False, reuse=False, scope='item_bh_att')
            item_bh_cls_embs = self.item_bh_drink_trm_output[:, :ITEM_BH_CLS_CNT, :]
            user_embs_for_cls = tf.tile(tf.expand_dims(self.user_embedded, 1), [1, ITEM_BH_CLS_CNT, 1])
            item_bh_cls_dot = item_bh_cls_embs * user_embs_for_cls
            item_bh_cls_dot = tf.reshape(item_bh_cls_dot, [-1, INC_DIM * ITEM_BH_CLS_CNT])
            item_bh_cls_mat = tf.matmul(item_bh_cls_embs, user_embs_for_cls, transpose_b=True)
            item_bh_cls_mat = tf.reshape(item_bh_cls_mat[:, 0, :], [-1, ITEM_BH_CLS_CNT])


        inp = tf.concat([inp, self.item_user_his_eb_sum, i_att, item_bh_cls_dot, item_bh_cls_mat], 1)

        self.build_fcn_net(inp, use_dice=True)


