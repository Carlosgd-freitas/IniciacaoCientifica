import models
import functions

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
# from transformers import TFAutoModel, AutoTokenizer, BertTokenizer
import tensorflow as tf ##
import time ##

np.random.seed()

# Hyperparameters
batch_size = 100               # Batch Size
training_epochs = 60           # Total number of training epochs
initial_learning_rate = 0.01   # Initial learning rate

# Pre-processing Parameters
sample_frequency = 160         # Frequency of the sampling
band_pass_1 = [1, 50]          # First filter option, 1~50Hz
band_pass_2 = [10, 30]         # Second filter option, 10~30Hz
band_pass_3 = [30, 50]         # Third filter option, 30~50Hz

# Parameters used in functions.load_data()
train = [1]                    # Tasks used for training and validation
test = [2]                     # Tasks used for testing
window_size = 1920             # Sliding window size, used when composing the dataset
offset = 200                   # Sliding window offset (deslocation), used when composing the dataset
train_val_ratio = 0.9          # 90% for training | 10% for validation

# Channels for some lobes of the brain
frontal_lobe   = ['Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..',
                  'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.']
motor_cortex   = ['C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..']
occipital_lobe = ['Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..']

# 9 channels present in Yang et al. article
frontal_lobe_yang = ['Af3.', 'Afz.', 'Af4.']
motor_cortex_yang = ['C1..', 'Cz..', 'C2..']
occipital_lobe_yang = ['O1..', 'Oz..', 'O2..']
all_channels_yang = ['C1..', 'Cz..', 'C2..', 'Af3.', 'Afz.', 'Af4.', 'O1..', 'Oz..', 'O2..']

# Other Parameters
num_classes = 10#9              # Total number of classes (individuals)
num_channels = 64              # Number of channels in an EEG signal

# Tasks:
# Task 1 - EO
# Task 2 - EC
# Task 3 - T1R1
# Task 4 - T2R1
# Task 5 - T3R1
# Task 6 - T4R1
# Task 7 - T1R2
# Task 8 - T2R2
# Task 9 - T3R2
# Task 10 - T4R2
# Task 11 - T1R3
# Task 12 - T2R3
# Task 13 - T3R3
# Task 14 - T4R3

# def get_angles(pos, i, d_model):
#   angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
#   return pos * angle_rates

# def positional_encoding(position, d_model):
#   angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                           np.arange(d_model)[np.newaxis, :],
#                           d_model)

#   # apply sin to even indices in the array; 2i
#   angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

#   # apply cos to odd indices in the array; 2i+1
#   angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

#   pos_encoding = angle_rads[np.newaxis, ...]

#   return tf.cast(pos_encoding, dtype=tf.float32)

# def scaled_dot_product_attention(q, k, v, mask):
#   """Calculate the attention weights.
#   q, k, v must have matching leading dimensions.
#   k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
#   The mask has different shapes depending on its type(padding or look ahead)
#   but it must be broadcastable for addition.

#   Args:
#     q: query shape == (..., seq_len_q, depth)
#     k: key shape == (..., seq_len_k, depth)
#     v: value shape == (..., seq_len_v, depth_v)
#     mask: Float tensor with shape broadcastable
#           to (..., seq_len_q, seq_len_k). Defaults to None.

#   Returns:
#     output, attention_weights
#   """

#   matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

#   # scale matmul_qk
#   dk = tf.cast(tf.shape(k)[-1], tf.float32)
#   scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

#   # add the mask to the scaled tensor.
#   if mask is not None:
#     scaled_attention_logits += (mask * -1e9)

#   # softmax is normalized on the last axis (seq_len_k) so that the scores
#   # add up to 1.
#   attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

#   output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

#   return output, attention_weights

# class MultiHeadAttention(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads):
#     super(MultiHeadAttention, self).__init__()
#     self.num_heads = num_heads
#     self.d_model = d_model

#     assert d_model % self.num_heads == 0

#     self.depth = d_model // self.num_heads

#     self.wq = tf.keras.layers.Dense(d_model)
#     self.wk = tf.keras.layers.Dense(d_model)
#     self.wv = tf.keras.layers.Dense(d_model)

#     self.dense = tf.keras.layers.Dense(d_model)

#   def split_heads(self, x, batch_size):
#     """Split the last dimension into (num_heads, depth).
#     Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#     """
#     x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#     return tf.transpose(x, perm=[0, 2, 1, 3])

#   def call(self, v, k, q, mask):
#     batch_size = tf.shape(q)[0]

#     q = self.wq(q)  # (batch_size, seq_len, d_model)
#     k = self.wk(k)  # (batch_size, seq_len, d_model)
#     v = self.wv(v)  # (batch_size, seq_len, d_model)

#     q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
#     k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
#     v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

#     # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
#     # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
#     scaled_attention, attention_weights = scaled_dot_product_attention(
#         q, k, v, mask)

#     scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

#     concat_attention = tf.reshape(scaled_attention,
#                                   (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

#     output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

#     return output, attention_weights

# def point_wise_feed_forward_network(d_model, dff):
#   return tf.keras.Sequential([
#       tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
#       tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
#   ])

# class EncoderLayer(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads, dff, rate=0.1):
#     super(EncoderLayer, self).__init__()

#     self.mha = MultiHeadAttention(d_model, num_heads)
#     self.ffn = point_wise_feed_forward_network(d_model, dff)

#     self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#     self.dropout1 = tf.keras.layers.Dropout(rate)
#     self.dropout2 = tf.keras.layers.Dropout(rate)

#   def call(self, x, training, mask):

#     attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
#     attn_output = self.dropout1(attn_output, training=training)
#     out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

#     ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
#     ffn_output = self.dropout2(ffn_output, training=training)
#     out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

#     return out2

# class DecoderLayer(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads, dff, rate=0.1):
#     super(DecoderLayer, self).__init__()

#     self.mha1 = MultiHeadAttention(d_model, num_heads)
#     self.mha2 = MultiHeadAttention(d_model, num_heads)

#     self.ffn = point_wise_feed_forward_network(d_model, dff)

#     self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#     self.dropout1 = tf.keras.layers.Dropout(rate)
#     self.dropout2 = tf.keras.layers.Dropout(rate)
#     self.dropout3 = tf.keras.layers.Dropout(rate)

#   def call(self, x, enc_output, training,
#            look_ahead_mask, padding_mask):
#     # enc_output.shape == (batch_size, input_seq_len, d_model)

#     attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
#     attn1 = self.dropout1(attn1, training=training)
#     out1 = self.layernorm1(attn1 + x)

#     attn2, attn_weights_block2 = self.mha2(
#         enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
#     attn2 = self.dropout2(attn2, training=training)
#     out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

#     ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
#     ffn_output = self.dropout3(ffn_output, training=training)
#     out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

#     return out3, attn_weights_block1, attn_weights_block2

# class Encoder(tf.keras.layers.Layer):
#   def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                maximum_position_encoding, rate=0.1):
#     super(Encoder, self).__init__()

#     self.d_model = d_model
#     self.num_layers = num_layers

#     self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
#     self.pos_encoding = positional_encoding(maximum_position_encoding,
#                                             self.d_model)

#     self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
#                        for _ in range(num_layers)]

#     self.dropout = tf.keras.layers.Dropout(rate)

#   def call(self, x, training, mask):

#     seq_len = tf.shape(x)[1]

#     # adding embedding and position encoding.
#     x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x += self.pos_encoding[:, :seq_len, :]

#     x = self.dropout(x, training=training)

#     for i in range(self.num_layers):
#       x = self.enc_layers[i](x, training, mask)

#     return x  # (batch_size, input_seq_len, d_model)

# class Decoder(tf.keras.layers.Layer):
#   def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
#                maximum_position_encoding, rate=0.1):
#     super(Decoder, self).__init__()

#     self.d_model = d_model
#     self.num_layers = num_layers

#     self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
#     self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

#     self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
#                        for _ in range(num_layers)]
#     self.dropout = tf.keras.layers.Dropout(rate)

#   def call(self, x, enc_output, training,
#            look_ahead_mask, padding_mask):

#     seq_len = tf.shape(x)[1]
#     attention_weights = {}

#     x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x += self.pos_encoding[:, :seq_len, :]

#     x = self.dropout(x, training=training)

#     for i in range(self.num_layers):
#       x, block1, block2 = self.dec_layers[i](x, enc_output, training,
#                                              look_ahead_mask, padding_mask)

#       attention_weights[f'decoder_layer{i+1}_block1'] = block1
#       attention_weights[f'decoder_layer{i+1}_block2'] = block2

#     # x.shape == (batch_size, target_seq_len, d_model)
#     return x, attention_weights

# class Transformer(tf.keras.Model):
#   def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                target_vocab_size, pe_input, pe_target, rate=0.1):
#     super(Transformer, self).__init__()

#     self.encoder = Encoder(num_layers, d_model, num_heads, dff,
#                              input_vocab_size, pe_input, rate)

#     self.decoder = Decoder(num_layers, d_model, num_heads, dff,
#                            target_vocab_size, pe_target, rate)

#     self.final_layer = tf.keras.layers.Dense(target_vocab_size)

#   def call(self, inp, tar, training, enc_padding_mask,
#            look_ahead_mask, dec_padding_mask):

#     enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

#     # dec_output.shape == (batch_size, tar_seq_len, d_model)
#     dec_output, attention_weights = self.decoder(
#         tar, enc_output, training, look_ahead_mask, dec_padding_mask)

#     final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

#     return final_output, attention_weights

model = models.create_model(window_size, num_channels, num_classes)
# model = models.create_model_with_inception(window_size, num_channels, num_classes)
# model = models.create_model_with_SE(window_size, num_channels, num_classes)
# model = models.create_model_identification(window_size, num_channels, num_classes)
model.summary()

# d_model = 512
# transformer = Transformer(
#     num_layers=2, d_model=512, num_heads=8, dff=2048,
#     input_vocab_size=9600, target_vocab_size=9600,
#     pe_input=10000, pe_target=10000)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

#     # add extra dimensions to add the padding
#     # to the attention logits.
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)

# def create_masks(inp, tar):
#     # Encoder padding mask
#     enc_padding_mask = create_padding_mask(inp)

#     # Used in the 2nd attention block in the decoder.
#     # This padding mask is used to mask the encoder outputs.
#     dec_padding_mask = create_padding_mask(inp)

#     # Used in the 1st attention block in the decoder.
#     # It is used to pad and mask future tokens in the input received by
#     # the decoder.
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

#     return enc_padding_mask, combined_mask, dec_padding_mask

# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)

#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask

#     return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


# def accuracy_function(real, pred):
#     accuracies = tf.equal(real, tf.argmax(pred, axis=2))

#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     accuracies = tf.math.logical_and(mask, accuracies)

#     accuracies = tf.cast(accuracies, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=4000):
#     super(CustomSchedule, self).__init__()

#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)

#     self.warmup_steps = warmup_steps

#   def __call__(self, step):
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps ** -1.5)

#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# learning_rate = CustomSchedule(d_model)

# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# checkpoint_path = "./checkpoints/train"

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print('Latest checkpoint restored!!')

# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
#     tar_inp = tar[:, :-1]
#     tar_real = tar[:, 1:]

#     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

#     with tf.GradientTape() as tape:
#         predictions, _ = transformer(inp, tar_inp,
#                                         True,
#                                         enc_padding_mask,
#                                         combined_mask,
#                                         dec_padding_mask)
#         loss = loss_function(tar_real, predictions)

#     gradients = tape.gradient(loss, transformer.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

#     train_loss(loss)
#     train_accuracy(accuracy_function(tar_real, predictions))

# train_batches = 10
# batch = 0

# for epoch in range(training_epochs):
#     start = time.time()

#     train_loss.reset_states()
#     train_accuracy.reset_states()

#     # inp -> portuguese, tar -> english
#     for (batch, (inp, tar)) in enumerate(train_batches):
#         train_step(inp, tar)

#     if batch % 50 == 0:
#         print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#     if (epoch + 1) % 5 == 0:
#         ckpt_save_path = ckpt_manager.save()
#     print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

#     print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#     print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

# Loading the data
x_train, x_val, x_test, y_train, y_val, y_test = functions.load_data('./Dataset/', train, test, num_classes, 
                                                                     band_pass_3, sample_frequency, window_size, 
                                                                     offset, train_val_ratio, 1)

# x_train, x_val, x_test, y_train, y_val, y_test = functions.load_data('/media/work/carlosfreitas/IniciacaoCientifica/RedeNeural/Dataset/', 
#                                                                      train, test, num_classes, band_pass_3, sample_frequency,
#                                                                      window_size, offset, train_val_ratio)                                                            

# Printing data formats
print('\nData formats:')
print(f'x_train: {x_train.shape}')
print(f'x_val: {x_val.shape}')
print(f'x_test: {x_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_val: {y_val.shape}')
print(f'y_test: {y_test.shape}\n')

# Defining the optimizer, compiling, defining the LearningRateScheduler and training the model
opt = SGD(learning_rate = initial_learning_rate, momentum = 0.9)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
callback = LearningRateScheduler(models.scheduler, verbose=0)
results = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = training_epochs,
                    callbacks = [callback],
                    validation_data = (x_val, y_val)
                    )

# results = model.fit(
#                     {"input_ids": input_ids,
#                     "attention_mask": mask},
#                     y_train,
#                     batch_size = batch_size,
#                     epochs = training_epochs,
#                     callbacks = [callback],
#                     validation_data = (x_val, y_val)
#                     )

# Saving model weights
model.save('model_weights.h5')

# Evaluate the model to see the accuracy
print('\nEvaluating on training set...')
(loss, accuracy) = model.evaluate(x_train, y_train, verbose = 0)
print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

print('Evaluating on validation set...')
(loss, accuracy) = model.evaluate(x_val, y_val, verbose = 0)
print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

print('Evaluating on testing set...')
(loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

# Summarize history for accuracy
plt.subplot(211)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])

# Summarize history for loss
plt.subplot(212)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.tight_layout()
plt.savefig(r'accuracy-loss.png', format='png')
plt.show()

max_loss = np.max(results.history['loss'])
min_loss = np.min(results.history['loss'])
print("Maximum Loss : {:.4f}".format(max_loss))
print("Minimum Loss : {:.4f}".format(min_loss))
print("Loss difference : {:.4f}\n".format((max_loss - min_loss)))

# Removing the last 2 layers of the model and getting the features array
model_for_verification = models.create_model(window_size, num_channels, num_classes, True)
model_for_verification.summary()
model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
model_for_verification.load_weights('model_weights.h5', by_name=True)
x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Removing the last layer of the model with inception blocks and getting the features array
# model_for_verification = models.create_model_with_inception(window_size, num_channels, num_classes, True)
# model_for_verification.summary()
# model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_for_verification.load_weights('model_weights.h5', by_name=True)
# x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Removing the last layer of the model with squeeze & excitation blocks and getting the features array
# model_for_verification = models.create_model_with_SE(window_size, num_channels, num_classes, True)
# model_for_verification.summary()
# model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_for_verification.load_weights('model_weights.h5', by_name=True)
# x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Removing the last layer of the model with the greatest performance on identification and getting the features array
# model_for_verification = models.create_model_identification(window_size, num_channels, num_classes, True)
# model_for_verification.summary()
# model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_for_verification.load_weights('model_weights.h5', by_name=True)
# x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Calculating EER and Decidability
y_test_classes = functions.one_hot_encoding_to_classes(y_test)
d, eer, thresholds = functions.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
print(f'EER: {eer*100.0} %')
print(f'Decidability: {d}')
