import numpy as np
import tensorflow as tf

from mpl_tf.data.base import BaseDataset


class MPLTrainer:
    def __init__(self, 
                 student: tf.keras.Model, 
                 teacher: tf.keras.Model, 
                 dataset: BaseDataset,
                 student_lr=0.001, teacher_lr=0.0001,
                 loss_fn=None, 
                 sup_teacher=True, approx=True):
        self.S = student
        self.T = teacher
        self.dataset = dataset
        self.opt_S = tf.keras.optimizers.Adam(learning_rate=student_lr)
        self.opt_T = tf.keras.optimizers.Adam(learning_rate=teacher_lr)
        self.loss = loss_fn or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.sup_teacher = sup_teacher
        self.approx = approx

        self.dataset.build()

    @tf.function
    def _mpl_step(self, x_u, x_l, y_l):
        # ----- Teacher creates soft pseudo-labels -----
        logits_u_T = self.T(x_u, training=True)
        probs_u_T  = tf.nn.softmax(logits_u_T)
        y_u_hat    = tf.squeeze(tf.random.categorical(tf.math.log(probs_u_T), 1), axis=1)

        # ==========  Student update on U  ==========
        if self.approx:
            logits_l_before = self.S(x_l, training=True)
            loss_l_before = self.loss(y_l, logits_l_before)

        with tf.GradientTape() as tape_u:
            logits_u_S = self.S(x_u, training=True)
            L_u = self.loss(y_u_hat, logits_u_S)
        grads_S_u = tape_u.gradient(L_u, self.S.trainable_variables)
        grads_S_u, _ = tf.clip_by_global_norm(grads_S_u, 1.0)
        self.opt_S.apply_gradients(zip(grads_S_u, self.S.trainable_variables))

        # ==========  Student forward on labelled batch ==========
        with tf.GradientTape() as tape_l:
            logits_l_S = self.S(x_l, training=True)
            L_l = self.loss(y_l, logits_l_S)
        grads_S_l = tape_l.gradient(L_l, self.S.trainable_variables)

        # ==========  Teacher MPL loss ==========
        if self.approx:
            h_scalar = loss_l_before - L_l
            teacher_mpl_loss = h_scalar * self.loss(y_u_hat, logits_u_T)
        else:
            h_scalar = tf.add_n([tf.reduce_sum(g_u * g_l)
                                 for g_u, g_l in zip(grads_S_u, grads_S_l)])
            teacher_mpl_loss = h_scalar * self.loss(y_u_hat, logits_u_T)

        if self.sup_teacher:
            teacher_logits_sup = self.T(x_l, training=True)
            teacher_sup_loss = self.loss(y_l, teacher_logits_sup)
            teacher_loss = teacher_mpl_loss + teacher_sup_loss
        else:
            teacher_loss = teacher_mpl_loss

        # ==========  Teacher update ==========
        teacher_vars = self.T.trainable_variables
        teacher_grads = tf.gradients(teacher_loss, teacher_vars)
        teacher_grads, _ = tf.clip_by_global_norm(teacher_grads, 1.0)
        self.opt_T.apply_gradients(zip(teacher_grads, teacher_vars))
        return logits_l_S

    def train(self, 
            steps=1000,
        ):
        x_val, y_val   = self.dataset.labelled

        unlabelled_iter = self.dataset.make_iter("unlabelled", shuffle=True)
        labelled_iter   = self.dataset.make_iter("labelled", shuffle=False)

        print(f"Start training model with Meta Pseudo Label startegy...")
        display = 10
        hist_val_acc = []
        for step in range(steps):
            x_u, _   = next(unlabelled_iter)
            x_l, y_l = next(labelled_iter)
            self._mpl_step(x_u, x_l, y_l)

            if step % display == 0:
                logits = self.S.predict(x_val, batch_size=1024, verbose=0)
                val_acc = np.mean(np.argmax(logits, -1) == y_val)

                print(f"Step {step:5d}: Validation Accuracy = {val_acc:.4f}")

        return hist_val_acc
    