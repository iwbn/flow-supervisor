import tensorflow as tf

class OneCycleLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            max_lr,
            total_steps=None,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.,
            final_div_factor=1e4,
            name=None):
        super(OneCycleLearningRate, self).__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        if anneal_strategy != "linear":
            raise NotImplementedError
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = self.max_lr / self.div_factor
        self.max_lr = self.max_lr
        self.min_lr = self.initial_lr / self.final_div_factor

        self.phases = [
            {
                'end_step': float(self.pct_start * self.total_steps),
                'start_lr': self.initial_lr,
                'end_lr': self.max_lr,
            },
            {
                'end_step': self.total_steps,
                'start_lr': self.max_lr,
                'end_lr': self.min_lr,
            },
        ]

    @tf.function
    def __call__(self, step):
        step = tf.convert_to_tensor(step)

        start_step = tf.constant(0., tf.float32)
        lr = self.min_lr
        for i, phase in enumerate(self.phases):
            end_step = tf.convert_to_tensor(phase['end_step'])
            end_step = tf.cast(end_step, tf.float32)

            if start_step <= tf.cast(step, tf.float32) and tf.cast(step, tf.float32) < end_step:
                pct = (tf.cast(step, tf.float32) - start_step) / (end_step - start_step)
                lr = self._annealing_linear(phase['start_lr'], phase['end_lr'], pct)
            else:
                lr = lr
            start_step = end_step

        return lr

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start


class ExponentialLearningRateSmurf(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            max_lr,
            min_lr,
            total_steps,
            const_portion=0.8,
            name=None):
        super(ExponentialLearningRateSmurf, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.const_portion = const_portion
        self.decay_steps = int(self.total_steps * (1. - self.const_portion))
        self.start_at = self.total_steps - self.decay_steps
        self.exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.max_lr,
                                                                                decay_steps=self.decay_steps,
                                                                                decay_rate=self.min_lr/self.max_lr,
                                                                                staircase=False)

    @tf.function
    def __call__(self, step):
        step = tf.convert_to_tensor(step)
        if step >= self.start_at:
            d_step = step - self.start_at
            return self.exponential_decay(d_step)
        else:
            return self.max_lr