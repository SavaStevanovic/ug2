from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy

class CNNEncoderDecoderPolicy(ActorCriticPolicy):
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False):
        super(CNNEncoderDecoderPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, scale=True)
        self.value_fn, self.policy = create_encoder_decoder_net(self.obs_ph)
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=None):
        return self.sess.run([self._, self,_value, self.neglogp], {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None, deterministic=None):
        return

    def value(self, obs, state=None, mask=None, deterministic=None):
        return self.sess.run(self,_value, {self.obs_ph: obs})

