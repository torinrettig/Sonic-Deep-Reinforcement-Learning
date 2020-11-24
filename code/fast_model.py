# Based on Alexandre Borghi's Sonic contest code: https://github.com/aborghi/retro_contest_agent

import os
import time
import random
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from collections import defaultdict
import pickle

import fast_sonic_env

class Model(object):
    def __init__(self, 
                 *, 
                 policy, 
                 ob_space, 
                 ac_space, 
                 nbatch_act, 
                 nbatch_train,
                 nsteps, 
                 ent_coef, 
                 vf_coef, 
                 max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            saver = tf.train.Saver()
            saver.save(sess, save_path)

        def load(load_path):
            saver = tf.train.Saver()
            print('Loading ' + load_path)
            saver.restore(sess, load_path)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.total_timesteps = total_timesteps

    def run(self, update):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []

        for s in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, 
          policy, 
          env, 
          nsteps, 
          total_timesteps, 
          ent_coef, 
          lr,
          vf_coef=0.5,  
          max_grad_norm=0.5, 
          gamma=0.99, 
          lam=0.95,
          log_interval=10, 
          nminibatches=4, 
          noptepochs=4, 
          cliprange=0.2,
          save_interval=0,
          load_path=None):

    #logger.configure('/tmp')

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    assert nbatch % nminibatches == 0

    make_model = lambda : Model(policy=policy, 
                                ob_space=ob_space, 
                                ac_space=ac_space, 
                                nbatch_act=nenvs, 
                                nbatch_train=nbatch_train,
                                nsteps=nsteps, 
                                ent_coef=ent_coef, 
                                vf_coef=vf_coef,
                                max_grad_norm=max_grad_norm)
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, 
                    model=model, 
                    nsteps=nsteps, 
                    total_timesteps=total_timesteps, 
                    gamma=gamma, 
                    lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch

    # Initialize loss dictionary and keys
    loss_dict = defaultdict(float)
    loss_dict['serial_timesteps'] = []
    loss_dict['nupdates'] = []
    loss_dict['total_timestamps'] = []
    loss_dict['fps'] = []
    loss_dict['explained_variance'] = []
    loss_dict['time_elapsed'] = []
    loss_dict['lossvals'] = []
    loss_dict['policy_loss'] = []
    loss_dict['policy_entropy'] = []
    loss_dict['value_loss'] = []
    # loss_dict['mean_test_score'] = []   

    for update in range(1, nupdates+1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(update) #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        ev = explained_variance(values, returns)
        if update % log_interval == 0 or update == 1:
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.record_tabular("policy_loss", float(lossvals[0]))
            logger.record_tabular("policy_entropy", float(lossvals[2]))
            logger.record_tabular("value_loss", float(lossvals[1]))
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

            savepath = "./models/m" + str(update) + "/model.ckpt"
            model.save(savepath)
            print('Saving to', savepath)

            # Test our agent with 3 trials and mean the score
            # This will be useful to see if our agent is improving
            # test_score = testing(model)

            # logger.record_tabular("Mean score test level", test_score)
            # logger.dump_tabular()


        # if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
        #     checkdir = osp.join(logger.get_dir(), 'checkpoints')
        #     os.makedirs(checkdir, exist_ok=True)
        #     savepath = osp.join(checkdir, '%.5i'%update)
        #     print('Saving to', savepath)
        #     model.save(savepath)


        loss_dict['serial_timesteps'].append(update*nsteps)
        loss_dict['nupdates'].append(update)
        loss_dict['total_timestamps'].append(update*nbatch)
        loss_dict['fps'].append(fps)
        loss_dict['explained_variance'].append(float(ev))
        loss_dict['time_elapsed'].append(float(tnow - tfirststart))
        loss_dict['lossvals'].append(lossvals)
        loss_dict['policy_loss'].append(float(lossvals[0]))
        loss_dict['policy_entropy'].append(float(lossvals[2]))
        loss_dict['value_loss'].append(float(lossvals[1]))
        # loss_dict['mean_test_score'].append(test_score)

        if update % log_interval == 0 or update == 1:
            with open("./data/loss_dict_"+str(update)+".pkl", 'wb') as f:
                pickle.dump(loss_dict, f)
                f.close()         
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def testing(model):
    """
    We'll use this function to calculate the score on test levels for each saved model,
    to generate the video version
    to generate the map version
    """

    test_env = DummyVecEnv([fast_sonic_env.make_test])
 
    # Play
    total_score = 0
    trial = 0
    
    # We make 3 trials
    for trial in range(3):
        obs = test_env.reset()
        done = False
        score = 0

        while done == False:
            # Get the action
            action, value, _, neg = model.step(obs)
            
            # Take action in env and look the results
            obs, reward, done, info = test_env.step(action)

            score += reward[0]
        total_score += score
        trial += 1
    test_env.close()

    # Divide the score by the number of trials
    total_test_score = total_score / 3
    return total_test_score

def play(policy, env, update):

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy,
                  ob_space=ob_space,
                  ac_space=ac_space,
                  nbatch_act=0,
                  nbatch_train=0,
                  nsteps=1,
                  ent_coef=0,
                  vf_coef=0,
                  max_grad_norm=0)
    
    # Load the model
    load_path = "./models/m1340/model.ckpt"
    print(load_path)

    obs = env.reset()

    # Play
    score = 0
    done = False

    while done == False:
        # Get the action
        actions, values, _ = model.step(obs)
        
        # Take actions in env and look the results
        obs, rewards, done, info = env.step(actions)
        
        score += rewards
    
        env.render()
        
    print("Score ", score)
    env.close()
