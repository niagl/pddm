# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os



os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import numpy.random as npr
import tensorflow as tf
import pickle
import sys
import argparse
import traceback

#my imports
from pddm.policies.policy_random import Policy_Random
from pddm.utils.helper_funcs import *
from pddm.regressors.dynamics_model import Dyn_Model
from pddm.regressors.distrib_model import Distrib_Model
from pddm.policies.mpc_rollout import MPCRollout
from pddm.utils.loader import Loader
from pddm.utils.saver import Saver
from pddm.utils.data_processor import DataProcessor
from pddm.utils.data_structures import *
from pddm.utils.convert_to_parser_args import convert_to_parser_args
from pddm.utils import config_reader

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

SCRIPT_DIR = os.path.dirname(__file__)

def viz_agent(args, save_dir):

    tf.reset_default_graph()
    with tf.Session(config=get_gpu_config(args.use_gpu, args.gpu_frac)) as sess:

        ##############################################
        ### initialize some commonly used parameters (from args)
        ##############################################

        env_name = args.env_name
        K = args.K
        num_trajectories_per_iter = args.num_trajectories_per_iter
        horizon = args.horizon

        ### set seeds
        npr.seed(args.seed)
        tf.set_random_seed(args.seed)

        #######################
        ### hardcoded args
        #######################

        ### data types
        args.tf_datatype = tf.float32
        args.np_datatype = np.float32

        ### supervised learning noise, added to the training dataset
        args.noiseToSignal = 0.01

        ### these are for *during* MPC rollouts,
        # they allow you to run the H-step candidate actions on the real dynamics
        # and compare the model's predicted outcomes vs. the true outcomes
        execute_sideRollouts = False
        plot_sideRollouts = True

        ########################################
        ### create loader, env, rand policy
        ########################################

        if args.add_noise_to_env:
            env, dt_from_xml = create_env(env_name, np.array(args.noise_params))
        else:
            env, dt_from_xml = create_env(env_name)
        args.dt_from_xml = dt_from_xml
        random_policy = Policy_Random(env.env)
        # writer = tf.summary.FileWriter(args.tensorboard_dir)

        #doing a render here somehow allows it to not produce a seg fault error later when visualizing
        if args.visualize_MPC_rollout:
            render_env(env)
            render_stop(env)

        #################################################
        ### initialize or load in info
        #################################################

        #check for a variable which indicates that we should duplicate each data point
        #e.g., for baoding, since ballA/B are interchangeable, we store as 2 different points
        if 'duplicateData_switchObjs' in dir(env.unwrapped_env):
            duplicateData_switchObjs = True
            indices_for_switching = [env.unwrapped_env.objInfo_start1, env.unwrapped_env.objInfo_start2,
                                    env.unwrapped_env.targetInfo_start1, env.unwrapped_env.targetInfo_start2]
        else:
            duplicateData_switchObjs = False
            indices_for_switching=[]

        #initialize data processor
        data_processor = DataProcessor(args, duplicateData_switchObjs, indices_for_switching)

        ##############################################
        ### dynamics model + controller
        ##############################################

        loader = Loader(save_dir)
        rollouts_trainRand, rollouts_valRand = loader.load_initialData()

            # convert (rollouts --> dataset)
        dataset_trainRand = data_processor.convertRolloutsToDatasets(
            rollouts_trainRand)

        inputSize, outputSize, acSize = check_dims(dataset_trainRand, env)

        dyn_models = Dyn_Model(inputSize, outputSize, acSize, sess, params=args)

        if args.use_distrib_reward:
            distrib_models = Distrib_Model(inputSize, acSize, sess, params=args)

            mpc_rollout = MPCRollout(env, dyn_models, random_policy,
                                     execute_sideRollouts, plot_sideRollouts, args, distrib_models)

        else:
            mpc_rollout = MPCRollout(env, dyn_models, random_policy,
                                     execute_sideRollouts, plot_sideRollouts, args)

        ### init TF variables
        sess.run(tf.global_variables_initializer())

        ######################
        ## loss plots for tensorboard

        # tensorboard_var = tf.Variable(0.0)
        # loss_write_pddm = tf.summary.scalar('loss/pddm', tensorboard_var)
        #
        # if args.use_distrib_reward: loss_write_dist = tf.summary.scalar('loss/dist', tensorboard_var)
        #
        # mean_reward_write = tf.summary.scalar('reward', tensorboard_var)
        # mean_score_write = tf.summary.scalar('score', tensorboard_var)

        # save_dir = 'test'
        # saver = Saver(save_dir, sess)
        # saver.iter_num = 0
        # saver.save_model()

        #####################################
        ## Loading the model
        #####################################

        restore_path = save_dir + '/models/finalModel' + '.ckpt'

        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(sess, restore_path)
        print("\n\nModel restored from ", restore_path, "\n\n")


        iter_data = loader.load_iter(166)
        data_processor.set_normalization_data(dyn_models, iter_data.normalization_data)


        #saving rollout info
        rollouts_info = []
        list_rewards = []
        list_scores = []
        list_mpes = []

        for rollout_num in range(5):

            ###########################################
            ########## perform 1 MPC rollout
            ###########################################

            if not args.print_minimal:
                print("\n####################### Performing MPC rollout #",
                      rollout_num)

            #reset env randomly
            starting_observation, starting_state = env.reset(return_start_state=True)

            rollout_info = mpc_rollout.perform_rollout(
                starting_state,
                starting_observation,
                controller_type=args.controller_type,
                use_dist_reward=args.use_distrib_reward,
                take_exploratory_actions=False)

            # Note: can sometimes set take_exploratory_actions=True
            # in order to use ensemble disagreement for exploration

            ###########################################
            ####### save rollout info (if long enough)
            ###########################################

            if len(rollout_info['observations']) > K:
                list_rewards.append(rollout_info['rollout_rewardTotal'])
                list_scores.append(rollout_info['rollout_meanFinalScore'])
                list_mpes.append(np.mean(rollout_info['mpe_1step']))
                rollouts_info.append(rollout_info)

        # visualize, if desired
        if args.visualize_MPC_rollout:
            print("\n\nPAUSED FOR VISUALIZATION. Continue when ready to visualize.")
            import IPython
            IPython.embed()
            for vis_index in range(len(rollouts_info)):
                visualize_rendering(rollouts_info[vis_index], env, args)

        #########################################################
        ### aggregate MPC rollouts into train/val
        #########################################################

        num_mpc_rollouts = len(rollouts_info)

        for i in range(num_mpc_rollouts):
            rollout = Rollout(rollouts_info[i]['observations'],
                              rollouts_info[i]['actions'],
                              rollouts_info[i]['rollout_rewardsPerStep'],
                              rollouts_info[i]['starting_state'],
                              rollouts_info[i]['rollout_done'])

        return


def main():

    #####################
    # training args
    #####################

    parser = argparse.ArgumentParser(
        # Show default value in the help doc.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        '-c',
        '--config',
        nargs='*',
        help=('Path to the job data config file. This is specified relative '
            'to working directory'))

    parser.add_argument('--use_gpu', action="store_true")
    parser.add_argument('-frac', '--gpu_frac', type=float, default=0.9)
    general_args = parser.parse_args()

    #####################
    # job configs
    #####################

    # Get the job config files
    jobs = config_reader.process_config_files(general_args.config)
    assert jobs, 'No jobs found from config.'

    # Run separate experiment for each variant in the config
    for index, job in enumerate(jobs):

        #add an index to jobname, if there is more than 1 job
        if len(jobs)>1:
            job['job_name'] = '{}_{}'.format(job['job_name'], index)

        #convert job dictionary to different format
        args_list = config_dict_to_flags(job)
        args = convert_to_parser_args(args_list)

        #copy some general_args into args
        args.use_gpu = general_args.use_gpu
        args.gpu_frac = general_args.gpu_frac

        args.use_distrib_reward = True

        ################
        ### run job
        ################

        # save_dir = './../../../results/baseline_tf_code/baoding/baoding_reRun_Feb2/baoding'
        save_dir = './../../../results/pddm_c51_tf/pddm_c51_baoding/baoding'

        try:
            viz_agent(args, save_dir)
        except (KeyboardInterrupt, SystemExit):
            print('Terminating...')
            sys.exit(0)
        except Exception as e:
            print('ERROR: Exception occured while running a job....')
            traceback.print_exc()


if __name__ == '__main__':
    main()

    # save_dir = './../../../results/baseline_tf_code/baoding/baoding_reRun_25Sep'
    # restore_path = save_dir + '/models/finalModel' + '.ckpt'
    # print_tensors_in_checkpoint_file(file_name=restore_path, all_tensors=True, tensor_name='')