import ray
import numpy as np
import math

@ray.remote
def calculate_m_prob(done_batch, rewards_batch, predicted_output, target_output, start_index, end_index, env_dict):
    m_prob = np.zeros([end_index - start_index, 51])
    for i in range(start_index, end_index):
        if done_batch[i]:  # Terminal State
            # Distribution collapses to a single point
            Tz = min(env_dict['v_max'], max(env_dict['v_min'], rewards_batch[i]))
            bj = (Tz - env_dict['v_min']) / env_dict['delta_z']
            l, u = math.floor(bj), math.ceil(bj)
            m_prob[i - start_index][int(l)] += (u - bj)
            m_prob[i - start_index][int(u)] += (bj - l)
        else:
            for j in range(env_dict['num_atoms']):
                Tz = min(env_dict['v_max'], max(env_dict['v_min'], rewards_batch[i] + env_dict['gamma'] * predicted_output[i][j]))
                bj = (Tz - env_dict['v_min']) / env_dict['delta_z']
                l, u = math.floor(bj), min(math.ceil(bj), env_dict['num_atoms'] - 1)
                m_prob[i - start_index][int(l)] += target_output[i][j] * (u - bj)
                m_prob[i - start_index][int(u)] += target_output[i][j] * (bj - l)

    return m_prob