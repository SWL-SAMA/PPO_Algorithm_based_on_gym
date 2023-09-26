"""
姓名：刘诗伟
学号：2011832
2023.05.22
python: 3.6
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt
from arguments import args

env = gym.make('Pendulum-v1').unwrapped
state_number = env.observation_space.shape[0]
action_number = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
# 超参数定义
epsilon = 0.2  # clip 的epsilon参数
gamma = 0.9  # 折扣率
batch_size = 128  # 批量大小
max_episode_len = 200  # 最大牧场
action_learning_rate = 1e-04
critic_learning_rate = 3e-04
torch.manual_seed(0)
env.seed(0)
RENDER = False


#  策略网络
class Actor_net(nn.Module):
    def __init__(self):
        super(Actor_net, self).__init__()
        self.in_layer = nn.Linear(state_number, 100)
        self.act_out = nn.Linear(100, action_number)
        self.std_out = nn.Linear(100, action_number)
        self.in_layer.weight.data.normal_(0, 0.1)
        self.act_out.weight.data.normal_(0, 0.1)
        self.std_out.weight.data.normal_(0, 0.1)

    #  前向传播，返回动作的分布
    def forward(self, state):
        state = self.in_layer(state)
        state = F.relu(state)
        act_out = self.act_out(state)
        mean = max_action * torch.tanh(act_out)
        std_out = self.std_out(state)
        std = F.softplus(std_out)
        return mean, std


class Critic_net(nn.Module):
    def __init__(self):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(state_number, 100)
        self.fc2 = nn.Linear(100, 1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    #  前向传播,返回q值
    def forward(self, state):
        state = self.fc1(state)
        state = F.relu(state)
        q = self.fc2(state)
        return q


#  策略选取
class Actor:
    def __init__(self, act_lr):
        #  新旧策略
        self.old_actor = Actor_net()
        self.new_actor = Actor_net()
        #  网络优化器
        self.optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=act_lr, eps=1e-05)

    #  用两种策略生成动作
    def get_act(self, state):
        state = torch.FloatTensor(state)
        mean, std = self.old_actor(state)
        dist = Normal(mean, std)
        act = dist.sample()
        act = torch.clip(act, min_action, max_action)  # 采样动作后将范围限制在动作的取值区间
        act_log = dist.log_prob(act)  # 动作的对数
        return act.detach(), act_log.detach()

    #  将新策略的参数结构全部复制给就策略
    def update_old(self):
        self.old_actor.load_state_dict(self.new_actor.state_dict())

    #  训练,接受一批数据，每一批数据训练10次,对new_actor进行更新
    def learn(self, batch_state, batch_act, batch_adv, batch_alog):
        batch_state = torch.FloatTensor(batch_state)
        batch_act = torch.FloatTensor(batch_act)
        batch_adv = torch.FloatTensor(batch_adv)
        batch_alog = torch.FloatTensor(batch_alog)
        for i in range(10):
            #  通过新策略产生新的动作分布与loga
            act_mean, act_std = self.new_actor(batch_state)
            dist_new = torch.distributions.Normal(act_mean, act_std)
            action_new_log = dist_new.log_prob(batch_act)
            ratio = torch.exp(action_new_log - batch_alog.detach())  # 重要度
            surr1 = ratio * batch_adv  # 公式的前面一项
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_adv  # 公式后面一相
            loss = -torch.min(surr1, surr2)
            loss = loss.mean()  # 均值作为loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.new_actor.parameters(), 0.5)  # 梯度裁剪，防止其过大
            self.optimizer.step()


class Critic:
    def __init__(self, cri_lr):
        self.q_critic = Critic_net()
        self.optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=cri_lr, eps=1e-05)
        self.loss_func = nn.MSELoss()

    def get_q(self, state):
        state = torch.FloatTensor(state)
        q = self.q_critic(state)
        return q

    def learn_and_adv(self, batch_state, batch_disr):
        bach_state = torch.FloatTensor(batch_state)
        target_q = torch.FloatTensor(batch_disr)
        for i in range(10):
            q = self.get_q(bach_state)
            td_error = self.loss_func(target_q, q)
            self.optimizer.zero_grad()
            td_error.backward()
            nn.utils.clip_grad_norm_(self.q_critic.parameters(), 0.5)
            self.optimizer.step()
        return (target_q - q).detach()  # 返回优势值函数


class PPO:
    def __init__(self):
        self.actor = Actor(act_lr=action_learning_rate)
        self.q_critic = Critic(cri_lr=critic_learning_rate)
        self.sum_re = []

    #  保存模型
    def save_model(self, file_path, index):
        save_data = {'net': self.actor.old_actor.state_dict(), 'opt': self.actor.optimizer.state_dict(), 'i': index}
        torch.save(save_data, file_path + "PPO_actor.pth")
        save_data = {'net': self.q_critic.q_critic.state_dict(), 'opt': self.q_critic.optimizer.state_dict(),
                     'i': index}
        torch.save(save_data, file_path + "PPO_critic.pth")

    #  在线学习，边采样边更新
    def train(self, train_num, file_path):
        print('PPO_clip training:')
        for num in range(train_num):
            b_state = []
            b_action = []
            b_reward = []
            b_loga = []
            reward_sum = 0
            state = env.reset()
            for t_step in range(max_episode_len):
                action, log_a = self.actor.get_act(state)  # 通过旧的策略网络获取act与其对数
                action = action.numpy()
                log_a = log_a.numpy()
                new_state, reward, done, _ = env.step(action)  # 与环境交互
                # 加入批数据
                b_state.append(state)
                b_reward.append((reward + 8) / 8)  # 缩小回报？？？？？
                b_loga.append(log_a)
                b_action.append(action)
                state = new_state  # 状态更新
                reward_sum += reward
                # 更新策略与值
                if (t_step + 1) % batch_size == 0 or t_step == max_episode_len - 1:
                    q_new = self.q_critic.get_q(new_state)
                    #  计算折扣累计回报
                    dis_r = []
                    for reward in b_reward[::-1]:
                        q_new = reward + gamma * q_new
                        dis_r.append(q_new.detach().numpy())
                    dis_r.reverse()  # 逆序
                    b_state, b_action, b_disr, b_loga = np.vstack(b_state), np.vstack(b_action), np.array(
                        dis_r), np.vstack(
                        b_loga)
                    b_adv = self.q_critic.learn_and_adv(b_state, b_disr)
                    self.actor.learn(b_state, b_action, b_adv, b_loga)
                    b_state = []
                    b_action = []
                    b_reward = []
                    b_loga = []
                    #  更新策略
                    self.actor.update_old()
            if num == 0:
                self.sum_re.append(reward_sum)
            else:
                self.sum_re.append(self.sum_re[-1] * 0.9 + reward_sum * 0.1)
            if reward_sum >= max(self.sum_re):
                self.save_model(file_path, num)
            print("\rEpoch: {} |average_rewards: {}".format(num, reward_sum), end="")
        self.plot_train_rewards(len(self.sum_re), self.sum_re)
        env.close()

    def test(self, act_file_path, cri_file_path):
        net_actor = torch.load(act_file_path)
        net_critic = torch.load(cri_file_path)
        self.actor.old_actor.load_state_dict(net_actor['net'])
        self.q_critic.q_critic.load_state_dict(net_critic['net'])
        #  测试10个episode
        for j in range(10):
            state = env.reset()
            total_rewards = 0
            for timestep in range(max_episode_len):
                env.render()
                action, action_logprob = self.actor.get_act(state)
                new_state, reward, done, _ = env.step(action)  # 执行动作
                total_rewards += reward
                state = new_state
            print("Score：", total_rewards)
        env.close()

    def plot_train_rewards(self, xl, data):
        plt.plot(np.arange(xl), data)
        plt.xlabel('Epochs')
        plt.ylabel('Average episode reward')
        plt.show()


if __name__ == '__main__':
    MountainCar_solver = PPO()  # 实例化
    if args.run_mode=='train':
        MountainCar_solver.train(train_num=args.train_num, file_path='trained_modules\\')
    else:
        MountainCar_solver.test(act_file_path='trained_modules\\PPO_actor.pth', cri_file_path='trained_modules\\PPO_critic.pth')