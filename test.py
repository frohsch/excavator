import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import deque
import random

# Env
class ExcavatorEnv(gym.Env):
    def __init__(self):
        super(ExcavatorEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*32 + [0, 0, 0]),  # 현재 위치, 목표 위치, G 매트릭스, joint_angles
            high=np.array([np.inf]*32 + [np.pi, np.pi/2, np.pi/2]),  # 각도 범위
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-np.pi, -np.pi/2, -np.pi/2]),  # a_1, a_2, a_3의 최솟값
            high=np.array([np.pi, np.pi/2, np.pi/2]),   # a_1, a_2, a_3의 최댓값
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.current_position = np.random.uniform(0, 5, size=(3,))
        self.target_position = np.random.uniform(0, 5, size=(1,))
        self.joint_angles = np.zeros(3)  # swing, boom, arm
        self.G = self.initialize_G_matrix()

        state = np.concatenate([
            self.current_position, 
            self.target_position, 
            self.G.flatten(), 
            self.joint_angles, 
            self.compute_p_coordinates()
        ])
    
        return state

    def initialize_G_matrix(self):
        initial_values = [2, 1, 0, -1, -2]
        g_values = np.random.choice(initial_values, size=(5, 5), replace=True)
        while np.sum(g_values) != 0:
            g_values = np.random.choice(initial_values, size=(5, 5), replace=True)
        return g_values
    
    def step(self, action):
        while True:
            # 액션을 적용하여 각도 업데이트
            self.joint_angles += action

            # p 좌표 업데이트
            p_coordinates = self.compute_p_coordinates()

            # p_x, p_y가 0부터 4까지의 범위 안에 있는지 확인
            if 0 <= p_coordinates[0] <= 4 and 0 <= p_coordinates[1] <= 4:
                # print(p_coordinates)
                # with open("coordinates.txt", "a") as f:
                #     f.write(f"{p_coordinates[0]}, {p_coordinates[1]}, {p_coordinates[2]}\n")
                break  # 유효한 좌표인 경우 루프 종료
            else:
                # 범위를 초과하는 경우 새로운 액션 샘플링
                action = self.action_space.sample()  # 새로운 액션 샘플링

        # 작업 수행
        if action[0] > 0:
            self.digging()
        elif action[1] > 0:
            self.dumping()

        reward = self.calculate_reward()
        done = self.check_done()

        # 상태 업데이트
        state = self._get_state()

        return state, reward, done, {}

    def _get_state(self):
        p_coordinates = self.compute_p_coordinates()
        return np.concatenate([
            self.current_position, 
            self.target_position, 
            self.G.flatten(), 
            self.joint_angles, 
            p_coordinates
        ])
    
    def compute_p_coordinates(self):
        # 좌표 구하기

        l_1 = 6
        l_2 = 4
        h = 1

        x_1 = l_1 * np.cos(self.joint_angles[1]) 
        z_1 = l_1 * np.sin(self.joint_angles[1]) + h

        x_2 = l_2 * np.cos(self.joint_angles[2] + self.joint_angles[1]) + x_1
        z_2 = l_2 * np.sin(self.joint_angles[2] + self.joint_angles[1]) + z_1
        
        x_3 = x_2 * np.cos(self.joint_angles[0])
        y = x_2 * np.sin(self.joint_angles[0])

        p_x = x_3
        p_y = y
        p_z = z_2

        return np.array([p_x, p_y, p_z])

    def update_p_coordinates(self):
        self.current_position = self.compute_p_coordinates()  # p 좌표를 새로운 현재 위치로 설정

    def digging(self):
        closest_x, closest_y = self.find_closest_g_coordinates()
        self.G[closest_x, closest_y] -= 0.125
        self.current_position[2] -= 0.125

    def dumping(self):
        closest_x, closest_y = self.find_closest_g_coordinates()
        self.G[closest_x, closest_y] += 0.125
        self.current_position[2] += 0.125

    def find_closest_g_coordinates(self):
        g_x = int(self.current_position[0] + 2)
        g_y = int(self.current_position[1] + 2)
        g_x = np.clip(g_x, 0, 4)
        g_y = np.clip(g_y, 0, 4)
        return g_x, g_y

    def calculate_reward(self):
        R_d = self.calculate_R_d()
        R_h = self.calculate_R_h()
        R_e = self.calculate_R_e()
        R_g = self.calculate_R_g()

        # print(R_d, R_h, R_e, R_g)

        k_d = 1.0
        k_h = 1.0
        k_e = 1.0
        k_g = 1000.0

        return k_d * R_d + k_h * R_h + k_e * R_e + k_g * R_g

    def calculate_R_d(self):
        closest_x, closest_y = self.find_closest_g_coordinates()
        if self.G[closest_x, closest_y] - self.current_position[2] <= 0:
            return 1.0
        return 0.0

    def calculate_R_h(self):
        D_Max = np.max(np.abs(self.G)) + 1
        sigma = np.std(self.G.flatten())
        return 1 - sigma / D_Max

    def calculate_R_e(self):
        distance = np.linalg.norm(self.current_position - self.target_position)
        return min(distance / 1.0, 1.0)

    def calculate_R_g(self):
        if np.all(self.G == 0):
            return 1.0
        return 0.0

    def check_done(self):
        return np.all(self.G == 0)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []

    def size(self):
        return len(self.buffer)

# Actor-Critic Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 180),
            nn.ReLU(),
            nn.Linear(180, 180),
            nn.ReLU(),
            nn.Linear(180, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 180),
            nn.ReLU(),
            nn.Linear(180, 180),
            nn.ReLU(),
            nn.Linear(180, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.01  # tau 값을 낮춰서 안정성 증가
        self.total_it = 0

        self.sync_networks()
        self.replay_buffer = ReplayBuffer()

    def sync_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        if self.replay_buffer.size() < 512:  # 배치 사이즈 증가
            return

        # 리플레이 버퍼에서 샘플링
        state, action, next_state, reward, not_done = zip(*self.replay_buffer.sample(512))
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        reward = torch.FloatTensor(np.array(reward)).to(device).unsqueeze(1)
        not_done = torch.FloatTensor(np.array(not_done)).to(device).unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * target_Q1

        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        # 크리틱 손실 계산
        critic_loss1 = F.mse_loss(current_Q1, target_Q)
        critic_loss2 = F.mse_loss(current_Q2, target_Q)

        # 크리틱 최적화
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        # 정책 업데이트는 일정 간격으로만 수행
        self.total_it += 1  # 총 업데이트 수 증가
        if self.total_it % 2 == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            # 액터 최적화
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 타겟 모델 업데이트
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 실행 코드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ExcavatorEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

agent = TD3(state_dim, action_dim, max_action)

num_episodes = 10000
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    for step in range(100): 
        action = agent.actor(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action)

        # 리플레이 버퍼에 추가
        agent.replay_buffer.add((state, action, next_state, reward, float(done)))

        # TD3 모델 훈련
        agent.train()

        state = next_state
        episode_reward += reward
        if done:
            break
    episode_rewards.append(episode_reward) 
    print(f"Episode {episode}, Reward: {episode_reward}")

# 그래프 출력
plt.plot(range(num_episodes), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode')
plt.grid(True)
plt.show()