import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.md.langevin import Langevin
from ase.io import read, write
from ase.data import atomic_numbers
from scipy.spatial.transform import Rotation

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DRLAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DeepQNetwork(state_size, action_size).to(self.device)
        self.target_dqn = DeepQNetwork(state_size, action_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(10000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.dqn(state)
        return q_values.argmax().item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.dqn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

class HPtEnvironment:
    def __init__(self, initial_atoms, temperature=300):
        self.initial_atoms = initial_atoms
        self.temperature = temperature
        self.reset()

    def reset(self):
        self.atoms = self.initial_atoms.copy()

    # def __init__(self, size=(5, 5, 4), temperature=300):
    #     self.size = size
    #     self.temperature = temperature
    #     self.reset()

    # def reset(self):
        # # Create Pt(111) surface
        # a = 3.92  # Pt lattice constant
        # pt_positions = []
        # for i in range(self.size[0]):
        #     for j in range(self.size[1]):
        #         for k in range(self.size[2]):
        #             pos = [i * a, j * a * np.sqrt(3) / 2, k * a * np.sqrt(2/3)]
        #             if (i + j + k) % 2 == 0:
        #                 pt_positions.append(pos)

        # # Add H2 molecules above the surface
        # h2_positions = []
        # for _ in range(5):  # Add 5 H2 molecules
        #     x = np.random.uniform(0, self.size[0] * a)
        #     y = np.random.uniform(0, self.size[1] * a * np.sqrt(3) / 2)
        #     z = self.size[2] * a * np.sqrt(2/3) + 2  # 2 Angstroms above the surface
        #     h2_positions.extend([[x, y, z], [x, y, z + 0.74]])  # H-H bond length is ~0.74 Angstroms

        # positions = pt_positions + h2_positions
        # symbols = ['Pt'] * len(pt_positions) + ['H'] * len(h2_positions)
        
        # self.atoms = Atoms(symbols, positions=positions, cell=[self.size[0]*a, self.size[1]*a*np.sqrt(3)/2, 20])
        # self.atoms.center(vacuum=10, axis=2)

        # Set up LAMMPS calculator
        lammps_parameters = {
            'pair_style': 'eam/alloy',
            'pair_coeff': ['* * PtH2.txt Pt H'],
            'mass': ['1 195.084', '2 1.008'],
        }
        self.calc = LAMMPS(parameters=lammps_parameters)
        self.atoms.calc = self.calc

        # Set up Langevin dynamics
        self.dyn = Langevin(self.atoms, 1 * units.fs, temperature_K=self.temperature, friction=0.002)

        return self._get_state()

    def step(self, action):
        # Implement the action (e.g., move an H atom)
        self._apply_action(action)

        # Run MD for a few steps
        for _ in range(10):
            self.dyn.run(1)

        # Calculate reward
        reward = self._calculate_reward()

        # Get new state
        new_state = self._get_state()

        # Check if episode is done
        done = self._is_done()

        return new_state, reward, done

    def _get_state(self):
        # Convert atomic positions to a flat array
        return self.atoms.get_positions().flatten()

    def _apply_action(self, action):
        # Example: move a random H atom in one of 6 directions
        h_indices = [i for i, atom in enumerate(self.atoms) if atom.symbol == 'H']
        if h_indices:
            atom_index = np.random.choice(h_indices)
            direction = action % 6
            move = np.array([
                [0.1, 0, 0], [-0.1, 0, 0],
                [0, 0.1, 0], [0, -0.1, 0],
                [0, 0, 0.1], [0, 0, -0.1]
            ])[direction]
            self.atoms.positions[atom_index] += move

    def _calculate_reward(self):
        # Example: reward based on potential energy change
        return -self.atoms.get_potential_energy()

    def _is_done(self):
        # Example: end episode if all H atoms are adsorbed or desorbed
        h_positions = self.atoms.positions[[atom.index for atom in self.atoms if atom.symbol == 'H']]
        surface_z = np.max(self.atoms.positions[[atom.index for atom in self.atoms if atom.symbol == 'Pt']][:, 2])
        adsorbed = np.all(h_positions[:, 2] < surface_z + 2)
        desorbed = np.all(h_positions[:, 2] > surface_z + 5)
        return adsorbed or desorbed

def train_agent(env, agent, num_episodes, max_steps):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train(64)

            state = next_state
            total_reward += reward

            if done:
                break

        if episode % 10 == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {step + 1}")

# Main execution
initial_atoms = read("initial.txt")
env = HPtEnvironment(initial_atoms=initial_atoms)
# env = HPtEnvironment()
state_size = env.atoms.get_positions().size
action_size = 6  # 6 possible directions to move H atoms
agent = DRLAgent(state_size, action_size)

train_agent(env, agent, num_episodes=1000, max_steps=200)

import matplotlib.pyplot as plt
from ase.visualize import view
from ase.geometry import analysis

def analyze_results(env, agent, num_episodes=10):
    rewards = []
    adsorption_rates = []
    h_positions = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_h_positions = []

        for step in range(200):  # Run for 200 steps
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Record H atom positions
            h_indices = [i for i, atom in enumerate(env.atoms) if atom.symbol == 'H']
            episode_h_positions.append(env.atoms.positions[h_indices])

            if done:
                break

            state = next_state

        rewards.append(total_reward)
        h_positions.append(episode_h_positions)

        # Calculate adsorption rate
        surface_z = np.max(env.atoms.positions[[atom.index for atom in env.atoms if atom.symbol == 'Pt']][:, 2])
        final_h_positions = episode_h_positions[-1]
        adsorbed = np.sum(final_h_positions[:, 2] < surface_z + 2) / len(final_h_positions)
        adsorption_rates.append(adsorbed)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Plot adsorption rates
    plt.figure(figsize=(10, 5))
    plt.plot(adsorption_rates)
    plt.title('H Adsorption Rate per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Adsorption Rate')
    plt.show()

    # Visualize final configuration
    view(env.atoms)

    # Analyze H-Pt distances
    analyzer = analysis.Analysis(env.atoms)
    h_indices = [atom.index for atom in env.atoms if atom.symbol == 'H']
    pt_indices = [atom.index for atom in env.atoms if atom.symbol == 'Pt']
    h_pt_distances = analyzer.get_distances(h_indices, pt_indices)

    print(f"Average H-Pt distance: {np.mean(h_pt_distances):.2f} Å")
    print(f"Minimum H-Pt distance: {np.min(h_pt_distances):.2f} Å")
    print(f"Maximum H-Pt distance: {np.max(h_pt_distances):.2f} Å")

    return rewards, adsorption_rates, h_positions

# After training, call this function
final_rewards, final_adsorption_rates, final_h_positions = analyze_results(env, agent)
