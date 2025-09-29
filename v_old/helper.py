"""
Helper Functions for Deterministic RL Inference Research
Contains utilities for environments, models, queries, and evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import pickle
import os
from collections import defaultdict, deque
import time
import random

# Set random seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# =================== ENVIRONMENTS ===================

class GridWorld(gym.Env):
    """Simple gridworld environment for testing inference queries"""
    
    def __init__(self, size=10, n_obstacles=5, seed=42):
        super().__init__()
        self.size = size
        self.n_obstacles = n_obstacles
        np.random.seed(seed)
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(size * size)
        
        # Generate obstacles
        self.obstacles = set()
        while len(self.obstacles) < n_obstacles:
            obs = np.random.randint(0, size*size)
            if obs != 0 and obs != size*size-1:  # Don't block start/goal
                self.obstacles.add(obs)
        
        # Start and goal
        self.start_state = 0
        self.goal_state = size * size - 1
        
        # Generate rewards (sparse)
        self.rewards = np.full(size*size, -0.01)  # Small negative step cost
        self.rewards[self.goal_state] = 1.0
        for obs in self.obstacles:
            self.rewards[obs] = -0.1
            
        self.current_state = self.start_state
        
    def pos_to_state(self, row, col):
        return row * self.size + col
    
    def state_to_pos(self, state):
        return state // self.size, state % self.size
    
    def get_next_state(self, state, action):
        """Deterministic transition function"""
        row, col = self.state_to_pos(state)
        
        # Action effects
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)
            
        next_state = self.pos_to_state(row, col)
        
        # Can't move into obstacles
        if next_state in self.obstacles:
            return state
        return next_state
    
    def step(self, action):
        next_state = self.get_next_state(self.current_state, action)
        reward = self.rewards[next_state]
        done = (next_state == self.goal_state)
        
        self.current_state = next_state
        return next_state, reward, done, {}
    
    def reset(self):
        self.current_state = self.start_state
        return self.start_state
    
    def render(self, mode='human'):
        grid = np.zeros((self.size, self.size))
        
        # Mark obstacles
        for obs in self.obstacles:
            row, col = self.state_to_pos(obs)
            grid[row, col] = -1
            
        # Mark goal
        row, col = self.state_to_pos(self.goal_state)
        grid[row, col] = 2
        
        # Mark current position
        row, col = self.state_to_pos(self.current_state)
        grid[row, col] = 1
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='coolwarm')
        plt.title(f'GridWorld {self.size}x{self.size}')
        plt.colorbar(label='Cell Type')
        
        # Add grid lines
        for i in range(self.size + 1):
            plt.axhline(i - 0.5, color='black', linewidth=0.5)
            plt.axvline(i - 0.5, color='black', linewidth=0.5)
            
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.show()

# =================== NEURAL NETWORKS ===================

class StandardDQN(nn.Module):
    """Standard DQN for comparison"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # One-hot encode states if state_dim is 1 (discrete states)
        if state_dim == 1:
            # Assume we're dealing with discrete states that need embedding
            self.state_embedding = None  # Will be set based on environment
            input_dim = hidden_dim // 2
        else:
            input_dim = state_dim
            
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # For discrete state spaces, create embedding
        self.discrete_states = (state_dim == 1)
        if self.discrete_states:
            # Will be initialized when we know the number of states
            self.state_embedding = None
    
    def set_state_space_size(self, n_states):
        """Set the size of discrete state space"""
        if self.discrete_states:
            hidden_dim = self.network[0].in_features
            self.state_embedding = nn.Embedding(n_states, hidden_dim)
    
    def forward(self, state):
        if isinstance(state, int):
            state = torch.tensor([state], dtype=torch.long if self.discrete_states else torch.float32)
        elif isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state, dtype=torch.long if self.discrete_states else torch.float32)
            
        if self.discrete_states:
            if self.state_embedding is None:
                # Emergency fallback - create embedding on the fly
                max_state = state.max().item() if state.numel() > 0 else 0
                hidden_dim = self.network[0].in_features
                self.state_embedding = nn.Embedding(max_state + 10, hidden_dim)
            
            state_emb = self.state_embedding(state)
            if state_emb.dim() > 2:
                state_emb = state_emb.squeeze(1)
        else:
            if state.dim() == 1 and len(state) == 1:
                state_emb = state.float().unsqueeze(0)
            else:
                state_emb = state.float()
                
        return self.network(state_emb)

class QueryConditionedDQN(nn.Module):
    """DQN that takes queries as additional input"""
    
    def __init__(self, state_dim, action_dim, query_dim, hidden_dim=64):
        super().__init__()
        
        self.discrete_states = (state_dim == 1)
        
        if self.discrete_states:
            self.state_embedding = None  # Will be set later
            state_embed_dim = hidden_dim // 2
        else:
            state_embed_dim = state_dim
            
        self.state_encoder = nn.Linear(state_embed_dim, hidden_dim//2)
        self.query_encoder = nn.Linear(query_dim, hidden_dim//2)
        
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def set_state_space_size(self, n_states):
        """Set the size of discrete state space"""
        if self.discrete_states:
            embed_dim = self.state_encoder.in_features
            self.state_embedding = nn.Embedding(n_states, embed_dim)
    
    def forward(self, state, query):
        if isinstance(state, int):
            state = torch.tensor([state], dtype=torch.long if self.discrete_states else torch.float32)
        elif isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state, dtype=torch.long if self.discrete_states else torch.float32)
            
        if self.discrete_states:
            if self.state_embedding is None:
                max_state = state.max().item() if state.numel() > 0 else 0
                embed_dim = self.state_encoder.in_features
                self.state_embedding = nn.Embedding(max_state + 10, embed_dim)
            
            state_emb = self.state_embedding(state)
            if state_emb.dim() > 2:
                state_emb = state_emb.squeeze(1)
        else:
            if state.dim() == 1 and len(state) == 1:
                state_emb = state.float().unsqueeze(0)
            else:
                state_emb = state.float()
        
        if isinstance(query, (list, np.ndarray)):
            query = torch.tensor(query, dtype=torch.float32)
        elif not isinstance(query, torch.Tensor):
            query = torch.tensor([query], dtype=torch.float32)
            
        state_encoded = self.state_encoder(state_emb)
        query_encoded = self.query_encoder(query)
        
        combined = torch.cat([state_encoded, query_encoded], dim=-1)
        return self.network(combined)

# =================== QUERY SYSTEM ===================

@dataclass
class Query:
    """Represents an inference query"""
    type: str  # 'point', 'path', 'set', 'comparative'
    parameters: Dict[str, Any]
    expected_answer_type: str  # 'scalar', 'vector', 'set', 'boolean'

class QueryGenerator:
    """Generate different types of queries for testing"""
    
    def __init__(self, env):
        self.env = env
        self.state_space_size = env.observation_space.n
    
    def generate_point_queries(self, n_queries=100):
        """Generate point queries: V(s), π(s), Q(s,a)"""
        queries = []
        for _ in range(n_queries):
            state = np.random.randint(0, self.state_space_size)
            
            # Value query
            queries.append(Query(
                type='point',
                parameters={'query_type': 'value', 'state': state},
                expected_answer_type='scalar'
            ))
            
            # Policy query
            queries.append(Query(
                type='point', 
                parameters={'query_type': 'policy', 'state': state},
                expected_answer_type='scalar'
            ))
            
            # Q-value query
            action = np.random.randint(0, self.env.action_space.n)
            queries.append(Query(
                type='point',
                parameters={'query_type': 'qvalue', 'state': state, 'action': action},
                expected_answer_type='scalar'
            ))
        
        return queries
    
    def generate_path_queries(self, n_queries=50):
        """Generate path queries: optimal path, path cost"""
        queries = []
        for _ in range(n_queries):
            start = np.random.randint(0, self.state_space_size)
            end = np.random.randint(0, self.state_space_size)
            
            # Optimal path query
            queries.append(Query(
                type='path',
                parameters={'query_type': 'optimal_path', 'start': start, 'end': end},
                expected_answer_type='vector'
            ))
            
            # Path cost query (for a specific path)
            path_length = np.random.randint(3, 8)
            path = [start]
            current = start
            for _ in range(path_length-1):
                action = np.random.randint(0, self.env.action_space.n)
                next_state = self.env.get_next_state(current, action)
                path.append(next_state)
                current = next_state
                
            queries.append(Query(
                type='path',
                parameters={'query_type': 'path_cost', 'path': path},
                expected_answer_type='scalar'
            ))
        
        return queries
    
    def generate_set_queries(self, n_queries=30):
        """Generate set queries: reachable states, high-value states"""
        queries = []
        for _ in range(n_queries):
            
            # Reachable states within k steps
            start = np.random.randint(0, self.state_space_size)
            k = np.random.randint(1, 6)
            queries.append(Query(
                type='set',
                parameters={'query_type': 'reachable', 'start': start, 'steps': k},
                expected_answer_type='set'
            ))
            
            # High value states
            threshold = np.random.uniform(0.3, 0.9)
            queries.append(Query(
                type='set',
                parameters={'query_type': 'high_value', 'threshold': threshold},
                expected_answer_type='set'
            ))
        
        return queries
    
    def generate_comparative_queries(self, n_queries=30):
        """Generate comparative queries: which is better"""
        queries = []
        for _ in range(n_queries):
            state = np.random.randint(0, self.state_space_size)
            action1 = np.random.randint(0, self.env.action_space.n)
            action2 = np.random.randint(0, self.env.action_space.n)
            
            queries.append(Query(
                type='comparative',
                parameters={
                    'query_type': 'better_action',
                    'state': state,
                    'action1': action1,
                    'action2': action2
                },
                expected_answer_type='boolean'
            ))
        
        return queries

# =================== TRAINING UTILITIES ===================

def train_dqn(env, model, episodes=1000, lr=0.001, gamma=0.99, epsilon_decay=0.995):
    """Train a DQN model"""
    
    # Set up state space size for embedding
    if hasattr(model, 'set_state_space_size'):
        model.set_state_space_size(env.observation_space.n)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    epsilon = 1.0
    epsilon_min = 0.01
    episode_rewards = []
    losses = []
    
    print(f"Training DQN for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        max_steps = env.size * env.size * 2  # Prevent infinite episodes
        
        while steps < max_steps:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    next_q_values = model(next_state)
                    target = reward + gamma * next_q_values.max().item()
            
            current_q = model(state)[action]
            loss = criterion(current_q, torch.tensor(target, dtype=torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            episode_reward += reward
            episode_loss += loss.item()
            state = next_state
            steps += 1
            
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / max(steps, 1))
        
        if episode % 200 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"  Episode {episode:4d}, Avg Reward: {avg_reward:6.3f}, Epsilon: {epsilon:.3f}")
    
    print("Training completed!")
    return episode_rewards, losses

# =================== EVALUATION UTILITIES ===================

def compute_ground_truth_values(env, gamma=0.99, max_iterations=1000):
    """Compute exact values using value iteration"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    print(f"Computing ground truth values for {n_states} states...")
    
    # Initialize
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    
    # Value iteration
    for iteration in range(max_iterations):
        V_old = V.copy()
        
        for s in range(n_states):
            for a in range(n_actions):
                next_s = env.get_next_state(s, a)
                reward = env.rewards[next_s]
                Q[s, a] = reward + gamma * V[next_s]
            
            V[s] = np.max(Q[s])
        
        # Check convergence
        if np.allclose(V, V_old, rtol=1e-6):
            print(f"  Value iteration converged after {iteration + 1} iterations")
            break
    else:
        print(f"  Value iteration reached max iterations ({max_iterations})")
    
    # Compute optimal policy
    policy = np.argmax(Q, axis=1)
    
    print("Ground truth computation completed!")
    return V, Q, policy

def evaluate_query_answering(model, env, queries, ground_truth):
    """Evaluate model's ability to answer queries"""
    V_true, Q_true, policy_true = ground_truth
    results = defaultdict(list)
    
    print(f"Evaluating {len(queries)} queries...")
    
    for query in queries:
        if query.type == 'point':
            state = query.parameters['state']
            
            if query.parameters['query_type'] == 'value':
                # Model prediction
                with torch.no_grad():
                    q_values = model(state)
                    predicted_value = q_values.max().item()
                
                true_value = V_true[state]
                error = abs(predicted_value - true_value)
                results['value_errors'].append(error)
                
            elif query.parameters['query_type'] == 'policy':
                with torch.no_grad():
                    q_values = model(state)
                    predicted_action = q_values.argmax().item()
                
                true_action = policy_true[state]
                correct = (predicted_action == true_action)
                results['policy_accuracy'].append(correct)
                
            elif query.parameters['query_type'] == 'qvalue':
                state = query.parameters['state']
                action = query.parameters['action']
                
                with torch.no_grad():
                    q_values = model(state)
                    predicted_q = q_values[action].item()
                
                true_q = Q_true[state, action]
                error = abs(predicted_q - true_q)
                results['qvalue_errors'].append(error)
        
        elif query.type == 'comparative':
            if query.parameters['query_type'] == 'better_action':
                state = query.parameters['state']
                action1 = query.parameters['action1']
                action2 = query.parameters['action2']
                
                # True comparison
                true_q1 = Q_true[state, action1]
                true_q2 = Q_true[state, action2]
                true_better = action1 if true_q1 > true_q2 else action2
                
                # Predicted comparison
                with torch.no_grad():
                    q_values = model(state)
                    pred_q1 = q_values[action1].item()
                    pred_q2 = q_values[action2].item()
                    pred_better = action1 if pred_q1 > pred_q2 else action2
                
                correct = (true_better == pred_better)
                results['comparison_accuracy'].append(correct)
    
    print("Query evaluation completed!")
    return results

# =================== VISUALIZATION UTILITIES ===================

def plot_training_curves(rewards, losses, title="Training Progress"):
    """Plot training rewards and losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Smooth the curves for better visualization
    def smooth_curve(data, window=50):
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2)
            smoothed.append(np.mean(data[start:end]))
        return smoothed
    
    rewards_smooth = smooth_curve(rewards)
    losses_smooth = smooth_curve(losses)
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(rewards_smooth, color='blue', linewidth=2, label='Smoothed')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    ax2.plot(losses, alpha=0.3, color='red', label='Raw')
    ax2.plot(losses_smooth, color='red', linewidth=2, label='Smoothed')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_query_results(results, title="Query Evaluation Results"):
    """Plot query answering performance"""
    
    # Count available metrics
    available_metrics = []
    if 'value_errors' in results:
        available_metrics.append('value_errors')
    if 'policy_accuracy' in results:
        available_metrics.append('policy_accuracy')
    if 'qvalue_errors' in results:
        available_metrics.append('qvalue_errors')
    if 'comparison_accuracy' in results:
        available_metrics.append('comparison_accuracy')
    
    if not available_metrics:
        print("No results to plot!")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    metric_idx = 0
    
    # Value errors
    if 'value_errors' in available_metrics:
        values = results['value_errors']
        axes[metric_idx].hist(values, bins=20, alpha=0.7, color='blue')
        axes[metric_idx].set_title('Value Prediction Errors')
        axes[metric_idx].set_xlabel('Absolute Error')
        axes[metric_idx].axvline(np.mean(values), color='red', 
                               linestyle='--', label=f'Mean: {np.mean(values):.4f}')
        axes[metric_idx].legend()
        metric_idx += 1
    
    # Policy accuracy
    if 'policy_accuracy' in available_metrics:
        accuracy = np.mean(results['policy_accuracy'])
        axes[metric_idx].bar(['Correct', 'Incorrect'], 
                           [accuracy, 1-accuracy], color=['green', 'red'], alpha=0.7)
        axes[metric_idx].set_title(f'Policy Accuracy: {accuracy:.3f}')
        axes[metric_idx].set_ylabel('Proportion')
        metric_idx += 1
    
    # Q-value errors  
    if 'qvalue_errors' in available_metrics:
        values = results['qvalue_errors']
        axes[metric_idx].hist(values, bins=20, alpha=0.7, color='purple')
        axes[metric_idx].set_title('Q-Value Prediction Errors')
        axes[metric_idx].set_xlabel('Absolute Error')
        axes[metric_idx].axvline(np.mean(values), color='red',
                               linestyle='--', label=f'Mean: {np.mean(values):.4f}')
        axes[metric_idx].legend()
        metric_idx += 1
    
    # Comparison accuracy
    if 'comparison_accuracy' in available_metrics:
        accuracy = np.mean(results['comparison_accuracy'])
        axes[metric_idx].bar(['Correct', 'Incorrect'],
                           [accuracy, 1-accuracy], color=['green', 'red'], alpha=0.7)
        axes[metric_idx].set_title(f'Comparison Accuracy: {accuracy:.3f}')
        axes[metric_idx].set_ylabel('Proportion')
        metric_idx += 1
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# =================== EXPERIMENT UTILITIES ===================

class ExperimentTracker:
    """Track experimental results across runs"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.results = defaultdict(list)
        self.metadata = {}
        self.start_time = time.time()
        
    def log_result(self, key, value):
        """Log a single result"""
        self.results[key].append(value)
    
    def log_metadata(self, **kwargs):
        """Log experiment metadata"""
        self.metadata.update(kwargs)
        self.metadata['timestamp'] = time.time()
    
    def get_summary_stats(self):
        """Get summary statistics for all metrics"""
        summary = {}
        for key, values in self.results.items():
            if isinstance(values[0], (int, float, np.number)):
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_min"] = np.min(values)
                summary[f"{key}_max"] = np.max(values)
        return summary
    
    def save(self, filename=None):
        """Save results to pickle file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"{self.experiment_name}_results_{timestamp}.pkl"
        
        data = {
            'experiment_name': self.experiment_name,
            'results': dict(self.results),
            'metadata': self.metadata,
            'summary_stats': self.get_summary_stats(),
            'duration': time.time() - self.start_time
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Results saved to {filename}")
        return filename
    
    def load(self, filename):
        """Load results from pickle file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.experiment_name = data['experiment_name']
        self.results = defaultdict(list, data['results'])
        self.metadata = data['metadata']
        
        print(f"Results loaded from {filename}")
        return data

def compare_experiments(trackers, metrics, title="Experiment Comparison"):
    """Compare results across multiple experiments"""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        for tracker in trackers:
            if metric in tracker.results:
                values = tracker.results[metric]
                axes[i].plot(values, label=tracker.experiment_name, marker='o', linewidth=2)
        
        axes[i].set_title(f'{metric}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Measurement')
        axes[i].set_ylabel('Value')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def print_experiment_summary(tracker):
    """Print a summary of experiment results"""
    print("\\n" + "="*50)
    print(f"EXPERIMENT SUMMARY: {tracker.experiment_name}")
    print("="*50)
    
    print(f"Duration: {tracker.metadata.get('duration', 0):.2f} seconds")
    
    summary = tracker.get_summary_stats()
    if summary:
        print("\\nKey Results:")
        for key, value in summary.items():
            if '_mean' in key:
                base_key = key.replace('_mean', '')
                std_key = base_key + '_std'
                std_value = summary.get(std_key, 0)
                print(f"  {base_key:20}: {value:.4f} ± {std_value:.4f}")
    
    print("\\nMetadata:")
    for key, value in tracker.metadata.items():
        if key != 'duration':
            print(f"  {key:20}: {value}")

# =================== UTILITY FUNCTIONS ===================

def create_test_environment(size=5, n_obstacles=2, seed=42):
    """Create a simple test environment for quick testing"""
    return GridWorld(size=size, n_obstacles=n_obstacles, seed=seed)

def quick_test():
    """Quick test of all major components"""
    print("Running quick test of helper functions...")
    
    # Test environment
    env = create_test_environment()
    print(f"✓ Created test environment: {env.size}x{env.size} gridworld")
    
    # Test model creation
    model = StandardDQN(state_dim=1, action_dim=4, hidden_dim=32)
    model.set_state_space_size(env.observation_space.n)
    print("✓ Created standard DQN model")
    
    # Test query generation
    query_gen = QueryGenerator(env)
    queries = query_gen.generate_point_queries(n_queries=5)
    print(f"✓ Generated {len(queries)} test queries")
    
    # Test ground truth computation
    ground_truth = compute_ground_truth_values(env, max_iterations=100)
    print("✓ Computed ground truth values")
    
    # Test training (very short)
    rewards, losses = train_dqn(env, model, episodes=10)
    print("✓ Completed test training")
    
    # Test evaluation
    results = evaluate_query_answering(model, env, queries, ground_truth)
    print("✓ Completed test evaluation")
    
    print("\\nAll tests passed! Helper functions are ready to use.")
    return env, model, queries, ground_truth, results

def demonstrate_usage():
    """Demonstrate typical usage of helper functions"""
    print("\\n" + "="*50)
    print("HELPER FUNCTIONS USAGE DEMONSTRATION")
    print("="*50)
    
    print("\\n1. ENVIRONMENT SETUP:")
    print("   env = GridWorld(size=8, n_obstacles=5)")
    print("   env.render()  # Visualize the environment")
    
    print("\\n2. MODEL SETUP:")
    print("   model = StandardDQN(state_dim=1, action_dim=4)")
    print("   model.set_state_space_size(env.observation_space.n)")
    
    print("\\n3. TRAINING:")
    print("   rewards, losses = train_dqn(env, model, episodes=1000)")
    print("   plot_training_curves(rewards, losses)")
    
    print("\\n4. GROUND TRUTH:")
    print("   ground_truth = compute_ground_truth_values(env)")
    
    print("\\n5. QUERY GENERATION:")
    print("   query_gen = QueryGenerator(env)")
    print("   queries = query_gen.generate_point_queries(100)")
    
    print("\\n6. EVALUATION:")
    print("   results = evaluate_query_answering(model, env, queries, ground_truth)")
    print("   plot_query_results(results)")
    
    print("\\n7. EXPERIMENT TRACKING:")
    print("   tracker = ExperimentTracker('my_experiment')")
    print("   tracker.log_result('accuracy', 0.85)")
    print("   tracker.save()")

def get_model_info(model):
    """Get information about a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")

def analyze_environment(env):
    """Analyze environment properties"""
    print(f"Environment: {env.__class__.__name__}")
    print(f"State space size: {env.observation_space.n}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Number of obstacles: {len(env.obstacles)}")
    print(f"Start state: {env.start_state}")
    print(f"Goal state: {env.goal_state}")
    
    # Analyze reward structure
    unique_rewards = np.unique(env.rewards)
    print(f"Unique rewards: {unique_rewards}")
    for reward in unique_rewards:
        count = np.sum(env.rewards == reward)
        print(f"  {reward:6.3f}: {count:3d} states ({count/len(env.rewards)*100:.1f}%)")

def save_experiment_config(config, filename):
    """Save experiment configuration"""
    with open(filename, 'wb') as f:
        pickle.dump(config, f)
    print(f"Configuration saved to {filename}")

def load_experiment_config(filename):
    """Load experiment configuration"""
    with open(filename, 'rb') as f:
        config = pickle.load(f)
    print(f"Configuration loaded from {filename}")
    return config

# =================== ADVANCED UTILITIES ===================

def batch_evaluate_queries(model, queries, ground_truth, batch_size=32):
    """Evaluate queries in batches for efficiency"""
    results = defaultdict(list)
    V_true, Q_true, policy_true = ground_truth
    
    # Group queries by type for efficient processing
    query_groups = defaultdict(list)
    for query in queries:
        query_groups[query.type].append(query)
    
    for query_type, type_queries in query_groups.items():
        print(f"Processing {len(type_queries)} {query_type} queries...")
        
        # Process in batches
        for i in range(0, len(type_queries), batch_size):
            batch = type_queries[i:i+batch_size]
            batch_results = evaluate_query_answering(model, None, batch, ground_truth)
            
            # Merge results
            for key, values in batch_results.items():
                results[key].extend(values)
    
    return results

def cross_validate_model(env, model_class, model_params, k_folds=5, episodes=500):
    """Perform k-fold cross validation on model performance"""
    
    # Create different environment configurations for cross-validation
    fold_results = []
    
    for fold in range(k_folds):
        print(f"\\nFold {fold + 1}/{k_folds}")
        
        # Create environment with different random seed
        fold_env = GridWorld(
            size=env.size,
            n_obstacles=env.n_obstacles,
            seed=42 + fold
        )
        
        # Create and train model
        fold_model = model_class(**model_params)
        fold_model.set_state_space_size(fold_env.observation_space.n)
        
        rewards, losses = train_dqn(fold_env, fold_model, episodes=episodes)
        
        # Evaluate on test queries
        query_gen = QueryGenerator(fold_env)
        queries = query_gen.generate_point_queries(50)
        ground_truth = compute_ground_truth_values(fold_env)
        results = evaluate_query_answering(fold_model, fold_env, queries, ground_truth)
        
        fold_results.append({
            'final_reward': np.mean(rewards[-50:]),
            'value_error': np.mean(results.get('value_errors', [0])),
            'policy_accuracy': np.mean(results.get('policy_accuracy', [0]))
        })
    
    # Compute cross-validation statistics
    cv_results = {}
    for metric in fold_results[0].keys():
        values = [fold[metric] for fold in fold_results]
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
        cv_results[f'{metric}_values'] = values
    
    print("\\nCross-validation results:")
    for metric in ['final_reward', 'value_error', 'policy_accuracy']:
        mean_val = cv_results[f'{metric}_mean']
        std_val = cv_results[f'{metric}_std']
        print(f"  {metric:15}: {mean_val:.4f} ± {std_val:.4f}")
    
    return cv_results

def hyperparameter_search(env, param_grid, episodes=500, n_trials=3):
    """Simple grid search for hyperparameters"""
    
    best_params = None
    best_score = -float('inf')
    results = []
    
    # Generate parameter combinations
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for param_combo in itertools.product(*param_values):
        params = dict(zip(param_names, param_combo))
        print(f"\\nTesting parameters: {params}")
        
        trial_scores = []
        
        for trial in range(n_trials):
            # Create model with these parameters
            model = StandardDQN(
                state_dim=1,
                action_dim=env.action_space.n,
                hidden_dim=params.get('hidden_dim', 64)
            )
            model.set_state_space_size(env.observation_space.n)
            
            # Train with these parameters
            rewards, losses = train_dqn(
                env, model, episodes=episodes,
                lr=params.get('lr', 0.001),
                gamma=params.get('gamma', 0.99)
            )
            
            # Score based on final performance
            score = np.mean(rewards[-50:])
            trial_scores.append(score)
        
        avg_score = np.mean(trial_scores)
        results.append({
            'params': params,
            'score': avg_score,
            'std': np.std(trial_scores)
        })
        
        print(f"  Average score: {avg_score:.4f} ± {np.std(trial_scores):.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    print(f"\\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return best_params, results

def profile_inference_speed(model, env, n_queries=1000):
    """Profile the inference speed of different query types"""
    query_gen = QueryGenerator(env)
    
    # Generate different types of queries
    point_queries = query_gen.generate_point_queries(n_queries)
    
    # Time different query types
    times = {}
    
    # Point queries
    start_time = time.time()
    for query in point_queries:
        state = query.parameters['state']
        with torch.no_grad():
            _ = model(state)
    point_time = time.time() - start_time
    times['point_queries'] = point_time / len(point_queries) * 1000  # ms per query
    
    print(f"Inference speed profiling ({n_queries} queries):")
    print(f"  Point queries: {times['point_queries']:.3f} ms per query")
    
    return times

# Initialize message
print("="*60)
print("DETERMINISTIC RL INFERENCE RESEARCH - HELPER FUNCTIONS")
print("="*60)
print("Helper functions loaded successfully!")
print("\\nAvailable utilities:")
print("- GridWorld environment")
print("- StandardDQN and QueryConditionedDQN models") 
print("- Query generation and evaluation")
print("- Training and visualization functions")
print("- Experiment tracking and analysis")
print("- Cross-validation and hyperparameter search")
print("\\nQuick test: run quick_test()")
print("Usage demo: run demonstrate_usage()")
print("\\nReady for experiments!")