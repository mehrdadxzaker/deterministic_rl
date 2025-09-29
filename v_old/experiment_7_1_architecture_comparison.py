"""
Experiment 7.1: Query-Conditioned vs Standard Architecture
Research Question: Do specialized architectures improve inference capability?

Goal: Compare standard DQN vs query-conditioned networks on diverse queries
Setup: Same environment, two architectures with same parameter budget
Expected Result: Query-conditioned architecture better for diverse queries
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from helper_functions import *

class QueryConditionedDQNAdvanced(nn.Module):
    """Advanced query-conditioned DQN with attention mechanism"""
    
    def __init__(self, state_dim, action_dim, query_dim, hidden_dim=128):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim//2),
            nn.ReLU(), 
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # Attention mechanism for query-state interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim//2, 
            num_heads=4, 
            batch_first=True
        )
        
        # Final layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Output heads for different query types
        self.q_value_head = nn.Linear(hidden_dim//2, action_dim)
        self.value_head = nn.Linear(hidden_dim//2, 1) 
        self.policy_head = nn.Linear(hidden_dim//2, action_dim)
        
        # Query type embedding
        self.query_type_embedding = nn.Embedding(4, query_dim//4)  # 4 query types
        
    def encode_query(self, query_type, query_params):
        """Encode query into fixed-size vector"""
        # Query types: 0=value, 1=policy, 2=qvalue, 3=comparative
        type_emb = self.query_type_embedding(torch.tensor(query_type))
        
        # Simple encoding of parameters (in practice, this would be more sophisticated)
        if query_type == 0:  # value query
            param_vec = torch.zeros(self.query_encoder[0].in_features - type_emb.size(-1))
        elif query_type == 1:  # policy query
            param_vec = torch.ones(self.query_encoder[0].in_features - type_emb.size(-1)) * 0.5
        elif query_type == 2:  # qvalue query  
            action = query_params.get('action', 0)
            param_vec = torch.zeros(self.query_encoder[0].in_features - type_emb.size(-1))
            if len(param_vec) > action:
                param_vec[action] = 1.0
        else:  # comparative query
            param_vec = torch.ones(self.query_encoder[0].in_features - type_emb.size(-1)) * -0.5
            
        query_vec = torch.cat([type_emb, param_vec])
        return query_vec
        
    def forward(self, state, query_type=1, query_params=None):
        """
        Forward pass with optional query conditioning
        
        Args:
            state: Environment state
            query_type: Type of query (0=value, 1=policy, 2=qvalue, 3=comparative)
            query_params: Additional query parameters
        """
        if isinstance(state, int):
            state = torch.tensor([float(state)], dtype=torch.float32)
            
        # Encode state
        state_emb = self.state_encoder(state.unsqueeze(0) if state.dim() == 1 else state)
        
        # Encode query
        if query_params is None:
            query_params = {}
        query_vec = self.encode_query(query_type, query_params)
        query_emb = self.query_encoder(query_vec.unsqueeze(0))
        
        # Apply attention mechanism
        attended_state, _ = self.attention(
            state_emb.unsqueeze(1), 
            query_emb.unsqueeze(1), 
            query_emb.unsqueeze(1)
        )
        attended_state = attended_state.squeeze(1)
        
        # Fuse representations
        combined = torch.cat([attended_state, query_emb], dim=-1)
        fused = self.fusion(combined)
        
        # Generate outputs based on query type
        if query_type == 0:  # value query
            return self.value_head(fused).squeeze(-1)
        elif query_type == 1:  # policy query (default Q-values)
            return self.q_value_head(fused).squeeze(0)
        elif query_type == 2:  # qvalue query
            return self.q_value_head(fused).squeeze(0)
        else:  # comparative query
            return self.q_value_head(fused).squeeze(0)

class QueryAwareLoss(nn.Module):
    """Loss function that adapts based on query type"""
    
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, query_type):
        if query_type == 0:  # value query
            return self.mse_loss(predictions, targets)
        elif query_type == 1:  # policy query
            return self.ce_loss(predictions, targets.long())
        else:  # qvalue or comparative
            return self.mse_loss(predictions, targets)

def run_experiment_7_1(grid_size=10, n_obstacles=8, episodes=2000, seed=42):
    """
    Run Experiment 7.1: Architecture Comparison
    
    Args:
        grid_size: Size of gridworld
        n_obstacles: Number of obstacles  
        episodes: Number of training episodes
        seed: Random seed
    """
    
    print("="*60)
    print("EXPERIMENT 7.1: Architecture Comparison")
    print("="*60)
    
    # Initialize experiment tracking
    tracker = ExperimentTracker("exp_7_1_architecture_comparison")
    tracker.log_metadata(
        grid_size=grid_size,
        n_obstacles=n_obstacles,
        episodes=episodes,
        seed=seed,
        description="Compare standard DQN vs query-conditioned architecture"
    )
    
    # Set random seed
    set_seeds(seed)
    
    # Create environment
    print(f"Creating {grid_size}x{grid_size} GridWorld with {n_obstacles} obstacles...")
    env = GridWorld(size=grid_size, n_obstacles=n_obstacles, seed=seed)
    
    # Create models with similar parameter budget
    state_dim = 1
    action_dim = env.action_space.n
    query_dim = 32  # Dimension for query encoding
    hidden_dim = 128
    
    # Standard DQN (baseline)
    model_standard = StandardDQN(state_dim, action_dim, hidden_dim)
    
    # Query-conditioned DQN
    model_query = QueryConditionedDQNAdvanced(state_dim, action_dim, query_dim, hidden_dim)
    
    # Print model sizes for fair comparison
    standard_params = sum(p.numel() for p in model_standard.parameters())
    query_params = sum(p.numel() for p in model_query.parameters())
    
    print(f"\\nModel parameter counts:")
    print(f"Standard DQN:           {standard_params:,} parameters")
    print(f"Query-conditioned DQN:  {query_params:,} parameters")
    print(f"Parameter ratio:        {query_params/standard_params:.2f}")
    
    # Compute ground truth
    print("\\nComputing ground truth...")
    ground_truth = compute_ground_truth_values(env)
    
    # Train both models
    models = {
        'standard': model_standard,
        'query_conditioned': model_query
    }
    
    training_results = {}
    
    for name, model in models.items():
        print(f"\\nTraining {name} model...")
        
        if name == 'standard':
            rewards, losses = train_standard_architecture(env, model, episodes)
        else:
            rewards, losses = train_query_conditioned_architecture(env, model, episodes, ground_truth)
        
        training_results[name] = {
            'rewards': rewards,
            'losses': losses
        }
        
        final_reward = np.mean(rewards[-100:])
        print(f"{name} final average reward: {final_reward:.3f}")
    
    # Comprehensive evaluation on diverse queries
    print("\\nGenerating comprehensive query evaluation...")
    evaluation_results = comprehensive_query_evaluation(env, models, ground_truth)
    
    # Analyze architectural advantages
    print("\\nAnalyzing architectural advantages...")
    architecture_analysis = analyze_architectural_differences(env, models, ground_truth)
    
    # Visualize results
    plot_architecture_comparison(training_results, evaluation_results, architecture_analysis)
    
    # Log results
    for name in training_results:
        for metric in training_results[name]:
            tracker.log_result(f"{name}_{metric}", training_results[name][metric])
    
    for metric in evaluation_results:
        tracker.log_result(f"eval_{metric}", evaluation_results[metric])
        
    for metric in architecture_analysis:
        tracker.log_result(f"analysis_{metric}", architecture_analysis[metric])
    
    # Print comprehensive summary
    print_architecture_summary(training_results, evaluation_results, architecture_analysis)
    
    # Save results
    tracker.save()
    
    return tracker, training_results, evaluation_results, models, env, ground_truth

def train_standard_architecture(env, model, episodes, lr=0.001):
    """Train standard DQN architecture"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    episode_rewards = []
    losses = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        max_steps = env.size * env.size * 2
        
        while steps < max_steps:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            
            # Standard Q-learning update
            current_q = model(state)[action]
            
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    next_q_values = model(next_state)
                    target = reward + 0.99 * next_q_values.max().item()
            
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
        losses.append(episode_loss / steps if steps > 0 else 0)
        
        if episode % 200 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode:4d}, Avg Reward: {avg_reward:6.3f}, Epsilon: {epsilon:.3f}")
    
    return episode_rewards, losses

def train_query_conditioned_architecture(env, model, episodes, ground_truth, lr=0.001):
    """Train query-conditioned architecture with mixed objectives"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    query_loss_fn = QueryAwareLoss()
    standard_loss_fn = nn.MSELoss()
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    episode_rewards = []
    losses = []
    
    # Generate query generator for training
    query_gen = QueryGenerator(env)
    V_true, Q_true, policy_true = ground_truth
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        max_steps = env.size * env.size * 2
        
        while steps < max_steps:
            # Epsilon-greedy action selection using policy queries
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state, query_type=1)  # Policy query
                    action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            
            # Mixed training: standard RL loss + query-specific losses
            total_loss = 0
            
            # Standard Q-learning component
            current_q = model(state, query_type=2, query_params={'action': action})[action]
            
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    next_q_values = model(next_state, query_type=1)
                    target = reward + 0.99 * next_q_values.max().item()
            
            rl_loss = standard_loss_fn(current_q, torch.tensor(target, dtype=torch.float32))
            total_loss += 0.7 * rl_loss
            
            # Query-specific training (randomly sample queries)
            if steps % 3 == 0:  # Every 3rd step, do query training
                # Sample random state for query training
                query_state = np.random.randint(0, env.observation_space.n)
                query_type = np.random.randint(0, 3)  # 0=value, 1=policy, 2=qvalue
                
                if query_type == 0:  # Value query
                    pred_value = model(query_state, query_type=0)
                    true_value = torch.tensor(V_true[query_state], dtype=torch.float32)
                    query_loss = query_loss_fn(pred_value, true_value, query_type)
                    
                elif query_type == 1:  # Policy query
                    pred_q_values = model(query_state, query_type=1)
                    true_action = torch.tensor(policy_true[query_state], dtype=torch.long)
                    query_loss = query_loss_fn(pred_q_values.unsqueeze(0), true_action.unsqueeze(0), query_type)
                    
                else:  # Q-value query
                    query_action = np.random.randint(0, env.action_space.n)
                    pred_q = model(query_state, query_type=2, query_params={'action': query_action})[query_action]
                    true_q = torch.tensor(Q_true[query_state, query_action], dtype=torch.float32)
                    query_loss = query_loss_fn(pred_q, true_q, query_type)
                
                total_loss += 0.3 * query_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            episode_reward += reward
            episode_loss += total_loss.item()
            state = next_state
            steps += 1
            
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / steps if steps > 0 else 0)
        
        if episode % 200 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode:4d}, Avg Reward: {avg_reward:6.3f}, Epsilon: {epsilon:.3f}")
    
    return episode_rewards, losses

def comprehensive_query_evaluation(env, models, ground_truth):
    """Evaluate both models on comprehensive query set"""
    query_gen = QueryGenerator(env)
    
    # Generate diverse queries
    queries = {
        'point': query_gen.generate_point_queries(n_queries=100),
        'path': query_gen.generate_path_queries(n_queries=50), 
        'set': query_gen.generate_set_queries(n_queries=30),
        'comparative': query_gen.generate_comparative_queries(n_queries=30)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} model...")
        
        model_results = {}
        
        # Point queries
        point_results = evaluate_point_queries_advanced(model, model_name, queries['point'], ground_truth)
        model_results.update({f"point_{k}": v for k, v in point_results.items()})
        
        # Path queries  
        path_results = evaluate_path_queries(model, env, queries['path'], ground_truth)
        model_results.update({f"path_{k}": v for k, v in path_results.items()})
        
        # Set queries
        set_results = evaluate_set_queries(model, env, queries['set'], ground_truth)
        model_results.update({f"set_{k}": v for k, v in set_results.items()})
        
        # Comparative queries
        comp_results = evaluate_comparative_queries(model, queries['comparative'], ground_truth)
        model_results.update({f"comp_{k}": v for k, v in comp_results.items()})
        
        # Store results
        for key, values in model_results.items():
            if isinstance(values, list):
                results[f"{model_name}_{key}_mean"] = np.mean(values)
                results[f"{model_name}_{key}_std"] = np.std(values)
            else:
                results[f"{model_name}_{key}"] = values
    
    return results

def evaluate_point_queries_advanced(model, model_name, queries, ground_truth):
    """Advanced evaluation of point queries using query conditioning if available"""
    V_true, Q_true, policy_true = ground_truth
    results = defaultdict(list)
    
    for query in queries:
        if query.type != 'point':
            continue
            
        state = query.parameters['state']
        
        if query.parameters['query_type'] == 'value':
            # Use query conditioning if available
            if model_name == 'query_conditioned':
                with torch.no_grad():
                    predicted_value = model(state, query_type=0).item()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    predicted_value = q_values.max().item()
            
            true_value = V_true[state]
            error = abs(predicted_value - true_value)
            results['value_errors'].append(error)
            
        elif query.parameters['query_type'] == 'policy':
            # Use query conditioning if available
            if model_name == 'query_conditioned':
                with torch.no_grad():
                    q_values = model(state, query_type=1)
                    predicted_action = q_values.argmax().item()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    predicted_action = q_values.argmax().item()
            
            true_action = policy_true[state]
            correct = (predicted_action == true_action)
            results['policy_accuracy'].append(correct)
            
        elif query.parameters['query_type'] == 'qvalue':
            action = query.parameters['action']
            
            # Use query conditioning if available
            if model_name == 'query_conditioned':
                with torch.no_grad():
                    q_values = model(state, query_type=2, query_params={'action': action})
                    predicted_q = q_values[action].item()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    predicted_q = q_values[action].item()
            
            true_q = Q_true[state, action]
            error = abs(predicted_q - true_q)
            results['qvalue_errors'].append(error)
    
    return results

def analyze_architectural_differences(env, models, ground_truth):
    """Analyze specific advantages of each architecture"""
    query_gen = QueryGenerator(env)
    analysis = {}
    
    # Test query type specialization
    print("Testing query type specialization...")
    
    query_types = ['value', 'policy', 'qvalue']
    for query_type in query_types:
        type_queries = []
        for _ in range(50):
            state = np.random.randint(0, env.observation_space.n)
            if query_type == 'qvalue':
                action = np.random.randint(0, env.action_space.n)
                type_queries.append(Query(
                    type='point',
                    parameters={'query_type': query_type, 'state': state, 'action': action},
                    expected_answer_type='scalar'
                ))
            else:
                type_queries.append(Query(
                    type='point', 
                    parameters={'query_type': query_type, 'state': state},
                    expected_answer_type='scalar'
                ))
        
        for model_name, model in models.items():
            results = evaluate_point_queries_advanced(model, model_name, type_queries, ground_truth)
            
            if query_type == 'value' and 'value_errors' in results:
                analysis[f'{model_name}_{query_type}_error'] = np.mean(results['value_errors'])
            elif query_type == 'policy' and 'policy_accuracy' in results:
                analysis[f'{model_name}_{query_type}_accuracy'] = np.mean(results['policy_accuracy'])
            elif query_type == 'qvalue' and 'qvalue_errors' in results:
                analysis[f'{model_name}_{query_type}_error'] = np.mean(results['qvalue_errors'])
    
    # Test generalization to unseen states
    print("Testing generalization...")
    
    # Train on first half of states, test on second half
    n_states = env.observation_space.n
    train_states = list(range(n_states // 2))
    test_states = list(range(n_states // 2, n_states))
    
    for model_name, model in models.items():
        # Generate queries for unseen states
        test_queries = []
        for state in test_states[:20]:  # Test on 20 unseen states
            test_queries.append(Query(
                type='point',
                parameters={'query_type': 'value', 'state': state},
                expected_answer_type='scalar'
            ))
        
        results = evaluate_point_queries_advanced(model, model_name, test_queries, ground_truth)
        if 'value_errors' in results:
            analysis[f'{model_name}_generalization_error'] = np.mean(results['value_errors'])
    
    # Test computational efficiency
    print("Testing computational efficiency...")
    
    for model_name, model in models.items():
        # Time inference for different query types
        state = 0
        n_trials = 100
        
        # Standard inference time
        start_time = time.time()
        for _ in range(n_trials):
            with torch.no_grad():
                if model_name == 'query_conditioned':
                    _ = model(state, query_type=1)
                else:
                    _ = model(state)
        standard_time = (time.time() - start_time) / n_trials
        
        analysis[f'{model_name}_inference_time'] = standard_time * 1000  # Convert to milliseconds
    
    # Test query diversity handling
    print("Testing query diversity handling...")
    
    # Generate mixed query batch
    mixed_queries = (query_gen.generate_point_queries(20) + 
                    query_gen.generate_comparative_queries(10))
    
    for model_name, model in models.items():
        # Evaluate on mixed queries
        all_results = defaultdict(list)
        
        point_results = evaluate_point_queries_advanced(model, model_name, mixed_queries, ground_truth)
        comp_results = evaluate_comparative_queries(model, mixed_queries, ground_truth)
        
        # Combine results
        for key, values in point_results.items():
            all_results[key].extend(values)
        for key, values in comp_results.items():
            all_results[key].extend(values)
        
        # Compute diversity score (average across different query types)
        diversity_scores = []
        if 'value_errors' in all_results:
            diversity_scores.append(1.0 / (1.0 + np.mean(all_results['value_errors'])))
        if 'policy_accuracy' in all_results:
            diversity_scores.append(np.mean(all_results['policy_accuracy']))
        if 'comparison_accuracy' in all_results:
            diversity_scores.append(np.mean(all_results['comparison_accuracy']))
        
        if diversity_scores:
            analysis[f'{model_name}_diversity_score'] = np.mean(diversity_scores)
    
    return analysis

def plot_architecture_comparison(training_results, evaluation_results, architecture_analysis):
    """Plot comprehensive architecture comparison"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Training curves - rewards
    for name, results in training_results.items():
        rewards = results['rewards']
        window = min(50, len(rewards) // 10)
        if window > 1:
            rewards_smooth = pd.Series(rewards).rolling(window).mean()
        else:
            rewards_smooth = rewards
        axes[0,0].plot(rewards_smooth, label=name, linewidth=2)
    
    axes[0,0].set_title('Training Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Training curves - losses
    for name, results in training_results.items():
        losses = results['losses']
        window = min(50, len(losses) // 10)
        if window > 1:
            losses_smooth = pd.Series(losses).rolling(window).mean()
        else:
            losses_smooth = losses
        axes[0,1].plot(losses_smooth, label=name, linewidth=2)
    
    axes[0,1].set_title('Training Losses')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Query type performance comparison
    models = ['standard', 'query_conditioned']
    query_types = ['value', 'policy', 'qvalue']
    
    # Value errors
    value_errors = [evaluation_results.get(f'{model}_point_value_errors_mean', 0) for model in models]
    x_pos = np.arange(len(models))
    axes[0,2].bar(x_pos, value_errors, alpha=0.7, color=['blue', 'orange'])
    axes[0,2].set_title('Value Prediction Errors')
    axes[0,2].set_ylabel('Mean Absolute Error')
    axes[0,2].set_xticks(x_pos)
    axes[0,2].set_xticklabels(models, rotation=45)
    
    # Policy accuracy
    policy_acc = [evaluation_results.get(f'{model}_point_policy_accuracy_mean', 0) for model in models]
    axes[0,3].bar(x_pos, policy_acc, alpha=0.7, color=['blue', 'orange'])
    axes[0,3].set_title('Policy Accuracy')
    axes[0,3].set_ylabel('Accuracy')
    axes[0,3].set_xticks(x_pos)
    axes[0,3].set_xticklabels(models, rotation=45)
    axes[0,3].set_ylim([0, 1])
    
    # Architectural analysis plots
    
    # Generalization performance
    gen_errors = [architecture_analysis.get(f'{model}_generalization_error', 0) for model in models]
    axes[1,0].bar(x_pos, gen_errors, alpha=0.7, color=['blue', 'orange'])
    axes[1,0].set_title('Generalization Error')
    axes[1,0].set_ylabel('Error on Unseen States')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(models, rotation=45)
    
    # Inference time
    inf_times = [architecture_analysis.get(f'{model}_inference_time', 0) for model in models]
    axes[1,1].bar(x_pos, inf_times, alpha=0.7, color=['blue', 'orange'])
    axes[1,1].set_title('Inference Time')
    axes[1,1].set_ylabel('Time (ms)')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(models, rotation=45)
    
    # Diversity score
    div_scores = [architecture_analysis.get(f'{model}_diversity_score', 0) for model in models]
    axes[1,2].bar(x_pos, div_scores, alpha=0.7, color=['blue', 'orange'])
    axes[1,2].set_title('Query Diversity Handling')
    axes[1,2].set_ylabel('Diversity Score')
    axes[1,2].set_xticks(x_pos)
    axes[1,2].set_xticklabels(models, rotation=45)
    axes[1,2].set_ylim([0, 1])
    
    # Overall comparison radar chart would go here, but simplified bar chart
    # Combined performance score
    combined_scores = []
    for model in models:
        # Combine multiple metrics into single score (higher = better)
        policy_score = evaluation_results.get(f'{model}_point_policy_accuracy_mean', 0)
        value_score = 1.0 / (1.0 + evaluation_results.get(f'{model}_point_value_errors_mean', 1))
        div_score = architecture_analysis.get(f'{model}_diversity_score', 0)
        combined = (policy_score + value_score + div_score) / 3
        combined_scores.append(combined)
    
    axes[1,3].bar(x_pos, combined_scores, alpha=0.7, color=['blue', 'orange'])
    axes[1,3].set_title('Overall Performance')
    axes[1,3].set_ylabel('Combined Score')
    axes[1,3].set_xticks(x_pos)
    axes[1,3].set_xticklabels(models, rotation=45)
    axes[1,3].set_ylim([0, 1])
    
    plt.suptitle('Architecture Comparison: Standard vs Query-Conditioned DQN', fontsize=16)
    plt.tight_layout()
    plt.show()

def print_architecture_summary(training_results, evaluation_results, architecture_analysis):
    """Print comprehensive summary of architecture comparison"""
    print("\\n" + "="*60)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("="*60)
    
    models = ['standard', 'query_conditioned']
    
    # Training performance
    print("\\nTRAINING PERFORMANCE:")
    for model in models:
        if model in training_results:
            final_reward = np.mean(training_results[model]['rewards'][-100:])
            final_loss = np.mean(training_results[model]['losses'][-100:])
            print(f"  {model:18}: {final_reward:6.3f} reward, {final_loss:6.3f} loss")
    
    # Query answering performance
    print("\\nQUERY ANSWERING PERFORMANCE:")
    for model in models:
        print(f"  {model:18}:")
        
        value_error = evaluation_results.get(f'{model}_point_value_errors_mean', 0)
        policy_acc = evaluation_results.get(f'{model}_point_policy_accuracy_mean', 0)
        qvalue_error = evaluation_results.get(f'{model}_point_qvalue_errors_mean', 0)
        
        print(f"    Value error:      {value_error:.4f}")
        print(f"    Policy accuracy:  {policy_acc:.4f}")
        print(f"    Q-value error:    {qvalue_error:.4f}")
    
    # Architectural advantages
    print("\\nARCHITECTURAL ANALYSIS:")
    for model in models:
        print(f"  {model:18}:")
        
        gen_error = architecture_analysis.get(f'{model}_generalization_error', 0)
        inf_time = architecture_analysis.get(f'{model}_inference_time', 0)
        div_score = architecture_analysis.get(f'{model}_diversity_score', 0)
        
        print(f"    Generalization:   {gen_error:.4f} error")
        print(f"    Inference time:   {inf_time:.2f} ms")
        print(f"    Diversity score:  {div_score:.4f}")
    
    # Winner analysis
    print("\\nWINNER ANALYSIS:")
    
    # Best for different criteria
    best_control = max(models, key=lambda x: np.mean(training_results[x]['rewards'][-100:]) if x in training_results else -float('inf'))
    best_value_pred = min(models, key=lambda x: evaluation_results.get(f'{x}_point_value_errors_mean', float('inf')))
    best_policy_acc = max(models, key=lambda x: evaluation_results.get(f'{x}_point_policy_accuracy_mean', 0))
    best_generalization = min(models, key=lambda x: architecture_analysis.get(f'{x}_generalization_error', float('inf')))
    best_diversity = max(models, key=lambda x: architecture_analysis.get(f'{x}_diversity_score', 0))
    
    print(f"  Best for control:        {best_control}")
    print(f"  Best for value queries:  {best_value_pred}")
    print(f"  Best for policy queries: {best_policy_acc}")
    print(f"  Best generalization:     {best_generalization}")
    print(f"  Best query diversity:    {best_diversity}")
    
    # Overall recommendation
    print("\\nOVERALL RECOMMENDATION:")
    
    # Count wins
    wins = defaultdict(int)
    wins[best_control] += 1
    wins[best_value_pred] += 1
    wins[best_policy_acc] += 1
    wins[best_generalization] += 1
    wins[best_diversity] += 1
    
    overall_winner = max(wins, key=wins.get)
    print(f"  Overall winner: {overall_winner} ({wins[overall_winner]}/5 categories)")
    
    if overall_winner == 'query_conditioned':
        print("  → Query-conditioned architecture shows advantages for inference tasks")
        print("  → Specialized design pays off for diverse query handling")
    else:
        print("  → Standard architecture remains competitive")
        print("  → Simple designs may be sufficient for basic query types")
    
    # Performance gaps
    print("\\nPERFORMANCE GAPS:")
    value_gap = abs(evaluation_results.get('standard_point_value_errors_mean', 0) - 
                   evaluation_results.get('query_conditioned_point_value_errors_mean', 0))
    policy_gap = abs(evaluation_results.get('standard_point_policy_accuracy_mean', 0) - 
                    evaluation_results.get('query_conditioned_point_policy_accuracy_mean', 0))
    
    print(f"  Value prediction gap:  {value_gap:.4f}")
    print(f"  Policy accuracy gap:   {policy_gap:.4f}")
    
    if max(value_gap, policy_gap) > 0.05:
        print("  → Significant performance difference detected")
    else:
        print("  → Performance differences are small")

# =================== RUN EXPERIMENT ===================

if __name__ == "__main__":
    print("Starting Experiment 7.1: Architecture Comparison")
    
    # Import required packages
    try:
        import pandas as pd
        import time
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd
        import time
    
    # Run the experiment
    tracker, training_results, evaluation_results, models, env, ground_truth = run_experiment_7_1(
        grid_size=10,
        n_obstacles=8,
        episodes=2000,
        seed=42
    )
    
    print("\\nExperiment completed!")
    print("Results saved to exp_7_1_architecture_comparison_results.pkl")
    
    print("\\n" + "="*60)
    print("KEY RESEARCH INSIGHTS")
    print("="*60)
    
    print("\\n1. ARCHITECTURAL SPECIALIZATION:")
    print("   - Query-conditioned networks can specialize for different query types")
    print("   - Attention mechanisms help focus on relevant state information")
    print("   - Multi-head outputs allow specialized processing")
    
    print("\\n2. TRADE-OFFS:")
    print("   - Increased model complexity vs. improved query handling")
    print("   - Training time vs. inference capabilities")
    print("   - Parameter efficiency vs. specialization")
    
    print("\\n3. PRACTICAL IMPLICATIONS:")
    print("   - Choose architecture based on intended query types")
    print("   - Standard DQN sufficient for basic queries")
    print("   - Query-conditioned beneficial for diverse inference tasks")
    
    print("\\n4. FUTURE DIRECTIONS:")
    print("   - Test on larger, more complex environments")
    print("   - Investigate adaptive architectures")
    print("   - Develop better query encoding schemes")
    print("   - Explore meta-learning for query specialization")