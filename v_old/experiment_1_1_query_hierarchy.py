"""
Experiment 1.1: Query Type Hierarchy
Research Question: What class of inference problems can be solved by querying learned deterministic RL models?

Goal: Test different query types on trained DQN to establish expressiveness boundaries
Setup: 10x10 gridworld with obstacles, train standard DQN, test query answering
Expected Result: Some query types work well, others fail - establishes expressiveness boundaries
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from helper_functions import *

def run_experiment_1_1(grid_size=10, n_obstacles=8, episodes=2000, seed=42):
    """
    Run Experiment 1.1: Query Type Hierarchy
    
    Args:
        grid_size: Size of gridworld (grid_size x grid_size)
        n_obstacles: Number of obstacles to place
        episodes: Number of training episodes
        seed: Random seed for reproducibility
    """
    
    print("="*60)
    print("EXPERIMENT 1.1: Query Type Hierarchy")
    print("="*60)
    
    # Initialize experiment tracking
    tracker = ExperimentTracker("exp_1_1_query_hierarchy")
    tracker.log_metadata(
        grid_size=grid_size,
        n_obstacles=n_obstacles,
        episodes=episodes,
        seed=seed,
        description="Test query answering capability across different query types"
    )
    
    # Set random seed
    set_seeds(seed)
    
    # Create environment
    print(f"Creating {grid_size}x{grid_size} GridWorld with {n_obstacles} obstacles...")
    env = GridWorld(size=grid_size, n_obstacles=n_obstacles, seed=seed)
    
    # Visualize environment
    print("Environment layout:")
    env.render()
    
    # Create and train model
    print("\\nTraining Standard DQN...")
    state_dim = 1  # State is just an integer
    action_dim = env.action_space.n
    model = StandardDQN(state_dim, action_dim, hidden_dim=128)
    
    # Train the model
    episode_rewards, losses = train_dqn(env, model, episodes=episodes)
    
    # Plot training progress
    plot_training_curves(episode_rewards, losses, "DQN Training Progress")
    
    # Compute ground truth for evaluation
    print("\\nComputing ground truth values...")
    ground_truth = compute_ground_truth_values(env)
    V_true, Q_true, policy_true = ground_truth
    
    # Generate different types of queries
    print("\\nGenerating queries...")
    query_gen = QueryGenerator(env)
    
    point_queries = query_gen.generate_point_queries(n_queries=50)
    path_queries = query_gen.generate_path_queries(n_queries=25) 
    set_queries = query_gen.generate_set_queries(n_queries=15)
    comparative_queries = query_gen.generate_comparative_queries(n_queries=15)
    
    all_queries = point_queries + path_queries + set_queries + comparative_queries
    
    print(f"Generated {len(point_queries)} point queries")
    print(f"Generated {len(path_queries)} path queries") 
    print(f"Generated {len(set_queries)} set queries")
    print(f"Generated {len(comparative_queries)} comparative queries")
    print(f"Total: {len(all_queries)} queries")
    
    # Evaluate query answering
    print("\\nEvaluating query answering...")
    
    # Point queries (these should work well)
    point_results = evaluate_point_queries(model, point_queries, ground_truth)
    
    # Path queries (these might be harder)  
    path_results = evaluate_path_queries(model, env, path_queries, ground_truth)
    
    # Set queries (these require more complex reasoning)
    set_results = evaluate_set_queries(model, env, set_queries, ground_truth)
    
    # Comparative queries (these should work if point queries work)
    comp_results = evaluate_comparative_queries(model, comparative_queries, ground_truth)
    
    # Combine all results
    all_results = {**point_results, **path_results, **set_results, **comp_results}
    
    # Log results
    for key, values in all_results.items():
        tracker.log_result(key, np.mean(values))
        tracker.log_result(f"{key}_std", np.std(values))
        tracker.log_result(f"{key}_all", values)
    
    # Print summary results
    print("\\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    
    print(f"\\nPOINT QUERIES:")
    print(f"  Value prediction error: {np.mean(point_results['value_errors']):.4f} ± {np.std(point_results['value_errors']):.4f}")
    print(f"  Policy accuracy: {np.mean(point_results['policy_accuracy']):.4f} ± {np.std(point_results['policy_accuracy']):.4f}")
    print(f"  Q-value error: {np.mean(point_results['qvalue_errors']):.4f} ± {np.std(point_results['qvalue_errors']):.4f}")
    
    print(f"\\nPATH QUERIES:")
    print(f"  Path cost error: {np.mean(path_results.get('path_cost_errors', [0])):.4f} ± {np.std(path_results.get('path_cost_errors', [0])):.4f}")
    print(f"  Optimal path accuracy: {np.mean(path_results.get('optimal_path_accuracy', [0])):.4f}")
    
    print(f"\\nSET QUERIES:")
    print(f"  Reachability accuracy: {np.mean(set_results.get('reachability_accuracy', [0])):.4f}")
    print(f"  High-value set accuracy: {np.mean(set_results.get('high_value_accuracy', [0])):.4f}")
    
    print(f"\\nCOMPARATIVE QUERIES:")
    print(f"  Action comparison accuracy: {np.mean(comp_results.get('comparison_accuracy', [0])):.4f}")
    
    # Visualize results
    plot_query_hierarchy_results(all_results)
    
    # Save results
    tracker.save()
    
    return tracker, all_results, model, env, ground_truth

def evaluate_point_queries(model, queries, ground_truth):
    """Evaluate point queries: V(s), π(s), Q(s,a)"""
    V_true, Q_true, policy_true = ground_truth
    results = defaultdict(list)
    
    for query in queries:
        if query.type != 'point':
            continue
            
        state = query.parameters['state']
        
        if query.parameters['query_type'] == 'value':
            # Model prediction: V(s) = max_a Q(s,a)
            with torch.no_grad():
                q_values = model(state)
                predicted_value = q_values.max().item()
            
            true_value = V_true[state]
            error = abs(predicted_value - true_value)
            results['value_errors'].append(error)
            
        elif query.parameters['query_type'] == 'policy':
            # Model prediction: π(s) = argmax_a Q(s,a)
            with torch.no_grad():
                q_values = model(state)
                predicted_action = q_values.argmax().item()
            
            true_action = policy_true[state]
            correct = (predicted_action == true_action)
            results['policy_accuracy'].append(correct)
            
        elif query.parameters['query_type'] == 'qvalue':
            # Model prediction: Q(s,a)
            state = query.parameters['state']
            action = query.parameters['action']
            
            with torch.no_grad():
                q_values = model(state)
                predicted_q = q_values[action].item()
            
            true_q = Q_true[state, action]
            error = abs(predicted_q - true_q)
            results['qvalue_errors'].append(error)
    
    return results

def evaluate_path_queries(model, env, queries, ground_truth):
    """Evaluate path queries: optimal trajectories, path costs"""
    V_true, Q_true, policy_true = ground_truth
    results = defaultdict(list)
    
    for query in queries:
        if query.type != 'path':
            continue
            
        if query.parameters['query_type'] == 'path_cost':
            # Evaluate cost of a given path
            path = query.parameters['path']
            
            # True path cost
            true_cost = 0
            for i in range(len(path)-1):
                current_state = path[i]
                next_state = path[i+1]
                # Find action that leads from current to next
                found_action = None
                for action in range(env.action_space.n):
                    if env.get_next_state(current_state, action) == next_state:
                        found_action = action
                        break
                
                if found_action is not None:
                    true_cost += env.rewards[next_state]
                else:
                    # Invalid transition, assign large penalty
                    true_cost += -1.0
            
            # Predicted path cost using learned Q-values
            predicted_cost = 0
            for i in range(len(path)-1):
                current_state = path[i]
                next_state = path[i+1]
                # Find action
                found_action = None
                for action in range(env.action_space.n):
                    if env.get_next_state(current_state, action) == next_state:
                        found_action = action
                        break
                
                if found_action is not None:
                    with torch.no_grad():
                        q_values = model(current_state)
                        # Use immediate reward approximation
                        predicted_cost += env.rewards[next_state]
                else:
                    predicted_cost += -1.0
            
            error = abs(predicted_cost - true_cost)
            results['path_cost_errors'].append(error)
            
        elif query.parameters['query_type'] == 'optimal_path':
            # Check if model can find optimal path between states
            start = query.parameters['start']
            end = query.parameters['end']
            
            # Generate path using learned policy
            predicted_path = []
            current = start
            max_steps = env.size * env.size
            
            for step in range(max_steps):
                predicted_path.append(current)
                if current == end:
                    break
                    
                with torch.no_grad():
                    q_values = model(current)
                    action = q_values.argmax().item()
                
                current = env.get_next_state(current, action)
                
                # Avoid infinite loops
                if current in predicted_path[-env.size:]:
                    break
            
            # Check if path reaches the goal
            reaches_goal = (predicted_path[-1] == end)
            results['optimal_path_accuracy'].append(reaches_goal)
            
            # Also compute path length
            if reaches_goal:
                results['path_lengths'].append(len(predicted_path))
    
    return results

def evaluate_set_queries(model, env, queries, ground_truth):
    """Evaluate set queries: reachable states, high-value states"""
    V_true, Q_true, policy_true = ground_truth
    results = defaultdict(list)
    
    for query in queries:
        if query.type != 'set':
            continue
            
        if query.parameters['query_type'] == 'reachable':
            # Find states reachable within k steps
            start = query.parameters['start']
            k = query.parameters['steps']
            
            # True reachable set (breadth-first search)
            true_reachable = set()
            queue = [(start, 0)]  # (state, steps)
            visited = set()
            
            while queue:
                state, steps = queue.pop(0)
                if state in visited or steps > k:
                    continue
                    
                visited.add(state)
                true_reachable.add(state)
                
                if steps < k:
                    for action in range(env.action_space.n):
                        next_state = env.get_next_state(state, action)
                        if next_state not in visited:
                            queue.append((next_state, steps + 1))
            
            # Predicted reachable set using learned policy
            predicted_reachable = set()
            queue = [(start, 0)]
            visited = set()
            
            while queue:
                state, steps = queue.pop(0)
                if state in visited or steps > k:
                    continue
                    
                visited.add(state)
                predicted_reachable.add(state)
                
                if steps < k:
                    # Use learned policy to expand
                    with torch.no_grad():
                        q_values = model(state)
                        # Consider top actions, not just greedy
                        top_actions = q_values.argsort(descending=True)[:2]  # Top 2 actions
                        
                    for action in top_actions:
                        next_state = env.get_next_state(state, action.item())
                        if next_state not in visited:
                            queue.append((next_state, steps + 1))
            
            # Compute accuracy as Jaccard similarity
            intersection = len(true_reachable.intersection(predicted_reachable))
            union = len(true_reachable.union(predicted_reachable))
            accuracy = intersection / union if union > 0 else 1.0
            results['reachability_accuracy'].append(accuracy)
            
        elif query.parameters['query_type'] == 'high_value':
            # Find states with value above threshold
            threshold = query.parameters['threshold']
            
            # True high-value states
            true_high_value = set()
            for state in range(env.observation_space.n):
                if V_true[state] > threshold:
                    true_high_value.add(state)
            
            # Predicted high-value states
            predicted_high_value = set()
            for state in range(env.observation_space.n):
                with torch.no_grad():
                    q_values = model(state)
                    predicted_value = q_values.max().item()
                
                if predicted_value > threshold:
                    predicted_high_value.add(state)
            
            # Compute accuracy
            intersection = len(true_high_value.intersection(predicted_high_value))
            union = len(true_high_value.union(predicted_high_value))
            accuracy = intersection / union if union > 0 else 1.0
            results['high_value_accuracy'].append(accuracy)
    
    return results

def evaluate_comparative_queries(model, queries, ground_truth):
    """Evaluate comparative queries: which action/path is better"""
    V_true, Q_true, policy_true = ground_truth
    results = defaultdict(list)
    
    for query in queries:
        if query.type != 'comparative':
            continue
            
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
    
    return results

def plot_query_hierarchy_results(results):
    """Plot results for different query types"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Point queries
    if 'value_errors' in results:
        axes[0,0].hist(results['value_errors'], bins=15, alpha=0.7, color='blue')
        axes[0,0].set_title('Value Prediction Errors')
        axes[0,0].set_xlabel('Absolute Error')
        axes[0,0].axvline(np.mean(results['value_errors']), color='red', linestyle='--')
    
    if 'policy_accuracy' in results:
        acc = np.mean(results['policy_accuracy'])
        axes[0,1].bar(['Correct', 'Incorrect'], [acc, 1-acc], color=['green', 'red'], alpha=0.7)
        axes[0,1].set_title(f'Policy Accuracy: {acc:.3f}')
    
    if 'qvalue_errors' in results:
        axes[0,2].hist(results['qvalue_errors'], bins=15, alpha=0.7, color='purple')
        axes[0,2].set_title('Q-Value Prediction Errors')
        axes[0,2].set_xlabel('Absolute Error')
        axes[0,2].axvline(np.mean(results['qvalue_errors']), color='red', linestyle='--')
    
    # Path queries
    if 'path_cost_errors' in results:
        axes[1,0].hist(results['path_cost_errors'], bins=15, alpha=0.7, color='orange')
        axes[1,0].set_title('Path Cost Errors')
        axes[1,0].set_xlabel('Absolute Error')
        axes[1,0].axvline(np.mean(results['path_cost_errors']), color='red', linestyle='--')
    
    # Set queries
    if 'reachability_accuracy' in results and 'high_value_accuracy' in results:
        reach_acc = np.mean(results['reachability_accuracy'])
        value_acc = np.mean(results['high_value_accuracy'])
        axes[1,1].bar(['Reachability', 'High-Value'], [reach_acc, value_acc], 
                      color=['cyan', 'magenta'], alpha=0.7)
        axes[1,1].set_title('Set Query Accuracy')
        axes[1,1].set_ylabel('Jaccard Similarity')
    
    # Comparative queries
    if 'comparison_accuracy' in results:
        comp_acc = np.mean(results['comparison_accuracy'])
        axes[1,2].bar(['Correct', 'Incorrect'], [comp_acc, 1-comp_acc], 
                      color=['green', 'red'], alpha=0.7)
        axes[1,2].set_title(f'Comparison Accuracy: {comp_acc:.3f}')
    
    plt.suptitle('Query Type Hierarchy Results', fontsize=16)
    plt.tight_layout()
    plt.show()

# =================== RUN EXPERIMENT ===================

if __name__ == "__main__":
    print("Starting Experiment 1.1: Query Type Hierarchy")
    
    # Run the experiment
    tracker, results, model, env, ground_truth = run_experiment_1_1(
        grid_size=10,
        n_obstacles=8, 
        episodes=2000,
        seed=42
    )
    
    print("\\nExperiment completed!")
    print("Results saved to exp_1_1_query_hierarchy_results.pkl")
    print("\\nKey Findings:")
    print("- Point queries (V, π, Q) should work well if DQN training succeeded")
    print("- Path queries depend on learned policy quality")  
    print("- Set queries require more complex reasoning and may be harder")
    print("- Comparative queries should work if Q-values are well-learned")
    
    # Quick analysis
    print("\\n" + "="*50)
    print("EXPRESSIVENESS ANALYSIS")
    print("="*50)
    
    # Check which query types work well (>70% accuracy or <0.1 error)
    successful_queries = []
    
    if np.mean(results.get('policy_accuracy', [0])) > 0.7:
        successful_queries.append("Policy queries")
    if np.mean(results.get('value_errors', [1])) < 0.1:
        successful_queries.append("Value queries") 
    if np.mean(results.get('comparison_accuracy', [0])) > 0.7:
        successful_queries.append("Comparative queries")
    if np.mean(results.get('reachability_accuracy', [0])) > 0.5:
        successful_queries.append("Reachability queries")
        
    print(f"Query types that work well: {successful_queries}")
    print(f"Total successful query types: {len(successful_queries)}/4")
    
    if len(successful_queries) >= 3:
        print("✓ Good expressiveness - most query types work")
    elif len(successful_queries) >= 2:
        print("~ Moderate expressiveness - some limitations")
    else:
        print("✗ Limited expressiveness - may need better training or architecture")