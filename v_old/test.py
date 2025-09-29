# Quick test everything works
quick_test()

# See usage examples
demonstrate_usage()

# Create environment and model
env = GridWorld(size=10, n_obstacles=8)
model = StandardDQN(state_dim=1, action_dim=4)
model.set_state_space_size(env.observation_space.n)

# Train and evaluate
rewards, losses = train_dqn(env, model, episodes=1000)
ground_truth = compute_ground_truth_values(env)
queries = QueryGenerator(env).generate_point_queries(100)
results = evaluate_query_answering(model, env, queries, ground_truth)