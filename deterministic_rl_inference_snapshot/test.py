
# test.py
# Runs all experiments with small budgets to smoke-test.

from experiment_1_1_query_hierarchy import run_experiment_1_1
from experiment_2_1_learning_objectives import run_experiment_2_1
from experiment_7_1_architecture_comparison import run_experiment_7_1
from helper import unit_test_value_scaling

print("Running E1.1 ...")
out1 = run_experiment_1_1(episodes=200)
print("Running E2.1 ...")
out2 = run_experiment_2_1(episodes=200)
print("Running E7.1 ...")
out3 = run_experiment_7_1(episodes=200)
print("Running value scaling unit test ...")
scale_metrics = unit_test_value_scaling()
print("Value scaling errors:", scale_metrics)

print("Done.")
