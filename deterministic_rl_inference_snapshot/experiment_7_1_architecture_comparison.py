
# experiment_7_1_architecture_comparison.py
# Compare Standard DQN vs Q-DIN on inference and control metrics.

from helper import *
import torch

def run_experiment_7_1(grid_size=8, n_obstacles=8, episodes=600, seed=0):
    env = build_default_env(seed=seed)
    gt = make_ground_truth(env)
    mmp = MultiMetricProgression(env, V=gt['V'])

    # Train DQN
    dqn = train_dqn(env, steps=1500)
    # Train Q-DIN
    device='cuda' if torch.cuda.is_available() else 'cpu'
    qdin = QDIN(env).to(device)
    w = LossWeights(td=0.1, inf=1.0, explic=0.05, model=0.25)
    balancer = MultiTaskLossBalancer([k for k,v in w.as_dict().items() if v>0]).to(device)
    params = list(qdin.parameters()) + list(balancer.parameters())
    opt = torch.optim.Adam(params, lr=5e-4)

    tracker = ExperimentTracker()
    for ep in range(episodes):
        phase, progress = curriculum_phase(ep, episodes)
        batch = select_queries_active_coverage(env, mmp, (gt['V'],gt['Q']), batch_size=24, phase=phase, phase_progress=progress)
        loss, parts = inference_aware_loss(qdin, env, batch, gt, w, balancer=balancer)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(qdin.parameters(), 1.0); opt.step()
        if (ep+1)%100==0:
            # evaluate
            test_q = select_queries_active_coverage(env, mmp, (gt['V'],gt['Q']), batch_size=60, phase="full", phase_progress=1.0)
            infm = evaluate_query_answering(qdin, env, test_q)
            metrics_qdin = rollouts_control_metrics(
                env,
                greedy_action_fn=lambda s: qdin.greedy_action(s),
                q_fn=lambda s: qdin.q_values(s),
                n_episodes=20,
            )
            metrics_dqn = rollouts_control_metrics(
                env,
                greedy_action_fn=lambda s: dqn.greedy_action(s),
                q_fn=lambda s: dqn.q_values(s),
                n_episodes=20,
            )
            tracker.log(
                ep=ep+1,
                loss=float(loss.item()),
                overall_acc=infm['overall_acc'],
                policy_acc=infm['policy_acc'],
                reach_iou=infm['reach_mean_iou'],
                path_mae=infm['path_mae'],
                compare_acc=infm['compare_acc'],
                R_QDIN=metrics_qdin['avg_return'],
                R_DQN=metrics_dqn['avg_return'],
                goal_rate_qdin=metrics_qdin['goal_reach_rate'],
                goal_rate_dqn=metrics_dqn['goal_reach_rate'],
                steps_qdin=metrics_qdin['steps_to_goal'],
                steps_dqn=metrics_dqn['steps_to_goal'],
            )
            print(
                f"[E7.1] ep={ep+1:4d} loss={loss.item():.3f} "
                f"acc={infm['overall_acc']:.3f} pol={infm['policy_acc']:.3f} "
                f"IoU={infm['reach_mean_iou']:.3f} pathMAE={infm['path_mae']:.3f} "
                f"R_QDIN={metrics_qdin['avg_return']:.1f} (succ={metrics_qdin['goal_reach_rate']*100:.1f}%) "
                f"R_DQN={metrics_dqn['avg_return']:.1f} (succ={metrics_dqn['goal_reach_rate']*100:.1f}%)"
            )
    summary = tracker.summary()
    print("[E7.1] Summary:", summary)
    return summary

if __name__=="__main__":
    run_experiment_7_1()
