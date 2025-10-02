
# experiment_2_1_learning_objectives.py
# Compare: control-only vs query-focused vs mixed objectives.

from helper import *
import torch, random

def train_mode(env, mode:str, episodes:int=600, seed:int=0):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    gt = make_ground_truth(env)
    mmp = MultiMetricProgression(env, V=gt['V'])

    if mode=="control-only":
        model = QDIN(env).to(device)
        w = LossWeights(td=1.0, inf=0.0, explic=0.0, model=0.25)
    elif mode=="query-only":
        model = QDIN(env).to(device)
        w = LossWeights(td=0.0, inf=1.0, explic=0.05, model=0.25)
    elif mode=="mixed":
        model = QDIN(env).to(device)
        w = LossWeights(td=0.1, inf=1.0, explic=0.05, model=0.25)
    else:
        raise ValueError("Unknown mode")

    balancer = MultiTaskLossBalancer([k for k,v in w.as_dict().items() if v>0]).to(device)
    params = list(model.parameters()) + list(balancer.parameters())
    opt = torch.optim.Adam(params, lr=5e-4)

    # baseline control reward estimator using trained DQN
    dqn = train_dqn(env, steps=1200)

    tracker = ExperimentTracker()
    for ep in range(episodes):
        phase, progress = curriculum_phase(ep, episodes)
        batch = select_queries_active_coverage(env, mmp, (gt['V'],gt['Q']), batch_size=24, phase=phase, phase_progress=progress)
        loss, parts = inference_aware_loss(model, env, batch, gt, w, balancer=balancer)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        if (ep+1)%100==0:
            # inference metrics
            test_q = select_queries_active_coverage(env, mmp, (gt['V'],gt['Q']), batch_size=50, phase="full", phase_progress=1.0)
            infm = evaluate_query_answering(model, env, test_q)
            metrics_qdin = rollouts_control_metrics(
                env,
                greedy_action_fn=lambda s: model.greedy_action(s),
                q_fn=lambda s: model.q_values(s),
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
                mode=mode,
                loss=float(loss.item()),
                overall_acc=infm['overall_acc'],
                policy_acc=infm['policy_acc'],
                reach_iou=infm['reach_mean_iou'],
                path_mae=infm['path_mae'],
                compare_acc=infm['compare_acc'],
                ret_qdin=metrics_qdin['avg_return'],
                ret_dqn=metrics_dqn['avg_return'],
                goal_rate_qdin=metrics_qdin['goal_reach_rate'],
                goal_rate_dqn=metrics_dqn['goal_reach_rate'],
                steps_qdin=metrics_qdin['steps_to_goal'],
                steps_dqn=metrics_dqn['steps_to_goal'],
            )
            print(
                f"[E2.1][{mode}] ep={ep+1:4d} loss={loss.item():.3f} "
                f"acc={infm['overall_acc']:.3f} pol={infm['policy_acc']:.3f} "
                f"IoU={infm['reach_mean_iou']:.3f} pathMAE={infm['path_mae']:.3f} "
                f"R_qdin={metrics_qdin['avg_return']:.1f} (succ={metrics_qdin['goal_reach_rate']*100:.1f}%) "
                f"R_dqn={metrics_dqn['avg_return']:.1f} (succ={metrics_dqn['goal_reach_rate']*100:.1f}%)"
            )
    return tracker.summary()

def run_experiment_2_1(grid_size=8, n_obstacles=8, episodes=600, seed=0):
    env = build_default_env(seed=seed)
    res = {}
    for mode in ["control-only","query-only","mixed"]:
        res[mode] = train_mode(env, mode=mode, episodes=episodes, seed=seed)
    print("[E2.1] Summary:", res)
    return res

if __name__=="__main__":
    run_experiment_2_1()
