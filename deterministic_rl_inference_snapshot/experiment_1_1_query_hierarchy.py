
# experiment_1_1_query_hierarchy.py
# Expressiveness experiment: compare baseline DQN vs Q-DIN across query types.

from helper import *
import torch, random

def run_experiment_1_1(grid_size=8, n_obstacles=8, episodes=600, seed=0):
    env = build_default_env(seed=seed)
    gt = make_ground_truth(env)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    # Train baseline DQN for control only
    dqn = train_dqn(env, steps=1500)

    # Train Q-DIN with inference-aware loss
    model = QDIN(env, hidden=128, K=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    w = LossWeights(td=0.1, inf=1.0, explic=0.05)
    mmp = MultiMetricProgression(env, V=gt['V'])

    tracker = ExperimentTracker()
    for ep in range(episodes):
        batch = select_queries_active_coverage(env, mmp, (gt['V'],gt['Q']), batch_size=24)
        loss, parts = inference_aware_loss(model, env, batch, gt, w)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep+1)%100==0:
            # eval on a held-out random batch
            test_q = select_queries_active_coverage(env, mmp, (gt['V'],gt['Q']), batch_size=40)
            metrics = evaluate_query_answering(model, env, test_q)
            tracker.log(ep=ep+1, loss=float(loss.item()), **metrics)
            print(f"[E1.1] ep={ep+1:4d} loss={loss.item():.3f} acc={metrics['acc']:.3f} meanIoU={metrics['set_mean_iou']:.3f}")
    summary = tracker.summary()
    print("[E1.1] Summary:", summary)
    return summary

if __name__=="__main__":
    run_experiment_1_1()
