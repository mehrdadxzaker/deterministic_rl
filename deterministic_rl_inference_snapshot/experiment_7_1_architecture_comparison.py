
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
            # control: returns
            rew_qdin = rollouts_return(env, lambda s: int(torch.argmax(qdin({'type':'policy','s':s})['policy']).item()), n_episodes=20)
            rew_dqn  = rollouts_return(env, lambda s: dqn.act(s, eps=0.01), n_episodes=20)
            tracker.log(ep=ep+1, loss=float(loss.item()), inf_acc=infm['acc'], set_iou=infm['set_mean_iou'], R_QDIN=rew_qdin, R_DQN=rew_dqn)
            print(f"[E7.1] ep={ep+1:4d} loss={loss.item():.3f} inf_acc={infm['acc']:.3f} IoU={infm['set_mean_iou']:.3f} R_QDIN={rew_qdin:.1f} R_DQN={rew_dqn:.1f}")
    summary = tracker.summary()
    print("[E7.1] Summary:", summary)
    return summary

def rollouts_return(env, policy_fn, n_episodes=10):
    tot=0.0
    max_steps = 4 * env.cfg.H * env.cfg.W
    for _ in range(n_episodes):
        s=env.reset(); ret=0.0
        for t in range(max_steps):
            if s==env.cfg.goal: break
            a=policy_fn(s)
            s,r,done,_=env.step(a)
            ret+=r
            if done: break
        tot+=ret
    return tot/n_episodes

if __name__=="__main__":
    run_experiment_7_1()
