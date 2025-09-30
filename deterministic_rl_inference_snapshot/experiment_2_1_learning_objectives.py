
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
            # control proxy: evaluate greedy policy from QDIN vs DQN returns on rollouts
            rew_qdin = rollouts_return(env, lambda s: int(torch.argmax(model({'type':'policy','s':s})['policy']).item()), n_episodes=20)
            rew_dqn = rollouts_return(env, lambda s: dqn.act(s, eps=0.01), n_episodes=20)
            tracker.log(ep=ep+1, mode=mode, loss=float(loss.item()), inf_acc=infm['acc'], set_iou=infm['set_mean_iou'], ret_qdin=rew_qdin, ret_dqn=rew_dqn)
            print(f"[E2.1][{mode}] ep={ep+1:4d} loss={loss.item():.3f} inf_acc={infm['acc']:.3f} IoU={infm['set_mean_iou']:.3f} R_qdin={rew_qdin:.1f} R_dqn={rew_dqn:.1f}")
    return tracker.summary()

def rollouts_return(env, policy_fn, n_episodes=10):
    tot=0.0
    max_steps = 4 * env.cfg.H * env.cfg.W
    for _ in range(n_episodes):
        s=env.reset(); ret=0.0
        for t in range(max_steps):
            if s==env.cfg.goal: break
            a=policy_fn(s)
            s,r,done,_=env.step(a)
            ret += r
            if done: break
        tot+=ret
    return tot/n_episodes

def run_experiment_2_1(grid_size=8, n_obstacles=8, episodes=600, seed=0):
    env = build_default_env(seed=seed)
    res = {}
    for mode in ["control-only","query-only","mixed"]:
        res[mode] = train_mode(env, mode=mode, episodes=episodes, seed=seed)
    print("[E2.1] Summary:", res)
    return res

if __name__=="__main__":
    run_experiment_2_1()
