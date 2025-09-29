
# helper.py
# Deterministic RL as a Query-Conditioned Inference Engine (Q-DIN)
# + Multi-Metric Progression (MMP) + Active inference-coverage training
# The code is intentionally lightweight and self-contained for Colab runs.
# Author: (your name), 2025

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math, random, itertools, collections

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except Exception as e:
    raise RuntimeError("PyTorch is required. Please run install_requirements.py first.")


# ------------------------- Simple deterministic GridWorld -------------------------

Action = int  # 0:up,1:right,2:down,3:left
ACTIONS = [0,1,2,3]
ACTION_VECS = {0: (-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
ACTION_NAMES = {0:"U",1:"R",2:"D",3:"L"}

@dataclass
class GridConfig:
    H:int=8
    W:int=8
    walls:Optional[List[Tuple[int,int]]]=None
    start:(int,int)=(0,0)
    goal:(int,int)=(7,7)
    step_cost:float=-1.0
    goal_reward:float=0.0 # using negative step costs so reaching goal yields 0 additional
    seed:int=0

class GridWorld:
    """
    Deterministic gridworld. States are (r,c). Episode ends on goal.
    """
    def __init__(self, cfg:GridConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.H, self.W = cfg.H, cfg.W
        self.walls = set(cfg.walls or [])
        self.start = cfg.start
        self.goal = cfg.goal
        self.state = self.start
        self._all_states = [(r,c) for r in range(self.H) for c in range(self.W) if (r,c) not in self.walls]
        self.S = len(self._all_states)
        self.A = 4
        # map state <-> index
        self.s2i = {s:i for i,s in enumerate(self._all_states)}
        self.i2s = {i:s for s,i in self.s2i.items()}

    def reset(self, state:Optional[Tuple[int,int]]=None):
        self.state = state if state is not None else self.start
        return self.state

    def in_bounds(self, r,c):
        return (0 <= r < self.H) and (0 <= c < self.W)

    def step(self, a:Action):
        dr,dc = ACTION_VECS[a]
        r,c = self.state
        nr, nc = r+dr, c+dc
        nxt = (r,c)
        if self.in_bounds(nr,nc) and (nr,nc) not in self.walls:
            nxt = (nr,nc)
        self.state = nxt
        rwd = self.cfg.step_cost
        done = (nxt == self.cfg.goal)
        if done:
            rwd += self.cfg.goal_reward
        return nxt, rwd, done, {}

    def neighbors(self, s):
        r,c = s
        for a in ACTIONS:
            dr,dc = ACTION_VECS[a]
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr,nc) and (nr,nc) not in self.walls:
                yield a,(nr,nc)

    def shortest_path(self, start=None, goal=None):
        """
        BFS shortest path (ties arbitrary). Returns list of actions.
        """
        start = self.start if start is None else start
        goal = self.goal if goal is None else goal
        if start == goal: return []
        Q = collections.deque([start])
        vis = {start: None}
        via_a = {}
        while Q:
            u = Q.popleft()
            if u == goal: break
            for a,v in self.neighbors(u):
                if v not in vis:
                    vis[v]=u
                    via_a[v]=a
                    Q.append(v)
        if goal not in vis: return None
        # reconstruct
        path = []
        cur = goal
        while cur != start:
            a = via_a[cur]
            path.append(a)
            cur = vis[cur]
        return list(reversed(path))

    def k_step_reachable(self, s, k):
        R = set([s])
        frontier = set([s])
        for _ in range(k):
            newf=set()
            for u in frontier:
                for a,v in self.neighbors(u):
                    newf.add(v)
            R |= newf
            frontier = newf
        return R


# ------------------------- Ground-truth values via dynamic programming -------------------------

def compute_value_fn(env:GridWorld, gamma=0.99):
    """
    Value iteration for deterministic grid with step cost.
    Returns V dict and Q dict as lookup tables.
    """
    S = env._all_states
    V = {s: 0.0 for s in S}
    V[env.cfg.goal] = 0.0
    for _ in range(200):
        delta=0.0
        for s in S:
            if s==env.cfg.goal:
                continue
            best = -1e9
            for a,v in env.neighbors(s):
                r = env.cfg.step_cost + (env.cfg.goal_reward if v==env.cfg.goal else 0.0)
                val = r + gamma*V[v]
                if val > best: best = val
            nd = abs(best - V[s])
            V[s]=best
            delta=max(delta,nd)
        if delta<1e-5: break
    # Q
    Q = {(s,a):-1e9 for s in S for a in ACTIONS}
    for s in S:
        for a,v in env.neighbors(s):
            r = env.cfg.step_cost + (env.cfg.goal_reward if v==env.cfg.goal else 0.0)
            Q[(s,a)] = r + gamma*V[v]
    return V, Q


# ------------------------- Query taxonomy -------------------------

# We encode queries as dicts: {'type': 'value'|'q'|'policy'|'set'|'pathcost'|'compare', ... params ...}

def make_point_queries(env:GridWorld, states:List[Tuple[int,int]], actions:List[int]):
    qs=[]
    for s in states:
        qs.append({'type':'value','s':s})
        qs.append({'type':'policy','s':s})
        for a in actions:
            qs.append({'type':'q','s':s,'a':a})
    return qs

def make_path_queries(env:GridWorld, pairs:List[Tuple[Tuple[int,int],Tuple[int,int]]], cost_per_step=1.0):
    qs=[]
    for s,g in pairs:
        qs.append({'type':'pathcost','s':s,'g':g,'c':cost_per_step})
    return qs

def make_set_queries(env:GridWorld, states:List[Tuple[int,int]], k:int=3):
    return [{'type':'reachable','s':s,'k':k} for s in states]

def make_comparative_queries(env:GridWorld, items:List[Tuple[List[int],List[int]]], s:Tuple[int,int]):
    return [{'type':'compare','s':s,'p1':p1,'p2':p2} for (p1,p2) in items]


# ------------------------- Plan / distance utilities -------------------------

def levenshtein(a:List[int], b:List[int]):
    m,n=len(a),len(b)
    if m==0: return n
    if n==0: return m
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            cost=0 if a[i-1]==b[j-1] else 1
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]

def lcs_length(a:List[int], b:List[int]):
    m,n=len(a),len(b)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if a[i]==b[j]: dp[i][j]=1+dp[i+1][j+1]
            else: dp[i][j]=max(dp[i+1][j],dp[i][j+1])
    return dp[0][0]

def action_multiset_divergence(a:List[int], b:List[int]):
    ca=collections.Counter(a); cb=collections.Counter(b)
    acts=set(ca)|set(cb)
    return sum(abs(ca[x]-cb[x]) for x in acts)

def prefix_disagreement(a:List[int], b:List[int]):
    m=min(len(a),len(b))
    for i in range(m):
        if a[i]!=b[i]:
            return m-i
    return abs(len(a)-len(b))

def jaccard(A:set, B:set):
    if not A and not B: return 1.0
    return len(A & B)/max(1,len(A | B))

# ------------------------- Multi-Metric Progression -------------------------

class MultiMetricProgression:
    """
    A learned (or user-weighted) mixture of distances for ordering queries
    and for acquisition during active coverage.
    """
    def __init__(self, env:GridWorld, V:Dict=None, weights:Dict[str,float]=None, include_lev=True):
        self.env=env
        self.V = V or compute_value_fn(env)[0]
        self.weights = weights or {
            'lcs': 0.25,
            'multiset': 0.25,
            'prefix': 0.2,
            'value_drift': 0.15,
            'reach_jaccard': 0.15,
            'levenshtein': 0.0 if not include_lev else 0.05,
        }

    def plan_for_query(self, q):
        # basic proxy plans for point queries: optimal from s to goal; for pathcost, shortest path from s to g
        if q['type'] in ('value','policy','q','compare'):
            return self.env.shortest_path(q['s'], self.env.cfg.goal) or []
        if q['type']=='pathcost':
            return self.env.shortest_path(q['s'], q['g']) or []
        if q['type']=='reachable':
            # encode reachability frontier as a pseudo-plan: BFS from s to near frontier using greedy steps toward goal
            path = self.env.shortest_path(q['s'], self.env.cfg.goal) or []
            return path[:q.get('k',3)]
        return []

    def set_for_query(self, q):
        if q['type']=='reachable':
            return self.env.k_step_reachable(q['s'], q.get('k', 3))
        return set()

    def value_trace(self, s, plan:List[int]):
        vals=[]; cur=s
        for a in plan:
            # take neighbor in deterministic way, if blocked stay
            nxt = cur
            r,c = cur; dr,dc = ACTION_VECS[a]; nr, nc = r+dr, c+dc
            if self.env.in_bounds(nr,nc) and (nr,nc) not in self.env.walls: nxt=(nr,nc)
            vals.append(self.V.get(nxt,0.0))
            cur=nxt
        return vals

    def value_drift(self, s, p1, p2):
        v1=self.value_trace(s,p1); v2=self.value_trace(s,p2)
        m=max(len(v1),len(v2))
        v1=v1+[v1[-1] if v1 else 0.0]*(m-len(v1))
        v2=v2+[v2[-1] if v2 else 0.0]*(m-len(v2))
        return sum(abs(a-b) for a,b in zip(v1,v2))/max(1,m)

    def dist(self, q_prev, q_next):
        p1=self.plan_for_query(q_prev); p2=self.plan_for_query(q_next)
        # normalize by lengths
        lcs = 1.0 - (lcs_length(p1,p2)/max(1,max(len(p1),len(p2))))
        mult = action_multiset_divergence(p1,p2)/max(1,len(p1)+len(p2))
        pref = prefix_disagreement(p1,p2)/max(1,max(len(p1),len(p2)))
        vdrift = self.value_drift(q_prev.get('s', self.env.cfg.start), p1,p2)
        # reachable set Jaccard (as distance)
        if q_prev['type']=='reachable' or q_next['type']=='reachable':
            A=self.set_for_query(q_prev); B=self.set_for_query(q_next)
            rjac = 1.0 - jaccard(A,B)
        else:
            rjac = 0.0
        lev = levenshtein(p1,p2)/max(1,max(len(p1),len(p2)))
        w=self.weights
        score = (
            w['lcs'] * lcs
            + w['multiset'] * mult
            + w['prefix'] * pref
            + w['value_drift'] * vdrift
            + w['reach_jaccard'] * rjac
            + w['levenshtein'] * lev
        )
        return float(score)

    def order_queries_progressively(self, queries:List[Dict], anchor:Optional[Dict]=None):
        if not queries: return []
        ordered=[]
        remaining=queries[:]
        cur = anchor if anchor is not None else remaining.pop(0)
        ordered.append(cur)
        # greedy nearest-neighbor under dist
        while remaining:
            best=None; bestd=1e9
            for q in remaining:
                d=self.dist(cur,q)
                if d<bestd:
                    bestd=d; best=q
            ordered.append(best)
            remaining.remove(best)
            cur=best
        return ordered


# ------------------------- Q-DIN (Query-conditioned Deterministic Inference Network) -------------------------

class QDIN(nn.Module):
    """
    A tiny query-conditioned network that (i) embeds (x,y), (ii) embeds query,
    (iii) runs a few value-iteration-like updates over a learned transition,
    and (iv) decodes typed answers.
    """
    def __init__(self, env:GridWorld, hidden=128, K=5):
        super().__init__()
        self.env=env
        self.S=len(env._all_states)
        self.A=4
        self.K=K

        # state embedding: index of state as one-hot -> MLP
        self.state_idx = {s:i for i,s in enumerate(env._all_states)}
        self.embed_s = nn.Embedding(self.S, hidden)

        # query embeddings: type id + a small numeric vector
        self.type2id = {'value':0,'q':1,'policy':2,'reachable':3,'pathcost':4,'compare':5}
        self.embed_t = nn.Embedding(len(self.type2id), hidden)
        self.fc_q = nn.Linear(6, hidden)  # room for small query params

        # learned "model": transition logits and rewards (dense)
        self.T_logits = nn.Parameter(torch.zeros(self.A, self.S, self.S))  # a -> s -> s'
        nn.init.xavier_uniform_(self.T_logits)
        self.R = nn.Parameter(torch.zeros(self.S, self.A))
        nn.init.xavier_uniform_(self.R)

        # The fusion layer consumes state/query embeddings along with scalar value
        # and action-value features. The concatenated vector therefore has size
        # ``hidden + hidden + 1 + A`` rather than ``hidden*3``.
        self.readout = nn.Linear(hidden * 2 + self.A + 1, hidden)

        # Heads
        self.h_value = nn.Linear(hidden, 1)
        self.h_q = nn.Linear(hidden, self.A)
        self.h_policy = nn.Linear(hidden, self.A)
        self.h_set = nn.Linear(hidden, self.S)      # reachability mask
        self.h_pathcost = nn.Linear(hidden, 1)
        self.h_explic = nn.Linear(hidden, 1)

    def vi_block(self, gamma=0.99):
        # Perform K value-iteration steps over learned (dense) model
        # Softmax over next states
        T = torch.softmax(self.T_logits, dim=-1)  # [A,S,S]
        R = self.R  # [S,A]
        V = torch.zeros(self.S, device=R.device)
        for _ in range(self.K):
            Q = R + gamma * torch.einsum('ask, k -> sa', T, V)  # [S,A]
            V = torch.max(Q, dim=1).values
        return V, Q  # [S], [S,A]

    def encode_query(self, q:Dict[str,Any]):
        # numeric features: s_idx, a, k, g_idx_r, g_idx_c (if provided)
        t = torch.tensor([self.type2id[q['type']]]).long()
        t_emb = self.embed_t(t)

        s = q.get('s', self.env.cfg.start)
        s_idx = self.state_idx.get(s, 0)
        a = q.get('a', -1)
        k = q.get('k', 0)
        g = q.get('g', self.env.cfg.goal)
        g_idx = self.state_idx.get(g, 0)
        nums = torch.tensor([s_idx, a, k, g_idx, self.env.cfg.H, self.env.cfg.W]).float().unsqueeze(0) # [1,6]
        qvec = self.fc_q(nums)
        return (t_emb + qvec)  # [1,hidden]

    def encode_state(self, s:Tuple[int,int]):
        i = self.state_idx[s]
        idx = torch.tensor([i]).long()
        return self.embed_s(idx)  # [1,hidden]

    def forward(self, q:Dict[str,Any]):
        device = next(self.parameters()).device
        # run VI block to get global value landscape
        V_all, Q_all = self.vi_block()
        V_all = V_all.unsqueeze(0)        # [1,S]
        Q_all = Q_all.unsqueeze(0)        # [1,S,A]

        # get state and query encodings
        s = q.get('s', self.env.cfg.start)
        s_emb = self.encode_state(s).to(device)    # [1,hidden]
        q_emb = self.encode_query(q).to(device)    # [1,hidden]

        # gather V,Q at s
        si = self.state_idx[s]
        V_s = V_all[:, si:si+1]                    # [1,1]
        Q_s = Q_all[:, si, :]                      # [1,A]

        # fuse
        x = torch.cat([s_emb, q_emb, torch.cat([V_s, Q_s], dim=1)], dim=1)  # [1,hidden+hidden+1+A]
        x = self.readout(x)              # [1,hidden]
        x = torch.tanh(x)

        out = {}
        out['value'] = self.h_value(x).squeeze(0)          # [1]
        out['q'] = self.h_q(x).squeeze(0)                  # [A]
        out['policy'] = self.h_policy(x).squeeze(0)        # [A]
        out['set_logits'] = self.h_set(x).squeeze(0)       # [S]
        out['pathcost'] = self.h_pathcost(x).squeeze(0)    # [1]
        out['explic'] = torch.sigmoid(self.h_explic(x)).squeeze(0) # [1] in [0,1]
        return out


# ------------------------- Losses and training utilities -------------------------

@dataclass
class LossWeights:
    td: float = 0.2
    inf: float = 1.0
    explic: float = 0.1

def inference_aware_loss(model:QDIN, env:GridWorld, batch_q:List[Dict], targets:Dict[str,Any], w:LossWeights):
    device = next(model.parameters()).device
    loss_inf = torch.tensor(0.0, device=device)
    bsz = len(batch_q)
    for q in batch_q:
        out = model(q)
        t = q['type']
        if t=='value':
            y = torch.tensor([targets['V'][q['s']]], device=device).float()
            loss_inf = loss_inf + F.mse_loss(out['value'].view_as(y), y)
        elif t=='q':
            y = torch.tensor([targets['Q'][(q['s'],q['a'])]], device=device).float()
            qsa = out['q'][q['a']]
            loss_inf = loss_inf + F.mse_loss(qsa.view_as(y), y)
        elif t=='policy':
            # best action label
            s=q['s']
            best = max(ACTIONS, key=lambda a: targets['Q'][(s,a)])
            logits = out['policy'].unsqueeze(0)
            y = torch.tensor([best], device=device).long()
            loss_inf = loss_inf + F.cross_entropy(logits, y)
        elif t=='reachable':
            # predict k-step reachable set mask
            k=q.get('k',3)
            R = env.k_step_reachable(q['s'], k)
            mask = torch.zeros(model.S, device=device)
            idxs=[model.state_idx[s] for s in R]
            mask[idxs]=1.0
            pred = out['set_logits']
            loss_inf = loss_inf + F.binary_cross_entropy_with_logits(pred, mask)
        elif t=='pathcost':
            # path cost approximated by step count * |step_cost|
            P = env.shortest_path(q['s'], q['g']) or []
            y = torch.tensor([float(len(P)) * abs(env.cfg.step_cost)], device=device).float()
            loss_inf = loss_inf + F.mse_loss(out['pathcost'].view_as(y), y)
        elif t=='compare':
            # binary: is path1 shorter than path2 from s?
            p1, p2 = q['p1'], q['p2']
            y = torch.tensor([1.0 if len(p1)<len(p2) else 0.0], device=device).float()
            # reuse policy head: map to a logit via a tiny projection
            logit = out['policy'].mean()
            loss_inf = loss_inf + F.binary_cross_entropy_with_logits(logit.view_as(y), y)
    loss_inf = loss_inf / max(1,bsz)

    # Explicability surrogate: higher for shorter, smoother plans (encourage 0-1 near 1)
    explic_targets=[]
    for q in batch_q:
        p = targets['oracle_plan'](q)
        # explicability proxy = 1 / (1 + edits + bends)
        edits = len(p)
        explic_targets.append(1.0/(1.0+edits))
    explic_targets = torch.tensor(explic_targets, device=device).float().view(-1,1)
    # ask model once (use last forward)
    explic_preds=[]
    for q in batch_q:
        explic_preds.append(model(q)['explic'])
    explic_preds = torch.stack([e.view(1) for e in explic_preds], dim=0) # [B,1]
    loss_explic = F.mse_loss(explic_preds, explic_targets)

    # TD control (optional): simple one-step TD on a random transition batch (tiny)
    loss_td = torch.tensor(0.0, device=device)
    for _ in range(max(1, bsz//2)):
        s = random.choice(env._all_states)
        a = random.choice(ACTIONS)
        # step deterministically
        env.state = s
        s2, r, done, _ = env.step(a)
        with torch.no_grad():
            # bootstrap from model's Q on s2
            qn = model({'type':'policy','s':s2})['policy']
            bootstrap = 0.0 if done else torch.max(qn).item()
            y_val = r + 0.99 * bootstrap
        qsa = model({'type':'q','s':s,'a':a})['q'][a]
        y = torch.tensor(y_val, device=qsa.device, dtype=qsa.dtype)
        loss_td = loss_td + F.mse_loss(qsa, y)
    loss_td = loss_td / max(1, bsz//2)

    return w.inf*loss_inf + w.explic*loss_explic + w.td*loss_td, dict(inf=float(loss_inf.item()), explic=float(loss_explic.item()), td=float(loss_td.item()))


# ------------------------- Active coverage selection -------------------------

def select_queries_active_coverage(env:GridWorld, mmp:MultiMetricProgression, VQ:Tuple[Dict,Dict], batch_size:int=16):
    """
    Simple heuristic: sample candidates; pick a progressive sequence that
    maximizes diversity over locations and types while minimizing progression cost.
    """
    V, Q = VQ
    # candidate pool
    cand=[]
    states = random.sample(env._all_states, k=min(len(env._all_states), 20))
    for s in states:
        cand += make_point_queries(env, [s], ACTIONS[:2])
        cand += make_set_queries(env, [s], k=3)
    # random path queries
    for _ in range(6):
        s = random.choice(env._all_states); g = random.choice(env._all_states)
        cand += make_path_queries(env, [(s,g)], cost_per_step=abs(env.cfg.step_cost))
    # comparative
    for _ in range(4):
        s = random.choice(env._all_states)
        p1 = env.shortest_path(s, env.cfg.goal) or []
        p2 = p1[:-1]
        cand += make_comparative_queries(env, [(p1,p2)], s)

    random.shuffle(cand)
    cand = cand[:max(batch_size*3, 32)]
    # order by MMP
    ordered = mmp.order_queries_progressively(cand)
    return ordered[:batch_size]


# ------------------------- Evaluation metrics -------------------------

def evaluate_query_answering(model:QDIN, env:GridWorld, queries:List[Dict]):
    V,Q = compute_value_fn(env)
    correct=0; total=0
    set_iou=[]
    for q in queries:
        out = model(q)
        t = q['type']
        total+=1
        if t=='value':
            y=V[q['s']]; pred=float(out['value'].detach().cpu().view(-1)[0])
            ok = abs(pred - y) < 1.0
        elif t=='q':
            y=Q[(q['s'],q['a'])]; pred=float(out['q'][q['a']].detach().cpu())
            ok = abs(pred - y) < 1.0
        elif t=='policy':
            s=q['s']; best=max(ACTIONS, key=lambda a:Q[(s,a)])
            pred=int(torch.argmax(out['policy']).item())
            ok = (pred==best)
        elif t=='reachable':
            k=q.get('k',3)
            R=env.k_step_reachable(q['s'],k)
            mask_true=set([model.state_idx[s] for s in R])
            pred=(out['set_logits']>0).nonzero().view(-1).tolist()
            inter=len(set(pred)&mask_true); union=len(set(pred)|mask_true) if (set(pred)|mask_true) else 1
            iou = inter/union
            set_iou.append(iou); ok = iou>0.5
        elif t=='pathcost':
            P=env.shortest_path(q['s'],q['g']) or []
            y=float(len(P)*abs(env.cfg.step_cost)); pred=float(out['pathcost'].detach().cpu().view(-1)[0])
            ok = abs(pred - y) < abs(env.cfg.step_cost)*2.0
        elif t=='compare':
            p1,p2=q['p1'],q['p2']; y=1.0 if len(p1)<len(p2) else 0.0
            pred=(torch.sigmoid(out['policy'].mean())>0.5).float().item()
            ok = (int(pred)==int(y))
        else:
            ok=False
        if ok: correct+=1
    acc = correct/max(1,total)
    mean_iou = float(np.mean(set_iou)) if set_iou else 0.0
    return {'acc':acc, 'set_mean_iou':mean_iou}


# ------------------------- Baseline DQN (minimal) -------------------------

class DQN(nn.Module):
    def __init__(self, env:GridWorld, hidden=128):
        super().__init__()
        self.env=env
        self.S=len(env._all_states); self.A=4
        self.embed = nn.Embedding(self.S, hidden)
        self.head = nn.Linear(hidden, self.A)
        self.s2i = {s:i for i,s in enumerate(env._all_states)}

    def forward(self, s):
        idx=torch.tensor([self.s2i[s]]).long().to(next(self.parameters()).device)
        x=self.embed(idx); x=torch.tanh(x)
        return self.head(x).squeeze(0)

    def act(self, s, eps=0.1):
        if random.random()<eps:
            return random.choice(ACTIONS)
        with torch.no_grad():
            q=self.forward(s); return int(torch.argmax(q).item())

def train_dqn(env:GridWorld, steps=2000, lr=1e-3, gamma=0.99):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dqn=DQN(env).to(device); opt=torch.optim.Adam(dqn.parameters(), lr=lr)
    s=env.reset()
    for t in range(steps):
        a=dqn.act(s, eps=max(0.01, 1.0 - t/steps))
        s2,r,done,_=env.step(a)
        with torch.no_grad():
            qn=dqn.forward(s2)
            y = r + (0.0 if done else gamma*torch.max(qn).item())
        qsa = dqn.forward(s)[a]
        loss = F.mse_loss(qsa, torch.tensor(y, device=device).float())
        opt.zero_grad(); loss.backward(); opt.step()
        s = env.reset() if done else s2
    return dqn


# ------------------------- Simple Experiment Runner -------------------------

class ExperimentTracker:
    def __init__(self):
        self.logs=[]
    def log(self, **kwargs):
        self.logs.append(kwargs)
    def summary(self):
        if not self.logs: return {}
        keys=self.logs[0].keys()
        return {k: float(np.mean([x[k] for x in self.logs if k in x])) for k in keys}

def build_default_env(seed=0):
    walls=[]
    H=W=8
    rng=random.Random(seed)
    # Add a few random walls
    for _ in range(8):
        r=rng.randint(0,H-1); c=rng.randint(0,W-1)
        if (r,c) not in [(0,0),(H-1,W-1)]: walls.append((r,c))
    cfg=GridConfig(H=H,W=W,walls=walls,start=(0,0),goal=(H-1,W-1),step_cost=-1.0,goal_reward=0.0,seed=seed)
    return GridWorld(cfg)


def make_ground_truth(env:GridWorld):
    V,Q = compute_value_fn(env)
    def oracle_plan(q):
        t=q['type']
        if t in ('value','policy','q','compare'):
            return env.shortest_path(q['s'], env.cfg.goal) or []
        if t=='pathcost':
            return env.shortest_path(q['s'], q['g']) or []
        if t=='reachable':
            # frontier-ish
            p = env.shortest_path(q['s'], env.cfg.goal) or []
            return p[:q.get('k',3)]
        return []
    return {'V':V,'Q':Q,'oracle_plan':oracle_plan}
