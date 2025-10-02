
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
    from torch.nn.utils import clip_grad_norm_
except Exception as e:
    raise RuntimeError("PyTorch is required. Please run install_requirements.py first.")


# ------------------------- Simple deterministic GridWorld -------------------------

Action = int  # 0:up,1:right,2:down,3:left
ACTIONS = [0,1,2,3]
ACTION_VECS = {0: (-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
ACTION_NAMES = {0:"U",1:"R",2:"D",3:"L"}

REACH_LOSS_SCALE = 0.3
ENTROPY_REG_WEIGHT = 1e-3
EXPLIC_TEMP_MIN = 0.5
EXPLIC_TEMP_MAX = 5.0


def log_event(tag: str, **kwargs: Any) -> None:
    """Light-weight structured logging helper."""
    fields = ", ".join(f"{k}={kwargs[k]}" for k in sorted(kwargs)) if kwargs else ""
    print(f"[LOG][{tag}] {fields}")

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


def _path_after_action(env:GridWorld, start:Tuple[int,int], action:int):
    r,c = start
    dr,dc = ACTION_VECS[action]
    nr,nc = r+dr, c+dc
    if env.in_bounds(nr,nc) and (nr,nc) not in env.walls:
        return (nr,nc)
    return start


def make_alternative_path(env:GridWorld, start:Tuple[int,int], base_path:List[int], max_extra_steps:int=3):
    """
    Create a path that diverges from ``base_path`` at some prefix but stays
    relatively close in length. Returns ``None`` if no viable alternative is
    found.
    """
    if not base_path:
        return None

    # Trace the states along the base path for convenience.
    states=[start]
    cur=start
    for a in base_path:
        cur=_path_after_action(env, cur, a)
        states.append(cur)

    for idx in range(len(base_path)):
        state = states[idx]
        preferred = base_path[idx]
        # try alternative moves except going immediately backwards
        alternatives=[]
        for a,(nr,nc) in env.neighbors(state):
            if a==preferred:
                continue
            if idx>0 and (nr,nc)==states[idx-1]:
                continue
            alternatives.append((a,(nr,nc)))
        random.shuffle(alternatives)
        for a,(nr,nc) in alternatives:
            tail = env.shortest_path((nr,nc), env.cfg.goal)
            if tail is None:
                continue
            alt_path = base_path[:idx] + [a] + tail
            if alt_path == base_path:
                continue
            if len(alt_path) <= len(base_path) + max_extra_steps:
                return alt_path
    return None


def sample_balanced_comparisons(env:GridWorld, n_pairs:int=4):
    """
    Sample comparative queries with non-degenerate, length-balanced pairs.
    """
    queries=[]
    attempts=0
    max_attempts=max(10, n_pairs*6)
    while len(queries)<n_pairs and attempts<max_attempts:
        attempts+=1
        s = random.choice(env._all_states)
        base = env.shortest_path(s, env.cfg.goal)
        if base is None or not base:
            continue
        alt = make_alternative_path(env, s, base)
        if not alt:
            continue
        if random.random()<0.5:
            p1,p2 = base, alt
        else:
            p1,p2 = alt, base
        queries.append({'type':'compare','s':s,'p1':p1,'p2':p2})
    return queries


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
            'levenshtein': 0.0 if not include_lev else 0.01,
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

        # state embedding: index of state as one-hot -> MLP + structured context
        self.state_idx = {s:i for i,s in enumerate(env._all_states)}
        self.embed_s = nn.Embedding(self.S, hidden)
        self.coord_proj = nn.Linear(2, hidden)

        conv_channels = max(4, hidden // 4)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(3, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, hidden, kernel_size=1),
            nn.ReLU(),
        )
        occ = torch.zeros(3, self.env.cfg.H, self.env.cfg.W)
        for (r,c) in self.env.walls:
            occ[0, r, c] = 1.0
        sr, sc = self.env.start
        gr, gc = self.env.goal
        occ[1, sr, sc] = 1.0
        occ[2, gr, gc] = 1.0
        self.register_buffer('occ_grid', occ.unsqueeze(0))
        self._cached_grid_feat: Optional[torch.Tensor] = None
        self._cached_grid_device: Optional[torch.device] = None

        # query embeddings: type id + a small numeric vector
        self.type2id = {'value':0,'q':1,'policy':2,'reachable':3,'pathcost':4,'compare':5}
        self.embed_t = nn.Embedding(len(self.type2id), hidden)
        self.fc_q = nn.Linear(6, hidden)  # room for small query params

        # learned "model": transition logits and rewards (masked by valid neighbors)
        neighbor_lists=[]
        for s in env._all_states:
            candidates=[]
            r,c=s
            for a in ACTIONS:
                dr,dc=ACTION_VECS[a]
                nr,nc=r+dr,c+dc
                if env.in_bounds(nr,nc) and (nr,nc) not in env.walls:
                    candidates.append(self.state_idx[(nr,nc)])
            if not candidates:
                candidates.append(self.state_idx[s])
            neighbor_lists.append(candidates)
        self.max_neighbors=max(len(c) for c in neighbor_lists)
        neighbor_idx=torch.full((self.S, self.max_neighbors), -1, dtype=torch.long)
        neighbor_counts=[]
        for i,cands in enumerate(neighbor_lists):
            count=len(cands)
            neighbor_counts.append(count)
            neighbor_idx[i, :count]=torch.tensor(cands, dtype=torch.long)
        self.register_buffer('neighbor_idx', neighbor_idx)
        self.register_buffer('neighbor_counts', torch.tensor(neighbor_counts, dtype=torch.long))

        self.T_logits = nn.Parameter(torch.zeros(self.A, self.S, self.max_neighbors))  # a -> s -> neighbors
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
        self.h_explic = nn.Linear(hidden, 1)
        self.log_explic_temp = nn.Parameter(torch.zeros(1))

        for head in [self.h_value, self.h_q, self.h_policy, self.h_explic]:
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.constant_(head.bias, 0.0)

    def transition_matrix(self):
        device = self.T_logits.device
        T_logits = torch.full((self.A, self.S, self.S), float('-inf'), device=device)
        for s_idx in range(self.S):
            count = int(self.neighbor_counts[s_idx].item())
            if count == 0:
                continue
            idxs = self.neighbor_idx[s_idx, :count]
            T_logits[:, s_idx, idxs] = self.T_logits[:, s_idx, :count]
        return torch.softmax(T_logits, dim=-1)

    def reachability_from_T(self, T:torch.Tensor, s:Tuple[int,int], k:int):
        device = T.device
        p = torch.zeros(self.S, device=device)
        p[self.state_idx[s]] = 1.0
        for _ in range(k):
            p = torch.einsum('ask,s->k', T, p)
            p = torch.clamp(p, 0.0, 1.0)
        return p

    def vi_block(self, gamma=0.99):
        # Perform K value-iteration steps over learned (masked) model
        T = self.transition_matrix()  # [A,S,S]
        R = self.R  # [S,A]
        V = torch.zeros(self.S, device=R.device)
        for _ in range(self.K):
            Q = R + gamma * torch.einsum('ask,k->sa', T, V)  # [S,A]
            V = torch.max(Q, dim=1).values
        return V, Q, T  # [S], [S,A], [A,S,S]

    def expected_path_cost(self, start: Tuple[int, int], actions: List[int], T: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the expected cumulative *cost* (negative reward) of following ``actions``."""
        if T is None:
            T = self.transition_matrix()
        device = T.device
        state_dist = torch.zeros(self.S, device=device)
        state_dist[self.state_idx[start]] = 1.0
        total_cost = torch.zeros(1, device=device)
        for a in actions:
            a = int(a)
            reward = torch.matmul(state_dist, self.R[:, a])
            total_cost = total_cost - reward
            state_dist = torch.matmul(state_dist, T[a])
        return total_cost.squeeze(0)

    @torch.no_grad()
    def greedy_action(self, s: Tuple[int, int]) -> int:
        q_vals = self({'type': 'q', 's': s})['q']
        return int(torch.argmax(q_vals).item())

    @torch.no_grad()
    def q_values(self, s: Tuple[int, int]) -> torch.Tensor:
        return self({'type': 'q', 's': s})['q']

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

    def _get_grid_features(self):
        device = next(self.parameters()).device
        occ = self.occ_grid.to(device)
        if self.training:
            # When training we must recompute the convolutional features on every
            # forward pass so that gradients can flow through the convolutional
            # parameters. Caching tensors that require gradients would cause
            # autograd to reuse a graph from a previous iteration and error out
            # during ``backward``.
            return self.grid_conv(occ)

        # During evaluation we can safely cache the features to avoid the extra
        # convolution as no gradients are required.
        if (self._cached_grid_feat is None) or (self._cached_grid_device != device):
            with torch.no_grad():
                self._cached_grid_feat = self.grid_conv(occ)
            self._cached_grid_device = device
        return self._cached_grid_feat

    def encode_state(self, s:Tuple[int,int]):
        i = self.state_idx[s]
        device = next(self.parameters()).device
        idx = torch.tensor([i], device=device).long()
        base = self.embed_s(idx)  # [1,hidden]

        H = max(1, self.env.cfg.H - 1)
        W = max(1, self.env.cfg.W - 1)
        r, c = s
        coord = torch.tensor([[r / H, c / W]], device=device).float()
        coord_feat = torch.tanh(self.coord_proj(coord))

        grid_feat = self._get_grid_features()[:, :, r, c]
        return base + coord_feat + grid_feat  # [1,hidden]

    def forward(self, q:Dict[str,Any]):
        device = next(self.parameters()).device
        # run VI block to get global value landscape
        V_all, Q_all, T_all = self.vi_block()
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
        k = q.get('k', 3)
        reach_probs = self.reachability_from_T(T_all, s, k)
        reach_logits = torch.logit(reach_probs.clamp(1e-6, 1-1e-6))
        out['set_logits'] = reach_logits                  # [S]
        path_actions = q.get('path_actions')
        if path_actions is None:
            if q['type'] == 'pathcost':
                path_actions = self.env.shortest_path(s, q.get('g', self.env.cfg.goal)) or []
            elif q['type'] == 'compare':
                path_actions = q.get('p1', [])
            else:
                path_actions = []
        path_cost = self.expected_path_cost(s, path_actions, T_all)
        out['pathcost'] = path_cost

        temp = torch.exp(self.log_explic_temp).clamp(EXPLIC_TEMP_MIN, EXPLIC_TEMP_MAX)
        explic_logits = self.h_explic(x) / temp
        out['explic'] = torch.sigmoid(explic_logits).squeeze(0) # [1] in [0,1]
        return out


# ------------------------- Losses and training utilities -------------------------

@dataclass
class LossWeights:
    td: float = 0.2
    inf: float = 1.0
    explic: float = 0.1
    model: float = 0.25

    def as_dict(self) -> Dict[str, float]:
        explic_w = float(min(self.explic, 0.1))
        return {
            'td': float(self.td),
            'inf': float(self.inf),
            'explic': explic_w,
            'model': float(self.model),
        }


class MultiTaskLossBalancer(nn.Module):
    """Homoscedastic uncertainty weighting for multi-head losses."""

    def __init__(self, task_names:List[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })
        if not self.log_vars:
            # maintain at least one parameter to keep device tracking simple
            self.log_vars['dummy'] = nn.Parameter(torch.zeros(1), requires_grad=False)

    def active_tasks(self) -> List[str]:
        return [k for k in self.log_vars.keys() if k != 'dummy']

    def combine(self, losses:Dict[str, torch.Tensor], base_weights:Dict[str, float]):
        device = next(iter(self.log_vars.values())).device
        total = torch.zeros(1, device=device)
        scaled = {}
        for name, loss in losses.items():
            weight = base_weights.get(name, 0.0)
            if weight <= 0.0:
                continue
            if name not in self.log_vars:
                # lazily add if not present (e.g., late-phase task)
                self.log_vars[name] = nn.Parameter(torch.zeros(1, device=device))
            log_var = self.log_vars[name]
            inv_var = torch.exp(-log_var)
            total = total + weight * inv_var * loss + log_var
            scaled[name] = float((weight * inv_var * loss).item())
        return total.squeeze(0), scaled


def _smooth_binary_targets(target: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    return target * (1.0 - eps) + 0.5 * eps


def path_cost_ground_truth(env: GridWorld, start: Tuple[int, int], actions: List[int]) -> float:
    cur = start
    total = 0.0
    for a in actions:
        dr, dc = ACTION_VECS[a]
        nr, nc = cur[0] + dr, cur[1] + dc
        if env.in_bounds(nr, nc) and (nr, nc) not in env.walls:
            nxt = (nr, nc)
        else:
            nxt = cur
        total -= env.cfg.step_cost
        cur = nxt
        if cur == env.cfg.goal:
            total -= env.cfg.goal_reward
            break
    return float(total)


def inference_aware_loss(model:QDIN, env:GridWorld, batch_q:List[Dict], targets:Dict[str,Any], w:LossWeights, balancer:Optional[MultiTaskLossBalancer]=None):
    device = next(model.parameters()).device
    bsz = len(batch_q)
    cached_outputs: List[Dict[str, torch.Tensor]] = []
    loss_inf = torch.tensor(0.0, device=device)
    for q in batch_q:
        out = model(q)
        cached_outputs.append(out)
        t = q['type']
        if t=='value':
            y = torch.tensor([targets['V'][q['s']]], device=device).float()
            loss_inf = loss_inf + F.mse_loss(out['value'].view_as(y), y)
        elif t=='q':
            y = torch.tensor([targets['Q'][(q['s'],q['a'])]], device=device).float()
            qsa = out['q'][q['a']]
            loss_inf = loss_inf + F.mse_loss(qsa.view_as(y), y)
        elif t=='policy':
            s=q['s']
            best = max(ACTIONS, key=lambda a: targets['Q'][(s,a)])
            logits = out['policy'].unsqueeze(0)
            y = torch.tensor([best], device=device).long()
            loss_inf = loss_inf + F.cross_entropy(logits, y)
        elif t=='reachable':
            k=q.get('k',3)
            R = env.k_step_reachable(q['s'], k)
            mask = torch.zeros(model.S, device=device)
            idxs=[model.state_idx[s] for s in R]
            mask[idxs]=1.0
            pred = out['set_logits']
            pos = mask.sum()
            neg = mask.numel() - pos
            pos_weight = ((neg + 1.0) / (pos + 1.0)).to(device)
            target = _smooth_binary_targets(mask, eps=0.05)
            reach_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
            loss_inf = loss_inf + REACH_LOSS_SCALE * reach_loss
        elif t=='pathcost':
            plan = targets['oracle_plan'](q)
            gt_cost = torch.tensor([path_cost_ground_truth(env, q['s'], plan)], device=device).float()
            pred_cost = out['pathcost'].view_as(gt_cost)
            loss_inf = loss_inf + F.mse_loss(pred_cost, gt_cost)
        elif t=='compare':
            p1, p2 = q['p1'], q['p2']
            cost_a = model.expected_path_cost(q['s'], p1)
            cost_b = model.expected_path_cost(q['s'], p2)
            logits = torch.stack([-cost_a, -cost_b]).unsqueeze(0)
            true_cost_a = path_cost_ground_truth(env, q['s'], p1)
            true_cost_b = path_cost_ground_truth(env, q['s'], p2)
            y = torch.tensor([0 if true_cost_a <= true_cost_b else 1], device=device).long()
            loss_inf = loss_inf + F.cross_entropy(logits, y)
    loss_inf = loss_inf / max(1,bsz)

    explic_targets=[]
    for q in batch_q:
        p = targets['oracle_plan'](q)
        edits = len(p)
        explic_targets.append(1.0/(1.0+edits))
    explic_targets = torch.tensor(explic_targets, device=device).float().view(-1,1)
    explic_preds = torch.stack([out['explic'].view(1) for out in cached_outputs], dim=0)
    loss_explic = F.mse_loss(explic_preds, explic_targets)

    loss_td = torch.tensor(0.0, device=device)
    for _ in range(max(1, bsz//2)):
        s = random.choice(env._all_states)
        a = random.choice(ACTIONS)
        env.state = s
        s2, r, done, _ = env.step(a)
        with torch.no_grad():
            qn = model({'type':'q','s':s2})['q']
            bootstrap = 0.0 if done else torch.max(qn).item()
            y_val = r + 0.99 * bootstrap
        qsa = model({'type':'q','s':s,'a':a})['q'][a]
        y = torch.tensor(y_val, device=qsa.device, dtype=qsa.dtype)
        loss_td = loss_td + F.mse_loss(qsa, y)
    loss_td = loss_td / max(1, bsz//2)

    loss_model_T = torch.tensor(0.0, device=device)
    loss_model_R = torch.tensor(0.0, device=device)
    n_transitions = max(1, bsz)
    if w.model > 0.0:
        T_matrix = model.transition_matrix()
        entropy = -(T_matrix.clamp_min(1e-8) * torch.log(T_matrix.clamp_min(1e-8))).sum(dim=-1).mean()
        for _ in range(n_transitions):
            s = random.choice(env._all_states)
            a = random.choice(ACTIONS)
            env.state = s
            s2, r, _, _ = env.step(a)
            si = model.state_idx[s]
            s2i = model.state_idx[s2]
            prob = T_matrix[a, si, s2i].clamp_min(1e-8)
            loss_model_T = loss_model_T - torch.log(prob)
            r_pred = model.R[si, a]
            loss_model_R = loss_model_R + F.mse_loss(r_pred, torch.tensor(float(r), device=device))
        loss_model_T = loss_model_T / n_transitions
        loss_model_R = loss_model_R / n_transitions
        loss_model = loss_model_T + loss_model_R + ENTROPY_REG_WEIGHT * entropy
    else:
        entropy = torch.tensor(0.0, device=device)
        loss_model = torch.tensor(0.0, device=device)

    losses = {
        'inf': loss_inf,
        'explic': loss_explic,
        'td': loss_td,
        'model': loss_model,
    }
    base_weights = w.as_dict()
    active_losses = {k: v for k, v in losses.items() if base_weights.get(k, 0.0) > 0.0}
    if balancer is not None and active_losses:
        total_loss, scaled_parts = balancer.combine(active_losses, base_weights)
    else:
        total_loss = torch.tensor(0.0, device=loss_inf.device)
        scaled_parts = {}
        for name, loss_val in active_losses.items():
            weight = base_weights.get(name, 1.0)
            total_loss = total_loss + weight * loss_val
            scaled_parts[name] = float((weight * loss_val).item())

    if not torch.isfinite(total_loss):
        diag = {k: float(v.detach().item()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
        diag.update({f'scaled_{k}': v for k, v in scaled_parts.items()})
        log_event("NON_FINITE", **diag)
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    parts = dict(
        inf=float(loss_inf.item()),
        explic=float(loss_explic.item()),
        td=float(loss_td.item()),
        model=float(loss_model.item()),
        model_T=float(loss_model_T.item()),
        model_R=float(loss_model_R.item()),
        entropy=float(entropy.item()),
    )
    for name, val in scaled_parts.items():
        parts[f'weighted_{name}'] = val
    return total_loss, parts


# ------------------------- Active coverage selection -------------------------

CURRICULUM_PHASES = [
    ("point", 0.25),
    ("reach", 0.55),
    ("path", 0.8),
    ("compare", 1.0),
]


def curriculum_phase(ep:int, total:int) -> Tuple[str, float]:
    if total <= 0:
        return CURRICULUM_PHASES[-1][0], 1.0
    ratio = (ep + 1) / total
    ratio = min(max(ratio, 0.0), 1.0)
    prev = 0.0
    for name, bound in CURRICULUM_PHASES:
        if ratio <= bound:
            span = max(bound - prev, 1e-6)
            progress = (ratio - prev) / span
            return name, float(min(max(progress, 0.0), 1.0))
        prev = bound
    return CURRICULUM_PHASES[-1][0], 1.0


def progressive_reach_ks(progress:float, include_history:bool=False) -> List[int]:
    if progress < 0.34:
        ks = [2]
    elif progress < 0.67:
        ks = [3]
    else:
        ks = [5]
    if include_history:
        history = [k for k in [2, 3, 5] if k <= ks[-1]]
        ks = sorted(set(history + ks))
    return ks


def enforce_prefix_stability(ordered:List[Dict], mmp:MultiMetricProgression, min_prefix:int=1) -> List[Dict]:
    if not ordered:
        return []
    stabilized = [ordered[0]]
    remaining = ordered[1:]
    while remaining:
        prev_plan = mmp.plan_for_query(stabilized[-1])
        prefix_len = min(min_prefix, len(prev_plan))
        prefix = tuple(prev_plan[:prefix_len]) if prefix_len > 0 else tuple()
        best_idx = None
        best_dist = float('inf')
        fallback_idx = None
        fallback_dist = float('inf')
        for idx, cand in enumerate(remaining):
            plan = mmp.plan_for_query(cand)
            dist = mmp.dist(stabilized[-1], cand)
            if dist < fallback_dist:
                fallback_dist = dist
                fallback_idx = idx
            if prefix:
                cand_prefix = tuple(plan[:prefix_len])
                if cand_prefix != prefix:
                    continue
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        choice = best_idx if best_idx is not None else fallback_idx
        stabilized.append(remaining.pop(choice))
    return stabilized


def select_queries_active_coverage(env:GridWorld, mmp:MultiMetricProgression, VQ:Tuple[Dict,Dict], batch_size:int=16, phase:str="full", phase_progress:float=1.0):
    """
    Sample curriculum-aware query batches ordered by MMP and stabilized via
    OEG-style prefix constraints.
    """
    include_point = phase in ("point", "reach", "path", "compare", "full")
    include_reach = phase in ("reach", "path", "compare", "full")
    include_path = phase in ("path", "compare", "full")
    include_compare = phase in ("compare", "full")

    cand: List[Dict] = []
    state_pool = env._all_states[:]
    random.shuffle(state_pool)

    if include_point:
        num_states = max(1, batch_size // 6)
        states = state_pool[:num_states]
        cand += make_point_queries(env, states, ACTIONS)

    if include_reach:
        num_states = max(1, batch_size // 4)
        states = state_pool[:num_states]
        ks = progressive_reach_ks(phase_progress, include_history=phase not in ("reach",))
        for s in states:
            for k in ks:
                cand.append({'type':'reachable','s':s,'k':k})

    if include_path:
        num_paths = max(1, batch_size // 6)
        for _ in range(num_paths * 2):
            s = random.choice(env._all_states)
            g = random.choice(env._all_states)
            if s == g:
                continue
            cand += make_path_queries(env, [(s, g)], cost_per_step=abs(env.cfg.step_cost))
            if len(cand) >= batch_size * 3:
                break

    if include_compare:
        num_pairs = max(1, batch_size // 6)
        cand += sample_balanced_comparisons(env, n_pairs=num_pairs)

    if not cand:
        return []

    max_candidates = max(batch_size * 3, 64)
    cand = cand[:max_candidates]
    ordered = mmp.order_queries_progressively(cand)
    stabilized = enforce_prefix_stability(ordered, mmp, min_prefix=1)
    return stabilized[:batch_size]


# ------------------------- Evaluation metrics -------------------------

def evaluate_query_answering(model:QDIN, env:GridWorld, queries:List[Dict]):
    V,Q = compute_value_fn(env)
    stats = {
        'value_abs': [],
        'q_abs': [],
        'policy_hits': [],
        'reach_iou': [],
        'path_abs': [],
        'compare_hits': [],
    }
    correct=0; total=0
    for q in queries:
        with torch.no_grad():
            out = model(q)
        t = q['type']
        total += 1
        if t=='value':
            y = V[q['s']]
            pred = float(out['value'].detach().cpu().view(-1)[0])
            stats['value_abs'].append(abs(pred - y))
            ok = abs(pred - y) < 1.0
        elif t=='q':
            y = Q[(q['s'],q['a'])]
            pred = float(out['q'][q['a']].detach().cpu())
            stats['q_abs'].append(abs(pred - y))
            ok = abs(pred - y) < 1.0
        elif t=='policy':
            s=q['s']; best=max(ACTIONS, key=lambda a:Q[(s,a)])
            pred=int(torch.argmax(out['policy']).item())
            hit = int(pred==best)
            stats['policy_hits'].append(hit)
            ok = bool(hit)
        elif t=='reachable':
            k=q.get('k',3)
            R=env.k_step_reachable(q['s'],k)
            mask_true=set([model.state_idx[s] for s in R])
            pred=(out['set_logits']>0).nonzero().view(-1).tolist()
            inter=len(set(pred)&mask_true); union=len(set(pred)|mask_true) if (set(pred)|mask_true) else 1
            iou = inter/union
            stats['reach_iou'].append(iou)
            ok = iou>0.5
        elif t=='pathcost':
            plan = env.shortest_path(q['s'], q['g']) or []
            y = path_cost_ground_truth(env, q['s'], plan)
            pred = float(out['pathcost'].detach().cpu())
            stats['path_abs'].append(abs(pred - y))
            ok = abs(pred - y) < max(1.0, abs(env.cfg.step_cost)*2.0)
        elif t=='compare':
            p1,p2=q['p1'],q['p2']
            true_cost_a = path_cost_ground_truth(env, q['s'], p1)
            true_cost_b = path_cost_ground_truth(env, q['s'], p2)
            cost_a_pred = float(out['pathcost'].detach().cpu())
            with torch.no_grad():
                cost_b_pred = float(model.expected_path_cost(q['s'], p2).detach().cpu())
            pred = 0 if cost_a_pred <= cost_b_pred else 1
            hit = int((true_cost_a <= true_cost_b and pred == 0) or (true_cost_b < true_cost_a and pred == 1))
            stats['compare_hits'].append(hit)
            ok = bool(hit)
        else:
            ok=False
        if ok: correct+=1
    acc = correct/max(1,total)
    summary = {
        'overall_acc': acc,
        'value_mae': float(np.mean(stats['value_abs'])) if stats['value_abs'] else 0.0,
        'q_mae': float(np.mean(stats['q_abs'])) if stats['q_abs'] else 0.0,
        'policy_acc': float(np.mean(stats['policy_hits'])) if stats['policy_hits'] else 0.0,
        'reach_mean_iou': float(np.mean(stats['reach_iou'])) if stats['reach_iou'] else 0.0,
        'path_mae': float(np.mean(stats['path_abs'])) if stats['path_abs'] else 0.0,
        'compare_acc': float(np.mean(stats['compare_hits'])) if stats['compare_hits'] else 0.0,
    }
    return summary


def rollouts_control_metrics(
    env: GridWorld,
    greedy_action_fn,
    q_fn=None,
    n_episodes: int = 10,
    max_steps: Optional[int] = None,
    enforce_greedy: bool = True,
):
    max_steps = max_steps or (4 * env.cfg.H * env.cfg.W)
    returns = []
    shaped_returns = []
    successes = 0
    steps_to_goal: List[int] = []
    for _ in range(n_episodes):
        s = env.reset()
        ret = 0.0
        shaped = 0.0
        reached = False
        for step in range(1, max_steps + 1):
            if s == env.cfg.goal:
                reached = True
                steps_to_goal.append(step - 1)
                break
            chosen = greedy_action_fn(s)
            if enforce_greedy and q_fn is not None:
                q_vals = q_fn(s)
                if isinstance(q_vals, torch.Tensor):
                    q_vals_cpu = q_vals.detach().cpu()
                else:
                    q_vals_cpu = torch.tensor(q_vals)
                greedy = int(torch.argmax(q_vals_cpu).item())
                if chosen != greedy:
                    log_event("NON_GREEDY_ACTION", chosen=chosen, greedy=greedy, state=s)
                    action = greedy
                else:
                    action = chosen
            else:
                action = chosen
            s, r, done, _ = env.step(action)
            ret += r
            shaped -= 0.01
            if done:
                reached = True
                shaped += 1.0
                steps_to_goal.append(step)
                break
        if not reached:
            steps_to_goal.append(max_steps)
        if reached:
            successes += 1
        returns.append(ret)
        shaped_returns.append(shaped)
    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_shaped = float(np.mean(shaped_returns)) if shaped_returns else 0.0
    goal_rate = successes / max(1, n_episodes)
    env.reset()
    return {
        'avg_return': avg_return,
        'avg_success_reward': avg_shaped,
        'goal_reach_rate': goal_rate,
        'steps_to_goal': steps_to_goal,
    }


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

    @torch.no_grad()
    def greedy_action(self, s: Tuple[int, int]) -> int:
        return int(torch.argmax(self.forward(s)).item())

    @torch.no_grad()
    def q_values(self, s: Tuple[int, int]) -> torch.Tensor:
        return self.forward(s)

def train_dqn(env:GridWorld, steps=2000, lr=1e-3, gamma=0.99):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dqn=DQN(env).to(device); opt=torch.optim.Adam(dqn.parameters(), lr=lr)
    s=env.reset()
    for t in range(steps):
        a=dqn.act(s, eps=max(0.01, 1.0 - t/steps))
        s2,r,done,_=env.step(a)
        with torch.no_grad():
            qn = dqn.forward(s2)
            y = r + (0.0 if done else gamma*torch.max(qn).item())
        qsa = dqn.forward(s)[a]
        loss = F.mse_loss(qsa, torch.tensor(y, device=device).float())
        opt.zero_grad(); loss.backward(); clip_grad_norm_(dqn.parameters(), 1.0); opt.step()
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
        # aggregate over all keys observed across the logs
        keys=set().union(*(entry.keys() for entry in self.logs))
        summary={}
        for k in keys:
            values=[entry[k] for entry in self.logs if k in entry]
            numeric=[]
            for v in values:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    numeric.append(float(v))
                else:
                    # keep track of non-numeric values to return a representative element
                    numeric=None
                    break
            if numeric:
                summary[k]=float(np.mean(numeric))
            else:
                summary[k]=values[-1]
        return summary

def build_default_env(seed=0):
    H=W=8
    rng=random.Random(seed)
    while True:
        walls=[]
        for _ in range(8):
            r=rng.randint(0,H-1); c=rng.randint(0,W-1)
            if (r,c) not in [(0,0),(H-1,W-1)]:
                if (r,c) not in walls:
                    walls.append((r,c))
        cfg=GridConfig(H=H,W=W,walls=walls,start=(0,0),goal=(H-1,W-1),step_cost=-1.0,goal_reward=0.0,seed=seed)
        env=GridWorld(cfg)
        if env.shortest_path(cfg.start, cfg.goal) is not None:
            return env


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
