# Value Iteration for Tic Tac Toe
# Reward for getting 3 in a row
# X always goes first, O goes second
# State is a 9-vector (row-major) representing what is in all 9 cells
# 0 = empty, 1 = X, 2 = O
# Transition is deterministic

EMPTY = 0
X = 1
O = 2

import os
import time

from collections import defaultdict

from pathlib import Path
#from tabulate import tabulate

import numpy as np

#from joblib import Memory
#mem = Memory(Path(os.getenv('HOME')) / '.cache' / 'joblib', verbose=0)

tocheck = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
           (0, 3, 6), (1, 4, 7), (2, 5, 8),
           (0, 4, 8), (2, 4, 6)]

tocheck = [np.array(tc) for tc in tocheck]

def empty():
  return np.zeros(9, dtype=int)

def checkwin(s):
  for tc in tocheck:
    cells = s[tc]
    if EMPTY in cells:
      continue

    if np.all(cells == X):
      return X
    if np.all(cells == O):
      return O

  return 0

def c(s):
  winner = checkwin(s)
  if winner == X:
    return 0
  if winner == O:
    return 1

  return 0.1

def actions(s):
  """
    get valid actions from a state s
  """
  inds = []
  for i in range(len(s)):
    if not s[i]:
      inds.append(i)

  return inds

def apply(s, i, p):
  ns = s.copy()
  ns[i] = p
  return ns

def opp_random(s):
  """ Plays randomly """
  return np.random.choice(actions(s))

def opp_best(s0, V):
  """ Plays optimally according to opponent value function """
  acts = np.random.permutation(actions(s0))
  best = None
  bestval = None
  for act in acts:
    s = apply(s0, act, O)
    c = V[tuple(s)]
    if best is None or c > bestval:
      best = act
      bestval = c

  return best

def printstate(s):
  ps = np.empty(s.shape, dtype=object)
  ps[s == X] = "X"
  ps[s == O] = "O"
  print(tabulate(ps.reshape((3, 3)), tablefmt="grid"))

#@mem.cache
def getstates(s):
  return _getstates(s, p=0)

def _getstates(s, p=0):
  states = [s]
  if checkwin(s):
    return states

  for i in range(len(s)):
    if not s[i]:
      next_s = s.copy()
      next_s[i] = p + 1

      states.extend(_getstates(next_s, 1 - p))

  return states

s0 = empty()

svf = "states"

print("Loading TicTacToe states...")

if not os.path.exists(svf):
  t0 = time.process_time()

  states = getstates(s0)
  N_raw = len(states)

  t1 = time.process_time()

  states = [np.array(s) for s in dict.fromkeys(map(tuple, states))]
  N = len(states)


  t2 = time.process_time()

  print(f"Time to enum states ({N_raw}):", t1 - t0)
  print(f"Time to dedup states ({N}):", t2 - t1)

  f = open(svf, "wb")
  np.save(f, np.array(states))
  f.close()

else:
  f = open(svf, "rb")
  states = list(np.load(f))
  f.close()

print(len(states), "states")

V = defaultdict(float)

N_vi = 10
gamma = 1

# Value Iteration
# Assume opponent plays uniformly random
for i in range(N_vi):
  print("Value iteration", i)
  updated = 0
  for s in np.random.permutation(states):
    hs = tuple(s)
    old_v = V[hs]
    V[hs] = c(s)
    best = None
    bestval = None
    acts = actions(s)
    if not acts:
      continue

    for j in acts:
      nexts = s.copy()
      nexts[j] = X
      val = 0
      opp_as = actions(nexts)
      if not opp_as:
        val += c(nexts)
      else:
        for k in opp_as:
          nexts2 = nexts.copy()
          nexts2[k] = O

          val += V[tuple(nexts2)]

        val /= len(opp_as)

      if best is None or val < bestval:
        best = j
        bestval = val

    V[hs] += gamma * bestval
    if abs(V[hs] - old_v) > 1e-5:
      updated += 1

  if not updated:
    print("Converged!")
    break
  else:
    print(f"{updated} states updated")

def extract(V):
  """ Given value function V, extracts policy pi """
  pi = dict()

  for s in states:
    cost = c(s)
    best = None
    bestval = None
    for j in actions(s):
      nexts = s.copy()
      nexts[j] = X
      val = 0
      opp_as = actions(nexts)
      if not opp_as:
        val += c(nexts)
      else:
        for k in opp_as:
          nexts2 = nexts.copy()
          nexts2[k] = O

          val += V[tuple(nexts2)]

        val /= len(opp_as)

      if best is None:
        best = [j]
        bestval = val
      else:
        if abs(val - bestval) < 2e-3:
          best.append(j)
        elif val < bestval:
          best = [j]
          bestval = val

    pi[tuple(s)] = best

  return pi

pi = extract(V)

def playagainst(opp, N_games=2000):
  wins = 0
  ties = 0
  loss = 0
  for i in range(N_games):
    s = empty()
    while 1:
      acts = pi[tuple(s)]

      # Choose a random action from list of best actions
      a = np.random.choice(acts)

      s = apply(s, a, X)

      #print("choosing randomly from", acts)
      #printstate(s)
      #input()

      win = checkwin(s)
      if win:
        wins += 1
        break

      if not actions(s):
        ties += 1
        break

      a_opp = opp(s)
      s = apply(s, a_opp, O)

      win = checkwin(s)
      if win:
        loss += 1
        break

  print(f"Wins/Ties/Losses ({N_games} games) = {wins}/{ties}/{loss}")

N_games = 2000
print(f"Playing {N_games} games against random opponent")
playagainst(opp_random, N_games)
print(f"Playing {N_games} games against self")
playagainst(lambda s, V=V: opp_best(s, V), N_games)
