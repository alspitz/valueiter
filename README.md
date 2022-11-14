# valueiter

[valueiter.py](valueiter.py) builds a tabular value function V and then extracts the optimal policy pi.

It takes 5 iterations for value iteration to converge.

Requires numpy.

```shell
$ python valueiter.py
Loading TicTacToe states...
5478 states
Value iteration 0
4836 states updated
Value iteration 1
1063 states updated
Value iteration 2
173 states updated
Value iteration 3
15 states updated
Value iteration 4
1 states updated
Value iteration 5
Converged!
Playing 2000 games against random opponent
Wins/Ties/Losses (2000 games) = 1987/13/0
Playing 2000 games against self
Wins/Ties/Losses (2000 games) = 0/2000/0
```
