import numpy as np
from numpy import transpose
from numpy.linalg import inv
from scipy.stats import norm


def eta(T):
    """
    Generates the cutoff probabilities for exploration rounds in interval
    chaining.

    :param T: the total number of iterations
    """
    return np.array([pow(t, -1/3) for t in range(1, T+1)])


def beta(k, d, c):
    """
    Generates the scaled down feature weights for a true model from the
    distribution β ∼ U[0, c]^d.

    :param k: the number of arms
    :param d: the number of features
    :param c: the scale of the feature weights
    """
    return np.random.uniform(0, c+1, size=(k, d))


def top_interval(X, Y, k, d, _delta, T):
    """
    Simulates T rounds of TopInterval for k.

    :param X: a 3-axis (T, k, d) ndarray of d-dimensional context vectors for
              each time-step and arm
    :param Y: a T x k ndarray of reward function output for each context vector
    :param k: the number of arms
    :param d: the number of features
    :param _delta: confidence parameter
    :param T: the number of iterations
    """
    _eta = eta(T)  # exploration cutoff probabilities
    picks = []
    for t in range(T):
        print('Iteration [{0} / {1}]'.format(t, T))
        if t <= d or np.random.rand() <= _eta[t]:
            # Play uniformly at random from [1, k].
            picks.append(np.random.randint(0, k))
            print('Exploration round.')
        else:
            intervals = []
            for i in range(k):
                # Compute beta hat.
                _Xti = X[:t+1, i]
                _XtiT = transpose(_Xti)
                try:
                    _XTX = inv(_XtiT.dot(_Xti))
                except:
                    print('Encountered singular matrix. Ignoring.')
                    continue
                _Yti = Y[:t+1, i]
                Bh_t_i = _XTX.dot(_XtiT).dot(_Yti)  # Compute OLS estimators.
                yh_t_i = Bh_t_i.dot(X[t, i])
                _s2 = np.var(Y[:t+1, i])
                # Compute the confidence interval width using the inverse CDF.
                w_t_i = norm.ppf(1 - _delta/(2*T*k), loc=0,
                                 scale=np.sqrt(_s2 * X[t, i].dot(_XTX).dot(transpose(X[t, i]))))
                intervals.append([yh_t_i - w_t_i, yh_t_i + w_t_i])
            # Pick the agent with the largest upper bound.
            picks.append(np.argmax(np.array(intervals)[:, 1]) if intervals else np.random.randint(0, k))
            print('Intervals: {0}'.format(intervals))
    # Compute sum of best picks over each iteration.
    best = [Y[i].max() for i in range(2, T)]
    performance = [Y[t][picks[t-2]] for t in range(2, T)]
    print('Cumulative Regret: {0}'.format(sum(best) - sum(performance)))
    print('Final Regret: {0}'.format(best[-1] - performance[-1]))


def compute_chain(i_st, intervals, k):
    # Sort intervals by decreasing order.
    chain = [i_st]
    print(intervals[:, 1])
    ordering = np.argsort(intervals[:, 1])[::-1]
    intervals = intervals[ordering, :]

    lowest_in_chain = intervals[0][0]
    for i in range(1, k):
        if intervals[i][1] >= lowest_in_chain:
            chain.append(i)
            lowest_in_chain = min(lowest_in_chain, intervals[i][0])
        else:
            return chain
    return chain


def interval_chaining(c, k, d, _delta, T):
    """
    Simulates T rounds of interval chaining.
    """
    X = np.random.uniform(0, 1, size=(k, T, d))  # 3-axis ndarray
    _eta = eta(T)  # exploration cutoff probabilities
    B = beta(k, d, c)  # true parameters. B[i]: params for arm i
    Y = np.array([X[i].dot(transpose(B[i])) for i in range(k)])  # not sure if there's a cleaner way to do this
    picks = []
    for t in range(T):
        print('Iteration [{0} / {1}]'.format(t, T))
        if np.random.rand() <= _eta[t]:
            # Play uniformly at random from [1, k].
            picks.append(np.random.randint(0, k))
            print('Exploration round.')
        else:
            intervals = []
            for i in range(k):
                # Compute beta hat.
                _Xti = X[i][:t+1]
                _XtiT = transpose(_Xti)
                try:
                    _XTX = inv(_XtiT.dot(_Xti))
                    _Yti = Y[i][:t+1]
                    Bh_t_i = _XTX.dot(_XtiT).dot(_Yti)  # Compute OLS estimators.
                    yh_t_i = Bh_t_i.dot(X[i][t])
                    _s2 = np.var(Y[i][:t+1])
                    w_t_i = norm.ppf(
                            1 - _delta/(2*T*k),
                            loc=0,
                            scale=np.sqrt(_s2 * X[i][t].dot(_XTX).dot(transpose(X[i][t])))
                    )
                    intervals.append([yh_t_i - w_t_i, yh_t_i + w_t_i])
                except:
                    print('Encountered singular matrix. Defaulting to exploration round.')
                    intervals = None
                    break
            # Pick a random uniformly at random from the chain containing the highest quality individual.
            if not intervals:
                picks.append(np.random.randint(0, k))
            else:
                i_st = np.argmax(np.array(intervals)[:, 1])
                chain = compute_chain(i_st, np.array(intervals), k)
                print('Computed chain: {0}'.format(chain))
                picks.append(np.random.choice(chain))
            print('Intervals: {0}'.format(intervals))
    # Compute sum of best picks over each iteration.
    best = [transpose(Y)[i].max() for i in range(2, T)]
    performance = [Y[picks[t-2]][t] for t in range(2, T)]
    print('Cumulative Regret: {0}'.format(sum(best) - sum(performance)))
    print('Final Regret: {0}'.format(best[-1] - performance[-1]))

