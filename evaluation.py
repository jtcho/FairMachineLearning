import matplotlib.pyplot as plt
import numpy as np

from fairml import beta, compute_chain, eta, interval_chaining, top_interval


def main():
    # X = np.random.uniform(0, 1, size=(T, k, d))  # 3-axis ndarray
    # B = beta(k, d, c)  # true parameters. B[i]: params for arm i
    # Y = np.array([np.diag(X[t].dot(np.transpose(B))) for t in range(T)])

    # cum_regret, avg_regret, final_regret = top_interval(X, Y, k, d, 0.05, T,
    #                                                     _print_progress=False)
    # print('Cumulative regret: %f' % cum_regret)
    # print('Average regret: %f' % avg_regret)
    # print('Final regret: %f' % final_regret)

    c_vals = [1.0, 2.0, 5.0, 10.0]

    # Plot: Varying T (# rounds)
    d = 2
    k = 2
    T_vals = range(3, 1000)
    results = []
    for c in c_vals:
        result = []
        for T in T_vals:
            X = np.random.uniform(0, 1, size=(T, k, d))  # 3-axis ndarray
            B = beta(k, d, c)  # true parameters. B[i]: params for arm i
            Y = np.array([np.diag(X[t].dot(np.transpose(B))) for t in range(T)])

            cum_regret_ti, avg_regret_ti, final_regret_ti = top_interval(
                    X, Y, k, d, 0.05, T, _print_progress=False)
            cum_regret_ic, avg_regret_ic, final_regret_ic = interval_chaining(
                    X, Y, c, k, d, 0.05, T, _print_progress=False)
            result.append(abs(avg_regret_ti - avg_regret_ic))
        results.append(result)

    for i in range(0, len(c_vals)):
        plt.plot(T_vals, results[i])

    # Plot: Varying k (# groups)
    # d = 2
    # k_vals = range(1, 50)
    # T = 1000
    # for c in c_vals:
    #     for k in k_vals:

    # Plot: Varying d (confidence)
    # d_vals = range(1, 50)
    # k = 2
    # T = 1000
    # for c in c_vals:
    #    for d in d_vals:



if __name__ == '__main__':
    main()
