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

    # Plot: Varying T (# of rounds)
    d = 2
    k = 2
    T_vals = range(3, 1000)

    results = {
        '0': {
            'ylabel': 'Average regret - TI',
            'name': 'avg_regret_ti'
        },
        '1': {
            'ylabel': 'Average regret - IC',
            'name': 'avg_regret_ic'
        },
        '2': {
            'ylabel': 'Average regret difference (TI - IC)',
            'name': 'avg_regret_diff'
        },
        '3': {
            'ylabel': 'Cumulative regret - TI',
            'name': 'cum_regret_ti'
        },
        '4': {
            'ylabel': 'Cumulative regret - IC',
            'name': 'cum_regret_ic'
        },
        '5': {
            'ylabel': 'Cumulative regret difference (TI - IC)',
            'name': 'cum_regret_diff'
        },
        '6': {
            'ylabel': 'Final regret - TI',
            'name': 'final_regret_ti'
        },
        '7': {
            'ylabel': 'Final regret - IC',
            'name': 'final_regret_ic'
        },
        '8': {
            'ylabel': 'Final regret difference (TI - IC)',
            'name': 'final_regret_diff'
        }
    }
    for _, v in results.items():  # 9 sets of results.
        for j in c_vals:
            v[str(j)] = []

    for c in c_vals:
        for T in T_vals:
            X = np.random.uniform(0, 1, size=(T, k, d))
            B = beta(k, d, c)
            Y = np.array([np.diag(X[t].dot(np.transpose(B))) for t in range(T)])

            cum_regret_ti, avg_regret_ti, final_regret_ti = top_interval(
                    X, Y, k, d, 0.05, T, _print_progress=False)
            cum_regret_ic, avg_regret_ic, final_regret_ic = interval_chaining(
                    X, Y, c, k, d, 0.05, T, _print_progress=False)

            results['0'][str(c)].append(avg_regret_ti)
            results['1'][str(c)].append(avg_regret_ic)
            results['2'][str(c)].append(abs(avg_regret_ti - avg_regret_ic))
            results['3'][str(c)].append(cum_regret_ti)
            results['4'][str(c)].append(cum_regret_ic)
            results['5'][str(c)].append(abs(cum_regret_ti - cum_regret_ic))
            results['6'][str(c)].append(final_regret_ti)
            results['7'][str(c)].append(final_regret_ic)
            results['8'][str(c)].append(abs(final_regret_ti - final_regret_ic))

    for k, v in results.items():
        plt.clf()
        c1, = plt.plot(T_vals, results[k]['1.0'], label='c=1')
        c2, = plt.plot(T_vals, results[k]['2.0'], label='c=2')
        c5, = plt.plot(T_vals, results[k]['5.0'], label='c=5')
        c10, = plt.plot(T_vals, results[k]['10.0'], label='c=10')
        plt.xticks(T_vals)
        plt.legend(handles=[c1, c2, c5, c10])
        plt.xlabel('T (# of rounds)', fontsize=18)
        plt.ylabel(v['ylabel'], fontsize=15)
        plt.savefig('figures/' + v['name'])

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
