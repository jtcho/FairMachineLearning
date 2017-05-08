import matplotlib.pyplot as plt
import numpy as np

from fairml import beta, compute_chain, eta, interval_chaining, top_interval


def main():
    c_vals = [1.0, 2.0, 5.0, 10.0]

    # Plot: Varying T (# of rounds)
    d = 2
    k = 2
    T_vals = range(3, 1000, 10)

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
            cum_regret_tis = []
            avg_regret_tis = []
            final_regret_tis = []
            cum_regret_ics = []
            avg_regret_ics = []
            final_regret_ics = []
            for i in range(0, 50):  # 50 trials.
                X = np.random.uniform(0, 1, size=(T, k, d))
                B = beta(k, d, c)
                Y = np.array([np.diag(X[t].dot(np.transpose(B))) for t in range(T)])

                cum_regret_ti, avg_regret_ti, final_regret_ti = top_interval(
                        X, Y, k, d, 0.05, T, _print_progress=False)
                cum_regret_ic, avg_regret_ic, final_regret_ic = interval_chaining(
                        X, Y, c, k, d, 0.05, T, _print_progress=False)
                cum_regret_tis.append(cum_regret_ti)
                avg_regret_tis.append(avg_regret_ti)
                final_regret_tis.append(final_regret_ti)
                cum_regret_ics.append(cum_regret_ic)
                avg_regret_ics.append(avg_regret_ic)
                final_regret_ics.append(final_regret_ic)
            cum_regret_ti = mean(cum_regret_tis)
            avg_regret_ti = mean(avg_regret_tis)
            final_regret_ti = mean(avg_regret_tis)
            cum_regret_ic = mean(cum_regret_ics)
            avg_regret_ic = mean(avg_regret_ics)
            final_regret_ics = mean(final_regret_ics)

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
        plt.savefig('figures_T_50x/' + v['name'])


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


if __name__ == '__main__':
    main()
