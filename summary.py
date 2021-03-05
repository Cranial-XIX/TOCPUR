import numpy as np

methods = ["greedy", "top", "tocp"]

results = {}
time = {}
failures = {}

folder_name = "results"
for m in methods:
    results[m] = {}
    time[m] = {}
    failures[m] = {}
    for H in [2,4,6,8,10]:
        results[m][H] = [-1] * 120
        time[m][H] = [-1] * 120
        failures[m][H] = [-1] * 120
        for seed in range(1, 121):
            try:
                stat = np.load(f"{folder_name}/{m}-H{H}-s{seed}.npy", allow_pickle=True).item()
                cost = stat['cost'].mean()
                t    = stat['time']
                f    = (stat['fail'].mean() > 0)
                if np.isnan(cost):
                    continue
                results[m][H][seed-1] = cost
                time[m][H][seed-1] = t
                failures[m][H][seed-1] = f
            except:
                continue

masks = {}
for H in [2,4,6,8,10]:
    masks[H] = np.ones((120,))
    for m in methods:
        x = np.array(results[m][H])
        masks[H] = masks[H] * (x >= 0)

p_results = {}
for H in [2,4,6,8,10]:
    p_results[H] = {}
    for m in methods:
        p_results[H][m] = {}
        x = np.array(results[m][H])
        t = np.array(time[m][H])
        f = np.array(failures[m][H])
        n_failures = (f * (x > 0)).sum()
        p_results[H][m]['cost_mean'] = x[masks[H] > 0].mean()
        p_results[H][m]['cost_std'] = x[masks[H] > 0].std()
        p_results[H][m]['t_mean'] = t[masks[H] > 0].mean()
        p_results[H][m]['t_std'] = t[masks[H] > 0].std()
        p_results[H][m]['n_valid'] = (masks[H] > 0).sum()
        p_results[H][m]['n_unfinished'] = (x < 0).sum()
        p_results[H][m]['n_failures'] = n_failures

with open(f"summary.npy", "wb") as f:
    np.save(f, p_results)
f.close()
