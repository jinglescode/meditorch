import matplotlib.pyplot as plt


def metrics_line(metrics):
    keys = metrics[0].keys()

    i = 0
    fig, axs = plt.subplots(1, len(keys))
    fig.set_figheight(5)
    fig.set_figwidth(15)
    for key in keys:
        axs[i].plot([i[key] for i in metrics], label=key)
        axs[i].set_title(key)
        i+=1

    plt.show()