# mask作图

import matplotlib.pyplot as plt
import numpy as np

# 可选配色
cmaps = [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]


for layer_id in range(1,14):

    model_name = 'Vgg16'
    msg = np.load('./mask_numpy/'+ model_name +'/layer' + str(layer_id) + '.npy')
    print(msg)
    print(msg.shape)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # for ax in ax_list:

    ax.pcolor(msg, linewidths=0, cmap='Set3')

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_yticks(np.array([-0.01, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]))
    ax.set_yticks(np.arange(msg.shape[0] + 1), minor=True)
    ax.set_yticklabels(np.array([0, 1, 2, 5, 10, 20, 40, 80, 120]))
    ax.grid(which="minor", color="w", linestyle='-', linewidth=14)

    ax.set_title('layer' + str(layer_id))

    fig.tight_layout()
    plt.show()
