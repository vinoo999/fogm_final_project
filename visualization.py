import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold

 
def matrix_heatmap(X, title='Heatmap', colorbar_label='colorbar'):
    ''' 
    Code adapted from stackoverflow id=9662995
    X is an expression matrix N cells x M genes
    Heat map has cell on X axis and genes on Y axis

    '''
    fig, ax1 = plt.subplots(1,1)

    im = ax1.imshow(X, cmap='hot', interpolation='nearest', extent=[0,X.shape[1], 0, X.shape[0]], aspect='auto')
    cb = plt.colorbar(im)

    fg_color = 'black'
    bg_color = 'white'

    # IMSHOW    
    # set title plus title color
    ax1.set_title(title, color=fg_color)

    # set figure facecolor
    ax1.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    for spine in im.axes.spines.values():
        spine.set_edgecolor(fg_color)    

    # COLORBAR
    # set colorbar label plus label color
    cb.set_label(colorbar_label, color=fg_color)

    # set colorbar tick color
    cb.ax.yaxis.set_tick_params(color=fg_color)

    # set colorbar edgecolor 
    cb.outline.set_edgecolor(fg_color)
    
    # set colorbar ticklabels
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)
    fig.patch.set_facecolor(bg_color)    
    plt.tight_layout()
    plt.show()
    return


def display_embeddings(embeddings, labels=None, cluster_labels=None, vis='tsne'):
    '''
    Embeddings are an NxD numpy matrix of N cells with D features
    Labels are a Nx1 vector each taking an integer value from 1 to K inclusive
    vis is the type of visualization to draw from (tsne or pca)
    cluster is whether or not to perform clustering on the embeddings
    K is the number of clusters to perform. k-means is currenlty the only function supported.
    '''


    if vis == 'tsne':
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Y = tsne.fit_transform(embeddings)
    elif vis == 'pca':
        pca = PCA(n_components=2)
        Y = pca.fit_transform(embeddings)
    else:
        print("Bad visualization type")
        return

    markers = ['.', 'o', 'v','s','*','x','d','+','8','<']
    colors = ['red', 'c','gold','olivedrab','mediumblue','peru','hotpink','magenta','green','lawngreen']

    if cluster_labels is not None:
        plot_marker = [markers[i] for i in cluster_labels]
    else:
        plot_marker = [markers[0]] * embeddings.shape[0]

    if labels is not None:
        plot_color = [colors[i-1] for i in labels]
    else:
        plot_color = [colors[0]] * embeddings.shape[0]
        
    
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(1,1,1) 
    for i in range(embeddings.shape[0]):
        ax.scatter(Y[i,0], Y[i,1], c=plot_color[i], marker=plot_marker[i])

    plt.show()
    return

