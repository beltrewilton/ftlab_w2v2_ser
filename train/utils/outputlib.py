import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def WriteConfusionSeaborn(m, labels, outpath):
    '''
    INPUT:
        m: confusion matrix (numpy array)
        labels: List of string, the category name of each entry in m
        name: Name for the output png plot
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    inn = m / m.sum(1, keepdims=True)
    ax = sns.heatmap(inn, cmap='Blues', fmt='.2%', xticklabels=labels, yticklabels=labels,
                     annot=True, annot_kws={"size": 12}, ax=ax)
    for t in ax.texts:
        t.set_text(t.get_text()[:-1])

    fig.savefig(outpath)
    print(m)
    print(f"Saved figure to {outpath}.")
    plt.close(fig)

#
# m = np.array(
# [[113.,1.,7.,1.,1.,0.,0.,6.]
# ,[0.,85.,0.,0.,0.,15.,1.,0.]
# ,[2.,0.,90.,0.,1.,0.,3.,0.]
# ,[2.,2.,0.,85.,2.,1.,16.,3.]
# ,[2.,0.,0.,2.,90.,5.,3.,7.]
# ,[0.,4.,0.,0.,1.,51.,3.,0.]
# ,[1.,19.,2.,2.,7.,9.,82.,1.]
# ,[0.,0.,2.,4.,6.,0.,0.,124.]]
# )
#
# WriteConfusionSeaborn(m, ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'], "porqueria.png")
