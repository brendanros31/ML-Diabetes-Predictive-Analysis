import matplotlib.pyplot as plt
import seaborn as sns


# Heatmap
def heatmap(df):
    plt.figure(figsize=(12, 8))

    sns.heatmap(df.corr(), annot=True)
    plt.title('Correlation Heatmap')

    plt.show()