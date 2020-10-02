import matplotlib.pyplot as plt
import numpy as np

class Visualize():
    def __init__(self, costs):
        self.costs = costs
    
    def cost_per_epoch(self):
        plt.plot(np.squeeze(self.costs))
        plt.title("Cost per Epoch")
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.show()