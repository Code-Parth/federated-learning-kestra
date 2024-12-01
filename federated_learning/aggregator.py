import numpy as np

class ModelAggregator:
    def __init__(self, local_models):
        self.local_models = local_models
    
    def federated_averaging(self):
        """
        Aggregate model weights from multiple local models
        using simple averaging
        """
        # Collect weights from local models
        all_weights = [model['model_weights'] for model in self.local_models]
        
        # Calculate average weights
        global_weights = np.mean(all_weights, axis=0)
        
        # Calculate average performance
        avg_performance = np.mean([model['local_score'] for model in self.local_models])
        
        return {
            'global_weights': global_weights,
            'global_performance': avg_performance
        }