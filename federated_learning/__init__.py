from .data_generator import DataGenerator
from .local_model import LocalModel
from .aggregator import ModelAggregator
from .main import run_federated_learning
import numpy as np

def run_federated_learning(num_institutions=5):
    # Generate data for multiple institutions
    data_generator = DataGenerator()
    
    # Simulate multiple institution models
    local_models = []
    
    for _ in range(num_institutions):
        # Generate institution-specific dataset
        institution_data = data_generator.generate_credit_data()
        
        # Train local model
        local_model = LocalModel(institution_data)
        local_model_result = local_model.train()
        
        local_models.append(local_model_result)
    
    # Aggregate models
    aggregator = ModelAggregator(local_models)
    global_model = aggregator.federated_averaging()
    
    return global_model

if __name__ == '__main__':
    global_model = run_federated_learning()
    print("Global Model Weights:", global_model['global_weights'])
    print("Global Model Performance:", global_model['global_performance'])