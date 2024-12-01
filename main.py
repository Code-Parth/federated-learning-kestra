import logging
import os
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from federated_learning.data_generator import DataGenerator
from federated_learning.local_model import LocalModel
from federated_learning.aggregator import ModelAggregator

# Configure logging
def setup_logging() -> logging.Logger:
    """Set up comprehensive logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate unique log filename with timestamp
    log_filename = f'logs/federated_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__)

def visualize_model_performance(local_models: List[Dict], global_model: Dict):
    """Create visualizations of model performance."""
    # Create output directory for visualizations
    os.makedirs('outputs', exist_ok=True)
    
    # Local model performance visualization
    local_scores = [model['local_score'] for model in local_models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(local_scores)), local_scores, color='blue', alpha=0.6)
    plt.axhline(y=global_model['global_performance'], color='r', linestyle='--', label='Global Performance')
    plt.title('Local Model Performance Comparison')
    plt.xlabel('Local Model Index')
    plt.ylabel('R² Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/local_model_performance.png')
    plt.close()

def save_model_results(local_models: List[Dict], global_model: Dict, logger: logging.Logger):
    """Save model results to JSON for further analysis."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'local_models': [
            {
                'local_score': float(model['local_score']),
                'model_weights': model['model_weights'].tolist()
            } for model in local_models
        ],
        'global_model': {
            'global_performance': float(global_model['global_performance']),
            'global_weights': global_model['global_weights'].tolist()
        }
    }
    
    # Save results to a JSON file
    os.makedirs('results', exist_ok=True)
    results_filename = f'results/federated_learning_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Model results saved to {results_filename}")

def run_federated_learning(
    num_institutions: int = 5, 
    num_samples: int = 1000, 
    random_seed: int = 42
) -> Dict:
    """
    Execute federated learning workflow with comprehensive logging and error handling.
    
    Args:
        num_institutions (int): Number of simulated institutions.
        num_samples (int): Number of samples per institution.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        Dict: Global model results
    """
    # Setup logging
    logger = setup_logging()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        logger.info(f"Starting Federated Learning with {num_institutions} institutions")
        
        # Generate data for multiple institutions
        data_generator = DataGenerator(num_samples=num_samples)
        
        # Simulate multiple institution models
        local_models = []
        
        for i in range(num_institutions):
            logger.info(f"Training local model for Institution {i+1}")
            
            # Generate institution-specific dataset
            institution_data = data_generator.generate_credit_data()
            
            # Train local model
            local_model = LocalModel(institution_data)
            local_model_result = local_model.train()
            
            logger.info(f"Institution {i+1} local model performance: {local_model_result['local_score']:.4f}")
            local_models.append(local_model_result)
        
        # Aggregate models
        aggregator = ModelAggregator(local_models)
        global_model = aggregator.federated_averaging()
        
        logger.info(f"Global Model Performance: {global_model['global_performance']:.4f}")
        
        # Visualize and save results
        visualize_model_performance(local_models, global_model)
        save_model_results(local_models, global_model, logger)
        
        return global_model
    
    except Exception as e:
        logger.error(f"An error occurred during federated learning: {e}", exc_info=True)
        raise

def main():
    """Main entry point for the federated learning workflow."""
    try:
        # Run federated learning
        global_model = run_federated_learning(
            num_institutions=50,   # Configurable number of institutions
            num_samples=5000,     # Configurable number of samples
            random_seed=42        # Reproducibility seed
        )
        
        # Print global model details
        print("\n--- Global Model Summary ---")
        print(f"Global Model Performance: {global_model['global_performance']:.4f}")
        print("Global Model Weights:")
        for i, weight in enumerate(global_model['global_weights']):
            print(f"Feature {i}: {weight:.4f}")
        
    except Exception as e:
        print(f"Federated Learning Failed: {e}")

if __name__ == '__main__':
    main()