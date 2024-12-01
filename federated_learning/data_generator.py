import pandas as pd
from faker import Faker
import numpy as np
import pickle

class DataGenerator:
    def __init__(self, num_samples=1000):
        self.faker = Faker()
        self.num_samples = num_samples
    
    def generate_credit_data(self):
        """
        Generate synthetic credit scoring data
        Features:
        - Income
        - Age
        - Employment duration
        - Credit history length
        - Number of credit cards
        - Total debt
        - Payment history
        """
        data = {
            'income': np.random.normal(50000, 15000, self.num_samples),
            'age': np.random.randint(22, 65, self.num_samples),
            'employment_duration': np.random.randint(0, 20, self.num_samples),
            'credit_history_length': np.random.randint(1, 15, self.num_samples),
            'num_credit_cards': np.random.randint(0, 5, self.num_samples),
            'total_debt': np.random.normal(20000, 10000, self.num_samples),
            'payment_history_score': np.random.randint(300, 850, self.num_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target variable (credit score)
        df['credit_score'] = (
            df['income'] * 0.3 / 1000 +
            df['age'] * 0.1 +
            df['employment_duration'] * 0.2 +
            df['credit_history_length'] * 0.15 -
            df['total_debt'] * 0.05 / 1000 +
            df['payment_history_score'] * 0.2 / 850
        ) * 100
        
        df['credit_score'] = df['credit_score'].clip(300, 850)
        
        return df
    
    def save_data(self, df, filename):
        """Save generated data to a pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
    
    def load_data(self, filename):
        """Load data from a pickle file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)