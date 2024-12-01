from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class LocalModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
    
    def preprocess_data(self):
        """Prepare data for training"""
        X = self.data.drop(['credit_score'], axis=1)
        y = self.data['credit_score']
        
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def train(self):
        """Train local model"""
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        self.model.fit(X_train, y_train)
        
        # Calculate local model performance
        local_score = self.model.score(X_test, y_test)
        
        return {
            'model_weights': self.model.feature_importances_,
            'local_score': local_score
        }