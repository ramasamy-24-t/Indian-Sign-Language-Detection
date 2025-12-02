import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os

class CustomActionModelTrainer:
    def __init__(self):
        self.user_actions = []
        self.model = None
        self.scaler = StandardScaler()
        self.load_user_actions()
        
    def load_user_actions(self):
        """Load user-defined actions from saved data"""
        try:
            with open('custom_label_data/user_actions.pkl', 'rb') as f:
                self.user_actions = pickle.load(f)
            print(f"✓ Loaded {len(self.user_actions)} user-defined actions:")
            for i, action in enumerate(self.user_actions, 1):
                print(f"  {i}. {action}")
        except FileNotFoundError:
            print("❌ No user actions found! Please run data collection first.")
            return False
        return True
        
    def load_custom_data(self):
        """Load custom labeled data for training"""
        X, y = [], []
        
        print("\nLoading training data...")
        for action_idx, action in enumerate(self.user_actions):
            try:
                action_data = np.load(f'custom_label_data/{action}.npy')
                for frame_data in action_data:
                    X.append(frame_data)
                    y.append(action_idx)
                print(f"✓ Loaded {len(action_data)} frames for '{action}'")
            except FileNotFoundError:
                print(f"⚠ Warning: No data found for '{action}'")
                continue
        
        if len(X) == 0:
            print("❌ No training data loaded!")
            return None, None
            
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n✓ Total training data: {X.shape} samples, {len(set(y))} classes")
        return X, y
    
    def create_and_train_model(self):
        """Create and train model for user-defined actions"""
        if not self.user_actions:
            print("❌ No user actions available for training!")
            return None, 0
            
        # Load custom data [3][6]
        X, y = self.load_custom_data()
        
        if X is None:
            return None, 0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape} samples")
        print(f"Testing set: {X_test.shape} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining RandomForest model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model [9][15]
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"\n✓ Training Accuracy: {train_accuracy:.4f}")
        print(f"✓ Testing Accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_test, test_predictions, 
              target_names=self.user_actions, zero_division=0))
        
        # Save model, scaler, and actions
        self.save_model()
        
        return self.model, test_accuracy
    
    def save_model(self):
        """Save trained model and related files"""
        print("\n💾 Saving model files...")
        
        with open('custom_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("✓ Model saved as 'custom_model.pkl'")
        
        with open('custom_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✓ Scaler saved as 'custom_scaler.pkl'")
        
        with open('custom_actions.pkl', 'wb') as f:
            pickle.dump(self.user_actions, f)
        print("✓ Actions saved as 'custom_actions.pkl'")
        
        # Update metadata with model info
        try:
            with open('custom_label_data/metadata.json', 'r') as f:
                metadata = json.load(f)
            
            metadata['model_trained'] = True
            metadata['model_file'] = 'custom_model.pkl'
            metadata['scaler_file'] = 'custom_scaler.pkl'
            metadata['actions_file'] = 'custom_actions.pkl'
            
            with open('custom_label_data/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            print("✓ Metadata updated")
            
        except:
            print("⚠ Could not update metadata")

if __name__ == "__main__":
    print("🚀 CUSTOM ACTION MODEL TRAINER")
    print("="*40)
    
    trainer = CustomActionModelTrainer()
    if trainer.user_actions:
        model, accuracy = trainer.create_and_train_model()
        
        if model and accuracy > 0:
            print(f"\n🎉 Model training completed successfully!")
            print(f"🎯 Final accuracy: {accuracy:.2%}")
        else:
            print("\n❌ Model training failed!")
    else:
        print("❌ No user actions found. Please run data collection first.")
