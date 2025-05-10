import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class DeepObliviousDecisionTreeLayer(nn.Module):
    def __init__(self, input_dim, num_trees, tree_depth, hidden_dim):
        super(DeepObliviousDecisionTreeLayer, self).__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.hidden_dim = hidden_dim
        
        # Define a deeper architecture with hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(tree_depth):
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim 
        
        self.output_layer = nn.Linear(hidden_dim, num_trees)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x)) 
        
        # Apply the output layer
        out = self.output_layer(x)
        return out

class NODEClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_trees, tree_depth, hidden_dim, batch_size, num_epochs, learning_rate):
        super(NODEClassifier, self).__init__()
        self.tree_layer = DeepObliviousDecisionTreeLayer(input_dim, num_trees=num_trees, tree_depth=tree_depth, hidden_dim=hidden_dim)
        
        # Additional fully connected layers to make it deeper
        self.fc1 = nn.Linear(num_trees, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.tree_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def fit(self, X_train, y_train):
        # Prepare dataset and dataloader
        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            # Calculate and display accuracy every 10 epochs
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.num_epochs:
                total_correct = 0
                total_samples = 0

                # Evaluate on training set to compute accuracy
                with torch.no_grad():  # Disable gradient calculation
                    for inputs, labels in train_loader:
                        outputs = self(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total_correct += (predicted == labels).sum().item()
                        total_samples += labels.size(0)

                accuracy = total_correct / total_samples

                # Print loss and accuracy
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}')

        print("Training process has finished")
    
    def predict(self, X_test):
        # Prepare dataloader
        test_loader = DataLoader(TensorDataset(X_test), batch_size=self.batch_size, shuffle=False)
        
        # Set the model to evaluation mode
        self.eval()
        
        y_pred = []

        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs in test_loader:
                inputs = inputs[0]  # Extract the input from the tuple
                outputs = self(inputs)  # Outputs are raw logits
                
                # Get predicted class by finding the max logit (since we are not using sigmoid)
                _, predicted = torch.max(outputs, 1)
                
                # Store predicted labels
                y_pred.extend(predicted.cpu().numpy())

        return y_pred
