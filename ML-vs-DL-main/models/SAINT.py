import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attn_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        
        self.query = nn.Linear(input_dim, attn_dim * num_heads)
        self.key = nn.Linear(input_dim, attn_dim * num_heads)
        self.value = nn.Linear(input_dim, attn_dim * num_heads)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(attn_dim * num_heads, input_dim)
        
    def forward(self, x):
        if len(x.size()) == 2:  # If input has 2 dimensions
            x = x.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, input_dim]

        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_layer(attn_output)

        return output

class IntersampleAttention(nn.Module):
    def __init__(self, input_dim, attn_dim, num_heads, dropout=0.1):
        super(IntersampleAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        
        self.query = nn.Linear(input_dim, attn_dim * num_heads)
        self.key = nn.Linear(input_dim, attn_dim * num_heads)
        self.value = nn.Linear(input_dim, attn_dim * num_heads)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(attn_dim * num_heads, input_dim)
        
    def forward(self, x):
        if len(x.size()) == 2:  # If input has 2 dimensions
            x = x.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, input_dim]

        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_layer(attn_output)

        return output

class SAINTClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, attn_dim, num_heads, num_layers, hidden_dim, dropout, size_of_batch, num_epochs, learning_rate):
        super(SAINTClassifier, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.self_attn_layers = nn.ModuleList([
            SelfAttention(hidden_dim, attn_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.intersample_attn_layers = nn.ModuleList([
            IntersampleAttention(hidden_dim, attn_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.size_of_batch = size_of_batch
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        for self_attn, intersample_attn in zip(self.self_attn_layers, self.intersample_attn_layers):
            x = self_attn(x)
            x = intersample_attn(x)
        
        x = torch.mean(x, dim=1)  # Global average pooling
        
        output = self.output_layer(x)
        
        return output
    
    def fit(self, X_train, y_train):
        # Prepare dataset and dataloader
        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=self.size_of_batch, shuffle=True)
        
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
        test_loader = DataLoader(TensorDataset(X_test), batch_size=self.size_of_batch, shuffle=False)
        
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
