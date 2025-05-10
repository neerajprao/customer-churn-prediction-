import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is expected to have shape [batch_size, sequence_length, embed_size]
        attention = self.attention(x, x, x)[0]
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TabTrClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, num_heads, forward_expansion, dropout, batch_size, num_epochs, learning_rate):
        super(TabTrClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_size)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size,
            heads=num_heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )
        self.fc = nn.Linear(embed_size, output_dim)

        # Training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def forward(self, x):
        # Add sequence dimension
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_block(x)
        x = x.squeeze(1)
        out = self.fc(x)
        return out
    
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
