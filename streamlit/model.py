# Define Model
from torch import nn
from torch import optim

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim // 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])  # take last time step
        out = self.fc(x)
        return out

# Instantiate model
model = NextWordLSTM(total_words, embed_dim=100, hidden_dim=150)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)