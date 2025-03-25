import torch
import torch.nn as nn
import torch.optim as optim
import math

def tokenize(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Removed unnecessary transpose here
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Changed indexing to match x's seq length
        return x + self.pe[:, :x.size(1), :]

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)
        return attended_values

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)
        forwarded = self.feed_forward(x)
        x = self.norm2(x + forwarded)
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(SimpleLLM, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_dim, hidden_dim) for _ in range(num_layers)])
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        #No need for transposes here.  Pos encoding and attention handle sequence dimension as second dimension.
        x = self.positional_encoding(x)
        x = self.transformer_blocks(x)
        x = self.output(x)
        return x

vocab = {"hello": 0, "world": 1, "how": 2, "are": 3, "you": 4, "<UNK>": 5}
vocab_size = len(vocab)
embedding_dim = 16
hidden_dim = 32
num_layers = 2

model = SimpleLLM(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data = ["hello world how are you", "how are you hello world"]
tokenized_data = [tokenize(sentence, vocab) for sentence in data]

for epoch in range(100):
    epoch_loss = 0
    for sentence in tokenized_data:
        for i in range(1, len(sentence)):
            input_seq = torch.tensor(sentence[:i]).unsqueeze(0)
            target = torch.tensor([sentence[i]]) #Changed to array for crossentropy loss
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output[:, -1, :], target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Avg Loss: {epoch_loss/len(data)}")

input_text = "hello world how"
input_tokens = tokenize(input_text, vocab)
input_tensor = torch.tensor(input_tokens).unsqueeze(0)
output = model(input_tensor)
predicted_token = torch.argmax(output[:, -1, :]).item()
print(f"Input: {input_text}, Predicted: {list(vocab.keys())[list(vocab.values()).index(predicted_token)]}")
