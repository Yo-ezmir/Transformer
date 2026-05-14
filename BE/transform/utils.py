import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
vocab_size = tokenizer.vocab_size

# Hyperparameters (match training)
batch_size = 16
block_size = 128
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.1

PATH = "/home/yon/Transformer/model/decoder_model_bert.pth"

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        attention_score = q @ k.transpose(-2, -1) * C**-0.5
        attention_score = attention_score.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        v = self.value(x)
        return attention_score @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.mHA = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mHA(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecodOnlyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, token, targets=None):
        B, T = token.shape
        token_emb = self.token_embedding(token)
        position_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, token, max_new_tokens):
        for _ in range(max_new_tokens):
            token_cond = token[:, -block_size:]
            logits, _ = self(token_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            token_next = torch.multinomial(probs, num_samples=1)
            token = torch.cat((token, token_next), dim=1)
        return token

def load_model():
    checkpoint = torch.load(PATH, map_location=device, weights_only=False)
    model = DecodOnlyTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def generate_text(prompt="The", max_new_tokens=200):
    model = load_model()
    context_tokens = tokenizer.encode(prompt)
    context = torch.tensor([context_tokens], dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens)[0].tolist()
    text = tokenizer.decode(generated_indices)
    # Clean BERT special tokens
    text = text.replace("[CLS]", "").replace("[SEP]", "").strip()
    return text

def evaluate_model():
    checkpoint = torch.load(PATH, map_location=device, weights_only=False)
    model = DecodOnlyTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    with open('/home/yon/Transformer/data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    val_data = data[n:]
    
    def get_val_batch():
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x = torch.stack([val_data[i:i+block_size] for i in ix])
        y = torch.stack([val_data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(100):
            x, y = get_val_batch()
            logits, loss = model(x, y)
            if loss is not None:
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return {'perplexity': perplexity, 'loss': avg_loss}