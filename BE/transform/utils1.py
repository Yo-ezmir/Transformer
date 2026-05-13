import math
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('/home/yon/Transformer/data/train.csv', 'r', encoding='utf-8') as f:
    text = f.read()


with open('/home/yon/Transformer/data/train.csv', 'r', encoding='utf-8') as f:
    text += f.read()

with open('/home/yon/Transformer/data/test.csv', 'r', encoding='utf-8') as f:
    text += f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]


batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def mask(tensor, T, C):
  matrix = torch.zeros(1, T, C)
  mask = torch.triu(torch.ones_like(matrix[0, :, :]), diagonal=1)
  matrix.masked_fill_(mask.bool(), float('-inf'))
  return tensor + matrix

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        attention_score = q @ k.transpose(-2,-1) * C**-0.5
        attention_score = attention_score.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)

        v = self.value(x)
        attention = attention_score @ v
        return attention

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

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
        self.linearLayer1 = nn.LayerNorm(n_embd)
        self.linearLayer2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mHA(self.linearLayer1(x))
        x = x + self.ffwd(self.linearLayer2(x))
        return x

class DecodOnlyTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.layernorm_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, token, targets=None):
        B, T = token.shape
        token_emb = self.token_embedding(token)
        position_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + position_emb 

        x = self.blocks(x)
        x = self.layernorm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, token, max_new_tokens):
        for _ in range(max_new_tokens):
            token_cond = token[:, -block_size:]
            logits, loss = self(token_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            token_next = torch.multinomial(probs, num_samples=1)
            token = torch.cat((token, token_next), dim=1)
        return token

model = DecodOnlyTransformer()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

PATH = '/home/yon/Transformer/model/decoder_model3.pth'



def evaluate_model():
    checkpoint = torch.load(PATH, map_location=device)
    model = DecodOnlyTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_loss = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for _ in range(100):
            x, y = get_batch('val')
            logits, loss = model(x, y)

            if loss is not None:
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()
                num_batches += 1

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')

    print(f"Evaluation completed:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'note': f'Evaluated on {num_batches} batches, {total_tokens} tokens'
    }


def generate_text(prompt="The", max_new_tokens=200):

    PATH = '/home/yon/Transformer/model/decoder_model3.pth'
    checkpoint = torch.load(PATH, map_location=device)
    model = DecodOnlyTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    m = model.to(device)
    m.eval()
    print("Model loaded successfully.")

    print(f"Starting prompt: '{prompt}'")
    context_tokens = encode(prompt)
    context = torch.tensor([context_tokens], dtype=torch.long, device=device)

    print("Generating text...")
    generated_indices = m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    output_text = decode(generated_indices)

    return output_text