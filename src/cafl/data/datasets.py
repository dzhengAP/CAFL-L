“”“Data loading and preprocessing utilities.”””

import requests
import torch

# Shakespeare dataset URL

SHAKESPEARE_URL = “https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt”

def download_text(url=SHAKESPEARE_URL, timeout=10):
“””
Download text data from URL.

```
Args:
    url: URL to download from
    timeout: Request timeout in seconds

Returns:
    Downloaded text string
"""
try:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text
except Exception as e:
    print(f"Warning: Failed to download from {url}: {e}")
    print("Using fallback text sample...")
    return (
        "From fairest creatures we desire increase,\n"
        "That thereby beauty's rose might never die,\n"
        "But as the riper should by time decease,\n"
        "His tender heir might bear his memory:\n"
    )
```

def normalize_text(text):
“””
Normalize text to printable ASCII characters.

```
Args:
    text: Input text string

Returns:
    Normalized text with only printable ASCII
"""
return "".join([ch if 32 <= ord(ch) <= 126 else " " for ch in text])
```

def build_charset(text):
“””
Build character vocabulary from text.

```
Args:
    text: Input text string

Returns:
    chars: Sorted list of unique characters
    stoi: Character to index mapping
    itos: Index to character mapping
"""
chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
return chars, stoi, itos
```

def encode_text(text, stoi):
“””
Encode text to token indices.

```
Args:
    text: Input text string
    stoi: Character to index mapping

Returns:
    Tensor of token indices
"""
return torch.tensor([stoi[c] for c in text], dtype=torch.long)
```

def split_clients(text, num_clients, overlap=0):
“””
Split text into client shards with optional overlap.

```
Args:
    text: Full text string
    num_clients: Number of clients
    overlap: Overlap between shards in characters

Returns:
    List of text shards for each client
"""
N = len(text)
segment_size = N // num_clients
shards = []

for i in range(num_clients):
    start = max(0, i * segment_size - overlap)
    end = min(N, (i + 1) * segment_size + overlap)
    shards.append(text[start:end])

return shards
```

def prepare_shakespeare_data(num_clients=16, overlap=48, train_split=0.9, device=“cpu”):
“””
Download and prepare Shakespeare dataset for federated learning.

```
Args:
    num_clients: Number of clients to split data for
    overlap: Overlap between client shards
    train_split: Fraction of data for training
    device: Device to place tensors on

Returns:
    clients_ids: List of encoded client data tensors
    valid_ids: Encoded validation data tensor
    vocab_info: Dictionary with vocabulary information
"""
# Download and normalize
raw_text = normalize_text(download_text())

# Build vocabulary
chars, stoi, itos = build_charset(raw_text)

# Split train/validation
split_idx = int(train_split * len(raw_text))
train_text = raw_text[:split_idx]
valid_text = raw_text[split_idx:]

# Split training data into client shards
client_texts = split_clients(train_text, num_clients, overlap)
clients_ids = [encode_text(text, stoi).to(device) for text in client_texts]

# Encode validation data
valid_ids = encode_text(valid_text, stoi).to(device)

vocab_info = {
    "chars": chars,
    "stoi": stoi,
    "itos": itos,
    "vocab_size": len(chars)
}

print(f"Dataset prepared:")
print(f"  - Vocabulary size: {len(chars)}")
print(f"  - Training chars: {len(train_text):,}")
print(f"  - Validation chars: {len(valid_text):,}")
print(f"  - Clients: {num_clients}")
print(f"  - Avg chars per client: {len(train_text) // num_clients:,}")

return clients_ids, valid_ids, vocab_info
```

def batchify(token_ids, block_size, batch_size, rng):
“””
Create random batches from token sequence.

```
Args:
    token_ids: Token tensor [N]
    block_size: Sequence length
    batch_size: Batch size
    rng: PyTorch random generator

Returns:
    x: Input sequences [B, T]
    y: Target sequences [B, T]
"""
seq_len = len(token_ids) - block_size - 1

if seq_len <= 0:
    # Handle case where data is too short
    x = token_ids[:block_size].unsqueeze(0).repeat(batch_size, 1)
    y = token_ids[1:block_size + 1].unsqueeze(0).repeat(batch_size, 1)
    return x, y

# Sample random starting positions
indices = torch.randint(0, seq_len, (batch_size,), generator=rng, device=token_ids.device)

# Create input and target sequences
x = torch.stack([token_ids[i:i + block_size] for i in indices.tolist()], dim=0)
y = torch.stack([token_ids[i + 1:i + 1 + block_size] for i in indices.tolist()], dim=0)

return x, y
```