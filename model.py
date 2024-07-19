import torch
import torch.nn as nn
from einops import rearrange
from transformers import BertTokenizer, BertModel


# LoRA for fine tuning
class LoRA(nn.Module):
    def __init__(self, original_layer, rank):
        super(LoRA, self).__init__()
        self.original_layer = original_layer
        self.rank = rank

        # Create the  matrices A and B
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.reset_parameters()

        # freeze the original paraments
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        # Initialize the weights of lora matrices to zeros
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_layer(x) + self.lora_B(self.lora_A(x))





class MHA(nn.Module):
    def __init__(self, d_model, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        
        # Define query, key, value, and output transformations
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        # Define dropout layer for regularization
        self.dropout = nn.Dropout(drop_p)
        
        # Calculate the scaling factor for the attention scores
        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

    def forward(self, x, mask=None):
        # Compute query (Q), key (K), and value (V) projections
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q = rearrange(Q, 'b t (h d) -> b h t d', h=self.n_heads)
        K = rearrange(K, 'b t (h d) -> b h t d', h=self.n_heads)
        V = rearrange(V, 'b t (h d) -> b h t d', h=self.n_heads)

        # Compute attention scores 
        attention_score = Q @ K.transpose(-2, -1) / self.scale

        # If exists, apply mask to the attention scores
        if mask is not None:
            attention_score[mask] = -1e10
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_score, dim=-1)
        # Apply dropout 
        attention_weights = self.dropout(attention_weights)
        # Compute the attention output by multiplying attention weights with V
        attention = attention_weights @ V

        # Rearrange the attention output to the original input shape
        x = rearrange(attention, 'b h t d -> b t (h d)')
        # Apply the output linear layer
        x = self.fc_o(x)

        return x, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        # Crrate a linear layer for feed_forward 
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        # Layer norm for self attention
        self.self_atten_LN = nn.LayerNorm(d_model)
        # Multi head self attention
        self.self_atten = MHA(d_model, n_heads, drop_p)

        # Layer norm for feed-forward 
        self.FF_LN = nn.LayerNorm(d_model)
        # Feed-forward network
        self.FF = FeedForward(d_model, d_ff, drop_p)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_mask):
        # Applly layer norm
        residual = self.self_atten_LN(x)
        # Apply self attention and dropout
        residual, atten_enc = self.self_atten(residual, enc_mask)
        residual = self.dropout(residual)
        # Add the attention output to the original input for residual connection
        x = x + residual

        # Apply layer norm
        residual = self.FF_LN(x)
        # Apply feedforward and droput
        residual = self.FF(residual)
        residual = self.dropout(residual)
        # Residual connection
        x = x + residual

        return x, atten_enc


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Positional embedding layer
        self.pos_embedding = nn.Embedding(max_len, d_model)
        # Segment embedding layer 
        self.seg_embedding = nn.Embedding(2, d_model)

        self.dropout = nn.Dropout(drop_p)
        # List of Encoderlayers
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])
        # Layer norm for output
        self.LN_out = nn.LayerNorm(d_model)

    def forward(self, x, seg, enc_mask, atten_map_save=False):
        # Create positional indices
        pos = torch.arange(x.shape[1]).expand_as(x).to(x.device)
        # Sum the token, positional, and segment embeddings
        x = self.token_embedding(x) + self.pos_embedding(pos) + self.seg_embedding(seg)
        # Apply dropout
        x = self.dropout(x)

        # Initialize to save attention maps
        atten_encs = torch.tensor([]).to(x.device)
        for layer in self.layers:
            # Pass through each layer
            x, atten_enc = layer(x, enc_mask)
            # Save attention maps if asked to
            if atten_map_save:
                atten_encs = torch.cat([atten_encs, atten_enc[0].unsqueeze(0)], dim=0)

        # Apply layer norm for output
        x = self.LN_out(x)
        return x, atten_encs




class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        
        # Intilialize encoder
        self.encoder = Encoder(config.vocab_size, config.max_len, config.n_layers, config.d_model, config.d_ff, config.n_heads, config.drop_p)

        # Store number of heads and pad index
        self.n_heads = config.n_heads
        self.pad_idx = config.pad_idx

        # Initialize weights for linear and embedding layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                m.weight.data *= 1 / torch.sqrt(torch.tensor(config.n_layers * 2))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def load_pretrained_weights(self, pretrained_model_name):
        # Load a pretrained BERT model from Hugging Face
        pretrained_model = BertModel.from_pretrained(pretrained_model_name)
        
        # Transfer weights to the custom model
        self.encoder.token_embedding.weight.data = pretrained_model.embeddings.word_embeddings.weight.data
        self.encoder.pos_embedding.weight.data = pretrained_model.embeddings.position_embeddings.weight.data
        self.encoder.seg_embedding.weight.data = pretrained_model.embeddings.token_type_embeddings.weight.data

        for i, layer in enumerate(self.encoder.layers):
            layer.self_atten.fc_q.weight.data = pretrained_model.encoder.layer[i].attention.self.query.weight.data
            layer.self_atten.fc_q.bias.data = pretrained_model.encoder.layer[i].attention.self.query.bias.data
            layer.self_atten.fc_k.weight.data = pretrained_model.encoder.layer[i].attention.self.key.weight.data
            layer.self_atten.fc_k.bias.data = pretrained_model.encoder.layer[i].attention.self.key.bias.data
            layer.self_atten.fc_v.weight.data = pretrained_model.encoder.layer[i].attention.self.value.weight.data
            layer.self_atten.fc_v.bias.data = pretrained_model.encoder.layer[i].attention.self.value.bias.data
            layer.self_atten.fc_o.weight.data = pretrained_model.encoder.layer[i].attention.output.dense.weight.data
            layer.self_atten.fc_o.bias.data = pretrained_model.encoder.layer[i].attention.output.dense.bias.data
            layer.FF.linear[0].weight.data = pretrained_model.encoder.layer[i].intermediate.dense.weight.data
            layer.FF.linear[0].bias.data = pretrained_model.encoder.layer[i].intermediate.dense.bias.data
            layer.FF.linear[3].weight.data = pretrained_model.encoder.layer[i].output.dense.weight.data
            layer.FF.linear[3].bias.data = pretrained_model.encoder.layer[i].output.dense.bias.data

    def make_enc_mask(self, x):
        # Create an encoder mask for the input sequence
        enc_mask = (x == self.pad_idx).unsqueeze(1).unsqueeze(2)
        enc_mask = enc_mask.expand(x.shape[0], self.n_heads, x.shape[1], x.shape[1])
        return enc_mask

    def forward(self, x, seg, enc_mask, atten_map_save=False):
        # Generate the encoder mask
        enc_mask = self.make_enc_mask(x)
        # Pass the input through the encoder
        out, atten_encs = self.encoder(x, seg, enc_mask, atten_map_save=atten_map_save)
        # Return the encoder output and attention maps
        return out, atten_encs



class BERTForToxicity(nn.Module):
    def __init__(self, bert, d_model, n_classes):
        super().__init__()
        self.bert = bert
        # Dropout before the final MLP
        self.drop = nn.Dropout(p=0.3)
        # Final linear classification layer
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask, seg_ids):
        # Get the output from bert
        bert_output, _ = self.bert(input_ids, seg_ids, attention_mask)
        pooled_output = bert_output[:, 0]
        # Apply dropout and pass through the classifcation layer
        output = self.drop(pooled_output)
        return self.out(output)