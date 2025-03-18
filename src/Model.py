import torch
import torch.nn as nn
import torch.nn.functional as F
import string
ALPHABET = string.ascii_lowercase + string.digits + "."
char2idx = {c: i + 1 for i, c in enumerate(ALPHABET)}  # padding=0
idx2char = {i: c for c, i in char2idx.items()}       # Reverse mapping index -> character
vocab_size = len(char2idx) + 1
MAX_LEN = 50

# ------------------- CNN MODEL FOR CLASSIFICATION ----------------------

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size=vocab_size, embed_dim=32, num_filters=128, kernel_sizes=(3, 5, 7, 9), num_classes=2,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding='same'),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ) for k in kernel_sizes
        ])
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        # Apply convolution and pooling
        conv_outputs = [conv(emb).squeeze(2) for conv in self.conv_layers]  # List of (batch_size, num_filters)
        # Concatenate features
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        # Fully Connected layers
        logits = self.fc(x)  # (batch_size, num_classes)
        return logits


# ------------------- BiLSTM MODEL FOR CLASSIFICATION ----------------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        bilstm_out, _ = self.bilstm(emb)  # (batch_size, seq_len, hidden_dim*2)

        # Attention Mechanism
        attn_weights = F.softmax(self.attention(bilstm_out).squeeze(2), dim=1)  # (batch_size, seq_len)
        attn_output = torch.bmm(attn_weights.unsqueeze(1), bilstm_out).squeeze(1)  # (batch_size, hidden_dim*2)

        attn_output = self.dropout(attn_output)
        logits = self.fc(attn_output)
        return logits


# ------------------- TRANSFORMER WITH POSITIONAL ENCODING MODEL FOR CLASSIFICATION ----------------------

# Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


# Transformer Model for DGA Detection
class PositionalEncodingTransformer(nn.Module):
    def __init__(self, vocab_size=vocab_size, embed_dim=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.2,
                 num_classes=2):
        super().__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True  # Fix the warning by enabling batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully Connected Output Layer
        self.fc = nn.Linear(embed_dim, num_classes)

        # Dropout & Layer Normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Embedding Layer
        emb = self.embedding(x)  # (B, L, E)
        emb = self.positional_encoding(emb)  # Add positional encodings
        emb = self.dropout(emb)  # Apply dropout
        emb = self.layer_norm(emb)  # Apply Layer Normalization

        # Transformer Encoder (no permute needed now)
        out = self.transformer_encoder(emb)  # (B, L, E)

        # Extract final representation (use first token's output)
        out_final = out[:, 0, :]  # (Batch, Embed_Dim)

        # Fully Connected Output
        logits = self.fc(out_final)  # (Batch, Num_Classes)

        return logits

# ------------------- AUTOENCODER FOR ANOMALY DETECTION ----------------------

class DomainVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=64, latent_dim=32, dropout_p=0.2):
        """
        vocab_size: kích thước từ vựng
        embed_dim: kích thước embedding cho từng token
        hidden_dim: số chiều ẩn của LSTM (sử dụng BiLSTM nên kết quả sẽ gấp đôi)
        latent_dim: kích thước không gian latent
        dropout_p: tỷ lệ dropout
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoder sử dụng BiLSTM, hidden_dim nhân đôi vì hai hướng
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Từ hidden state kết hợp, ta tính cả trung bình (mu) và log-variance (logvar)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        """
        Đầu vào: x có shape (B, L)
        Đầu ra: mu và logvar có shape (B, latent_dim)
        """
        emb = self.dropout(self.embedding(x))  # (B, L, embed_dim)
        enc_out, (h, c) = self.encoder_lstm(emb)  # h có shape (num_directions, B, hidden_dim)
        # Ghép 2 hidden states từ cả 2 chiều
        h_cat = torch.cat((h[0], h[1]), dim=1)  # (B, hidden_dim*2)
        mu = self.fc_mu(h_cat)      # (B, latent_dim)
        logvar = self.fc_logvar(h_cat)  # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Áp dụng trick reparameterization để lấy mẫu z từ phân phối N(mu, sigma^2)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, max_len=MAX_LEN):
        """
        Giải mã latent vector z thành chuỗi domain
        """
        h_dec = self.fc_dec(z).unsqueeze(0)  # (1, B, hidden_dim)
        c_dec = torch.zeros_like(h_dec)       # (1, B, hidden_dim)
        
        # Sử dụng input ban đầu là các token 0 (có thể hiểu là token <SOS>)
        dec_input = torch.zeros((z.size(0), max_len), device=z.device, dtype=torch.long)  # (B, L)
        emb_dec = self.embedding(dec_input)  # (B, L, embed_dim)
        
        dec_out, (h_new, c_new) = self.decoder_lstm(emb_dec, (h_dec, c_dec))  # (B, L, hidden_dim)
        logits = self.fc_out(dec_out)  # (B, L, vocab_size)
        return logits

    def forward(self, x):
        """
        Forward: from x -> (mu, logvar) -> sample z -> decode -> logits
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, max_len=x.size(1))
        return logits, mu, logvar

    def loss_function(self, x, logits, mu, logvar):
        """
        Hàm loss tổng hợp: reconstruction loss (cross entropy) và KL divergence
        """
        B, L = x.size()
        # Tính reconstruction loss cho từng timestep
        logits_2d = logits.view(B * L, -1)
        target_1d = x.view(B * L)
        recon_loss = F.cross_entropy(logits_2d, target_1d, ignore_index=0)
        
        # Tính KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / B  # Có thể chia theo batch size
        
        return recon_loss + kl_loss, recon_loss, kl_loss
    
    def compute_reconstruction_error(self, x):
        """
        Tính loss tổng hợp làm chỉ số error để phân biệt benign và DGA.
        """
        self.eval()
        with torch.no_grad():
            logits, mu, logvar = self.forward(x)
            loss, recon_loss, kl_loss = self.loss_function(x, logits, mu, logvar)
        return loss.item()
class SimpleCNN(nn.Module):
    '''
    SimpleCNN for MNIST
    '''

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class LeNet_MNIST(nn.Module):
    '''
    LeNet for MNIST
    '''

    def __init__(self):
        super(LeNet_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet_CIFAR10(nn.Module):
    '''
    LeNet for CIFAR10
    '''

    def __init__(self):
        super(LeNet_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    '''
    ResNet for CIFAR10
    '''

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class VGG16(torch.nn.Module):
    '''
    VGG16 for CIFAR10
    '''

    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(64)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer9 = nn.BatchNorm2d(128)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer12 = nn.BatchNorm2d(128)
        self.layer13 = nn.ReLU()
        self.layer14 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer15 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer16 = nn.BatchNorm2d(256)
        self.layer17 = nn.ReLU()
        self.layer18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer19 = nn.BatchNorm2d(256)
        self.layer20 = nn.ReLU()
        self.layer21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer22 = nn.BatchNorm2d(256)
        self.layer23 = nn.ReLU()
        self.layer24 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer25 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer26 = nn.BatchNorm2d(512)
        self.layer27 = nn.ReLU()
        self.layer28 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer29 = nn.BatchNorm2d(512)
        self.layer30 = nn.ReLU()
        self.layer31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer32 = nn.BatchNorm2d(512)
        self.layer33 = nn.ReLU()
        self.layer34 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer35 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer36 = nn.BatchNorm2d(512)
        self.layer37 = nn.ReLU()
        self.layer38 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer39 = nn.BatchNorm2d(512)
        self.layer40 = nn.ReLU()
        self.layer41 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer42 = nn.BatchNorm2d(512)
        self.layer43 = nn.ReLU()
        self.layer44 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer45 = nn.Flatten(1, -1)
        self.layer46 = nn.Dropout(0.5)
        self.layer47 = nn.Linear(1 * 1 * 512, 4096)
        self.layer48 = nn.ReLU()
        self.layer49 = nn.Dropout(0.5)
        self.layer50 = nn.Linear(4096, 4096)
        self.layer51 = nn.ReLU()
        self.layer52 = nn.Linear(4096, 10)

    def forward(self, input0):
        out1 = self.layer1(input0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        return out52


class VGG19(torch.nn.Module):
    '''
    VGG19 for CIFAR10
    '''

    def __init__(self):
        super(VGG19, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(64)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer9 = nn.BatchNorm2d(128)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer12 = nn.BatchNorm2d(128)
        self.layer13 = nn.ReLU()
        self.layer14 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer15 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer16 = nn.BatchNorm2d(256)
        self.layer17 = nn.ReLU()
        self.layer18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer19 = nn.BatchNorm2d(256)
        self.layer20 = nn.ReLU()
        self.layer21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer22 = nn.BatchNorm2d(256)
        self.layer23 = nn.ReLU()
        self.layer24 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer25 = nn.BatchNorm2d(256)
        self.layer26 = nn.ReLU()
        self.layer27 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer28 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer29 = nn.BatchNorm2d(512)
        self.layer30 = nn.ReLU()
        self.layer31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer32 = nn.BatchNorm2d(512)
        self.layer33 = nn.ReLU()
        self.layer34 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer35 = nn.BatchNorm2d(512)
        self.layer36 = nn.ReLU()
        self.layer37 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer38 = nn.BatchNorm2d(512)
        self.layer39 = nn.ReLU()
        self.layer40 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer41 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer42 = nn.BatchNorm2d(512)
        self.layer43 = nn.ReLU()
        self.layer44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer45 = nn.BatchNorm2d(512)
        self.layer46 = nn.ReLU()
        self.layer47 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer48 = nn.BatchNorm2d(512)
        self.layer49 = nn.ReLU()
        self.layer50 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer51 = nn.BatchNorm2d(512)
        self.layer52 = nn.ReLU()
        self.layer53 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer54 = nn.Flatten(1, -1)
        self.layer55 = nn.Dropout(0.5)
        self.layer56 = nn.Linear(1 * 1 * 512, 4096)
        self.layer57 = nn.ReLU()
        self.layer58 = nn.Dropout(0.5)
        self.layer59 = nn.Linear(4096, 4096)
        self.layer60 = nn.ReLU()
        self.layer61 = nn.Linear(4096, 10)

    def forward(self, input0):
        out1 = self.layer1(input0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        out53 = self.layer53(out52)
        out54 = self.layer54(out53)
        out55 = self.layer55(out54)
        out56 = self.layer56(out55)
        out57 = self.layer57(out56)
        out58 = self.layer58(out57)
        out59 = self.layer59(out58)
        out60 = self.layer60(out59)
        out61 = self.layer61(out60)
        return out61


class DGAClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(DGAClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(output).squeeze()


def LSTM():
    """
    LSTM for domain detection
    :return:
    """
    return DGAClassifier(128, 32, 64, 1)
