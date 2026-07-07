import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class WarmStartGRL(nn.Module):
    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=False):
        super(WarmStartGRL, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.iter_num += 1
        return GradientReverseFunction.apply(input, coeff)

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, K_length, scale_factors=[1, 2, 4]):
        super(MultiScaleTemporalConv, self).__init__()
        self.convs = nn.ModuleList()
        for h_i in scale_factors:
            kernel_size = K_length // h_i
            padding = kernel_size // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.LeakyReLU(),
                    nn.AdaptiveAvgPool1d(out_channels)
                )
            )

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            out = conv(x)
            outputs.append(out.unsqueeze(2))
        return torch.cat(outputs, dim=2)

class DynamicTemporalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(DynamicTemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = feature_dim // num_heads
        
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)
        self.W_O = nn.Linear(feature_dim, feature_dim)
        
        self.Pi_inv = nn.Linear(feature_dim, feature_dim)

    def forward(self, R):
        batch_size, num_scales, d_model = R.size()
        
        Q = self.W_Q(R).view(batch_size, num_scales, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(R).view(batch_size, num_scales, self.num_heads, self.d_k).transpose(1, 2)
        
        V_raw = self.W_V(R)
        V_proj = self.Pi_inv(V_raw)
        V = V_proj.view(batch_size, num_scales, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        Z_head = torch.matmul(attn_weights, V)
        Z_head = Z_head.transpose(1, 2).contiguous().view(batch_size, num_scales, d_model)
        
        Z = self.W_O(Z_head)
        return Z

class ContextualGateLayer(nn.Module):
    def __init__(self, feature_dim):
        super(ContextualGateLayer, self).__init__()
        self.fc_g = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, Z):
        c = torch.mean(Z, dim=0, keepdim=True)
        c_expanded = c.expand(Z.size(0), -1, -1)
        
        concat_input = torch.cat([Z, c_expanded], dim=-1)
        w = torch.sigmoid(self.fc_g(concat_input))
        
        if self.training:
            l1_loss = torch.mean(torch.abs(w))
        else:
            l1_loss = 0.0
            
        Z_prime = Z * w
        return Z_prime, l1_loss

class MDTN(nn.Module):
    def __init__(self, in_channels, hidden_dim, K_length=16):
        super(MDTN, self).__init__()
        self.ms_conv = MultiScaleTemporalConv(in_channels, hidden_dim, K_length)
        self.dta = DynamicTemporalAttention(hidden_dim)
        self.ctx_gate = ContextualGateLayer(hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        R = self.ms_conv(x)
        R_combined = torch.sum(R, dim=2) 
        
        Z = self.dta(R_combined.unsqueeze(1)) 
        Z_prime, l1_loss = self.ctx_gate(Z)
        
        output = self.fc_final(Z_prime.squeeze(1))
        return output, l1_loss

class ChebyNetLayer(nn.Module):
    def __init__(self, K, in_features, out_features):
        super(ChebyNetLayer, self).__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, L):
        batch_size, num_nodes, in_feat = x.size()
        
        x0 = x
        x1 = torch.matmul(L, x)
        
        cheby_poly = [x0, x1]
        for k in range(2, self.K):
            xk = 2 * torch.matmul(L, cheby_poly[-1]) - cheby_poly[-2]
            cheby_poly.append(xk)
            
        cheby_poly = torch.stack(cheby_poly, dim=0)
        
        out = torch.einsum('kbni,kio->bno', cheby_poly, self.weight)
        out = out + self.bias
        return F.relu(out)

class ChebyDiscriminator(nn.Module):
    def __init__(self, num_nodes, feature_dim, num_classes, K=3):
        super(ChebyDiscriminator, self).__init__()
        self.num_nodes = num_nodes
        self.A_0 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        nn.init.uniform_(self.A_0, 0, 1)
        
        self.cheby_layer = ChebyNetLayer(K, feature_dim, feature_dim)
        
        self.fc_domain = nn.Linear(feature_dim, 1)
        self.fc_emotion = nn.Linear(feature_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def get_adj(self):
        A = F.relu(self.A_0 + self.A_0.t())
        I = torch.eye(self.num_nodes).to(self.A_0.device)
        return A + I

    def get_laplacian(self, A):
        D = torch.sum(A, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(D + 1e-5, -0.5))
        L = torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
        return -L

    def forward(self, x):
        A = self.get_adj()
        L = self.get_laplacian(A)
        
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.num_nodes, -1)
            
        L_batch = L.unsqueeze(0).expand(x.size(0), -1, -1)
        
        Z = self.cheby_layer(x, L_batch)
        H_node = self.dropout(Z)
        h = torch.mean(H_node, dim=1)
        
        domain_pred = torch.sigmoid(self.fc_domain(h))
        emotion_pred = self.fc_emotion(h)
        
        return domain_pred, emotion_pred, h, A

class MDTN_GMDA_Model(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, num_nodes, max_iter=1000):
        super(MDTN_GMDA_Model, self).__init__()
        self.feature_extractor = MDTN(in_channels, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.grl = WarmStartGRL(max_iters=max_iter)
        self.discriminator = ChebyDiscriminator(num_nodes, hidden_dim, num_classes)

    def forward(self, x_s, x_t=None):
        f_s, l1_loss_s = self.feature_extractor(x_s)
        y_s = self.classifier(f_s)
        
        if self.training and x_t is not None:
            f_t, l1_loss_t = self.feature_extractor(x_t)
            y_t = self.classifier(f_t) 
            
            f_combined = torch.cat((f_s, f_t), dim=0)
            f_reversed = self.grl(f_combined)
            
            d_pred, e_pred_disc, h_graph, adj_matrix = self.discriminator(f_reversed)
            
            l1_loss = (l1_loss_s + l1_loss_t) / 2
            
            return y_s, y_t, d_pred, e_pred_disc, h_graph, adj_matrix, l1_loss
        
        return y_s

class MDTN_GMDA_Loss(nn.Module):
    def __init__(self, lambda_match=0.1, alpha=0.01, beta=0.01):
        super(MDTN_GMDA_Loss, self).__init__()
        self.lambda_match = lambda_match
        self.alpha = alpha 
        self.beta = beta
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def get_mmd(self, source, target):
        delta = source.mean(0) - target.mean(0)
        return torch.sum(delta ** 2)

    def get_conditional_mmd(self, source, target, s_label, t_label_pred, num_classes):
        loss = 0.0
        for c in range(num_classes):
            s_c = source[s_label == c]
            t_c = target[t_label_pred == c]
            
            if len(s_c) > 0 and len(t_c) > 0:
                loss += self.get_mmd(s_c, t_c)
        return loss

    def forward(self, y_s, y_t_logits, d_pred, e_pred_disc, h_graph, adj_matrix, label_s, l1_loss):
        batch_size = y_s.size(0)
        num_classes = y_s.size(1)

        L_cls = self.ce(y_s, label_s)
        
        d_label_s = torch.ones(batch_size, 1).to(y_s.device)
        d_label_t = torch.zeros(batch_size, 1).to(y_s.device)
        d_labels = torch.cat((d_label_s, d_label_t), dim=0)
        L_dis = self.bce(d_pred, d_labels)
        
        h_s, h_t = h_graph.split(batch_size, dim=0)
        
        D_marginal = self.get_mmd(h_s, h_t)
        
        t_pseudo_label = torch.argmax(y_t_logits, dim=1)
        s_label_idx = torch.argmax(label_s, dim=1) if label_s.dim() > 1 else label_s
        D_conditional = self.get_conditional_mmd(h_s, h_t, s_label_idx, t_pseudo_label, num_classes)
        
        A_s_proxy = torch.mm(h_s, h_s.t())
        A_t_proxy = torch.mm(h_t, h_t.t())
        A_s_norm = F.normalize(A_s_proxy, p=2, dim=1)
        A_t_norm = F.normalize(A_t_proxy, p=2, dim=1)
        L_sim = torch.norm(A_s_norm - A_t_norm, p='fro') ** 2
        
        e_pred_s, _ = e_pred_disc.split(batch_size, dim=0)
        L_cls_graph = self.ce(e_pred_s, s_label_idx)
        
        L_match = L_cls_graph + self.lambda_match * L_sim
        
        L_final = L_dis + L_match + self.alpha * D_marginal + self.beta * D_conditional + 0.01 * l1_loss + L_cls
        
        return L_final, L_cls, L_dis, L_match
