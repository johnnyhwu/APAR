import torch
import torch.nn as nn

class ArithmeticsEstimator(nn.Module):

    class MLP(nn.Module):
        def __init__(self, hidden_dim, out_dim):
            super().__init__()
            self.normalization = nn.LayerNorm((hidden_dim, ))
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(hidden_dim, out_dim)

        def forward(self, x):
            x = x[:, -1]
            x = self.normalization(x)
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x
        
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.addition_estimator = ArithmeticsEstimator.MLP(hidden_dim=hidden_dim*2, out_dim=1)    
    
    def forward(self, x):
        bs = x.shape[0]
        assert bs % 2 == 0
        feat_1 = x[0:int(bs/2), :, :]
        feat_2 = x[int(bs/2):, :, :]
        x = torch.concat([feat_1, feat_2], dim=-1)
        outputs = self.addition_estimator(x)
        return outputs

class Reconstructor(nn.Module):

    class MLP(nn.Module):
        def __init__(self, hidden_dim, out_dim):
            super().__init__()
            self.normalization = nn.LayerNorm((hidden_dim, ))
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(hidden_dim, out_dim)

        def forward(self, x):
            x = x[:, -1]
            x = self.normalization(x)
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x

    class MaskReconstructor(nn.Module):
            
        def __init__(self, hidden_dim, out_dim) -> None:
            super().__init__()
            self.mask_recon = Reconstructor.MLP(hidden_dim=hidden_dim, out_dim=out_dim)  
            self.sigmoid = nn.Sigmoid()  
        
        def forward(self, x):
            x = self.mask_recon(x)
            x = self.sigmoid(x)
            return x
    
    class FeatureReconstructor(nn.Module):
            
        def __init__(self, hidden_dim, out_dim, cat_feat_cardinality) -> None:
            super().__init__()
            self.num_feature_recon = Reconstructor.MLP(hidden_dim=hidden_dim, out_dim=out_dim)
            self.cat_feature_recon = nn.ModuleList([Reconstructor.MLP(hidden_dim=hidden_dim, out_dim=card) for card in cat_feat_cardinality])
        
        def forward(self, x):
            num_feat = self.num_feature_recon(x)
            cat_feat = [recon(x) for recon in self.cat_feature_recon]
            return num_feat, cat_feat
    
    def __init__(self, hidden_dim, num_feat_quantity, cat_feat_cardinality) -> None:
        super().__init__()
        self.mask_recon = Reconstructor.MaskReconstructor(hidden_dim=hidden_dim, out_dim=num_feat_quantity+len(cat_feat_cardinality))
        self.feature_recon = Reconstructor.FeatureReconstructor(hidden_dim=hidden_dim, out_dim=num_feat_quantity, cat_feat_cardinality=cat_feat_cardinality)
    
    def forward(self, x):
        mask = self.mask_recon(x)
        num_feat, cat_feat = self.feature_recon(x)
        return num_feat, cat_feat, mask
    
class PretrainModel(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.enc = encoder
        self.dec = decoder
    def forward(self, num_feat, cat_feat):
        embedding, _ = self.enc(num_feat, cat_feat)
        return self.dec(embedding)

class APARModel(nn.Module):
    def __init__(self, ft_trans, input_dim) -> None:
        super().__init__()
        self.ft_trans = ft_trans
        self.feature_tokenizer = ft_trans.feature_tokenizer
        self.cls_token = ft_trans.cls_token
        self.transformer = ft_trans.transformer

        self.pi_logit = nn.Parameter(torch.zeros((input_dim, )))
    
    def gaussian_cdf(self, x):
        return 0.5 * (1. + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
    def log(self, x): 
        return torch.log(x + 1e-8)

    def forward(self, num_feat, cat_feat, feat_mask, inference=False):
        tokens = self.feature_tokenizer(num_feat, cat_feat)

        self.pi = torch.sigmoid(self.pi_logit)
        u = self.gaussian_cdf(feat_mask)
        m = torch.sigmoid(self.log(self.pi) - self.log(1.0 - self.pi) + self.log(u) - self.log(1.0 - u))
        m = m.unsqueeze(dim=2)
        m = m.repeat((1, 1, 192))
        tokens_aug = tokens * m

        x = self.cls_token(tokens)
        out, _ = self.transformer(x)

        if not inference:
            x = self.cls_token(tokens_aug)
            out_aug, _ = self.transformer(x)
        else:
            out_aug = None
        return out, out_aug