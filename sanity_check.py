import torch
import torch.nn as nn
from m3ae.modules.mambapy.mamba import Mamba, MambaConfig, AdaptiveKGFusion
from m3ae.modules.prediction_heads import Pooler
from m3ae.modules.language_encoders.bert_model import GAT_module

def run_sanity_check():
    print("=== Starting MedVQA Architecture Sanity Check ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    batch_size = 2
    seq_len = 32
    hidden_dim = 768

    # 1. Test Semantic-Aligned Bidirectional Mamba
    print("\n[1/4] Testing Mamba Multi-Modal Module...")
    mamba_cfg = MambaConfig(d_model=hidden_dim, n_layers=2)
    mamba_mod = Mamba(mamba_cfg).to(device)
    
    x_vis = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    x_txt = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)

    out_v, out_q = mamba_mod(x_vis, x_txt, mask_v=mask, mask_q=mask)
    assert out_v.shape == (batch_size, seq_len, hidden_dim)
    assert out_q.shape == (batch_size, seq_len, hidden_dim)
    print("✓ Mamba Forward Pass Passed!")

    # 2. Test Adaptive Knowledge Graph Fusion
    print("\n[2/4] Testing Adaptive KG Fusion Module...")
    kg_fusion = AdaptiveKGFusion(d_model=hidden_dim).to(device)
    t_z2t = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    t_k2t = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    k_t2k = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    fused_text, fused_vis = kg_fusion(t_z2t, t_k2t, k_t2k, z_t2z=x_vis)
    assert fused_text.shape == (batch_size, seq_len, hidden_dim)
    assert fused_vis.shape == (batch_size, seq_len, hidden_dim)
    print("✓ Adaptive KG Fusion Passed!")

    # 3. Test Corrected GAT Module & Organ Average Projection
    print("\n[3/4] Testing GAT Graph Attention Module...")
    gat_mod = GAT_module(feature_size=hidden_dim, head=8).to(device)
    organ_avg = nn.Conv1d(577, 32, 1).to(device)
    num_nodes = 577
    node_feats = torch.randn(batch_size, num_nodes, hidden_dim, device=device)
    adj_mat = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    gat_out = gat_mod(x_txt, node_feats, adj_mat)
    assert gat_out.shape == (batch_size, num_nodes, hidden_dim)
    gat_pooled = organ_avg(gat_out)
    assert gat_pooled.shape == (batch_size, 32, hidden_dim)
    print("✓ GAT Module & Projection Passed!")

    # 4. Test Attention-Based Multi-Token Pooler
    print("\n[4/4] Testing Attention Pooler Module...")
    pooler = Pooler(hidden_size=hidden_dim).to(device)
    visual_queries = torch.randn(batch_size, 32, hidden_dim, device=device)
    
    pooled_out = pooler(visual_queries)
    assert pooled_out.shape == (batch_size, hidden_dim)
    print("✓ Pooler Forward Pass Passed!")

    print("\n=== All Sanity Checks Completed Successfully! ===")

if __name__ == "__main__":
    run_sanity_check()