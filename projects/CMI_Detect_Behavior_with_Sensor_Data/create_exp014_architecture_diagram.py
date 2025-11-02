"""
Create exp014 model architecture diagram using torchviz
"""

import sys
from pathlib import Path

import torch

# Add codes directory to path
sys.path.append(str(Path(__file__).resolve().parent / "codes" / "exp" / "exp014"))


def create_architecture_diagram():
    """Create CMISqueezeformerHybrid architecture diagram using torchviz"""

    try:
        # Import necessary modules
        from config import Exp014Config
        from model import CMISqueezeformerHybrid

        print("Creating exp014 architecture diagram...")

        # Create config with small parameters for visualization
        config = Exp014Config()

        # Override some config for smaller diagram
        config.model.d_model = 128
        config.model.n_layers = 2
        config.model.n_heads = 4
        config.model.d_ff = 256
        config.model.fusion_dim = 256
        config.model.minirocket_input_dim = 100  # Smaller for demo
        config.model.imu_input_dim = 7
        config.model.num_classes = 18
        config.demographics.enabled = True

        # Initialize model
        model = CMISqueezeformerHybrid(config)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        seq_length = 100

        # IMU input: [batch, input_dim, seq_len]
        imu_input = torch.randn(batch_size, config.model.imu_input_dim, seq_length)

        # MiniRocket features: [batch, minirocket_dim]
        minirocket_features = torch.randn(batch_size, config.model.minirocket_input_dim)

        # Attention mask: [batch, seq_len]
        attention_mask = torch.ones(batch_size, seq_length).bool()

        # Demographics (dummy)
        demographics = {
            "age": torch.tensor([25, 30], dtype=torch.float32),
            "sex": torch.tensor([0, 1], dtype=torch.long),  # 0=Male, 1=Female
            "height": torch.tensor([175.0, 165.0], dtype=torch.float32),
            "shoulder_to_wrist": torch.tensor([65.0, 60.0], dtype=torch.float32),
            "elbow_to_wrist": torch.tensor([30.0, 28.0], dtype=torch.float32),
        }

        print("Running forward pass...")

        # Forward pass
        with torch.no_grad():
            multiclass_logits, binary_logits = model(
                imu=imu_input,
                minirocket_features=minirocket_features,
                attention_mask=attention_mask,
                demographics=demographics,
            )

        print("Model output shapes:")
        print(f"  - Multiclass logits: {multiclass_logits.shape}")
        print(f"  - Binary logits: {binary_logits.shape}")

        # Create architecture diagram using torchviz
        try:
            from torchviz import make_dot

            # Create computational graph
            dot = make_dot(
                (multiclass_logits, binary_logits),
                params=dict(model.named_parameters()),
                show_attrs=True,
                show_saved=True,
            )

            # Save the diagram
            output_dir = Path("outputs/claude")
            output_dir.mkdir(exist_ok=True)

            diagram_path = output_dir / "exp014_architecture_diagram"
            dot.render(diagram_path, format="png", cleanup=True)

            print(f"✓ Architecture diagram saved: {diagram_path}.png")

            # Also create a simplified version focusing on the high-level structure
            simplified_dot = make_dot(
                multiclass_logits,  # Only show multiclass branch
                params={
                    k: v
                    for k, v in model.named_parameters()
                    if "multiclass" in k or "feature_fusion" in k or "imu" in k or "minirocket" in k
                },
                show_attrs=False,
                show_saved=False,
            )

            simplified_path = output_dir / "exp014_architecture_simplified"
            simplified_dot.render(simplified_path, format="png", cleanup=True)

            print(f"✓ Simplified diagram saved: {simplified_path}.png")

            return str(diagram_path) + ".png", str(simplified_path) + ".png"

        except ImportError:
            print("⚠️  torchviz not available. Installing...")
            import subprocess

            subprocess.run([sys.executable, "-m", "pip", "install", "torchviz", "graphviz"], check=True)
            print("✓ torchviz installed. Please re-run this script.")
            return None, None

    except Exception as e:
        print(f"❌ Failed to create architecture diagram: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def create_manual_architecture_description():
    """Create a text-based architecture description"""

    architecture_desc = """
## 🏗️ CMISqueezeformerHybrid Architecture

### Input Processing
```
IMU Input: [batch, 7, seq_len] (acc_x, acc_y, acc_z, rot_x, rot_y, rot_z, rot_w)
MiniRocket Features: [batch, ~10000] (from 9 physics features)
Demographics: {age, sex, height, shoulder_to_wrist, elbow_to_wrist}
Attention Mask: [batch, seq_len]
```

### Dual-Branch Architecture

#### IMU Branch (Squeezeformer)
```
IMU Input [batch, 7, seq_len]
    ↓ transpose → [batch, seq_len, 7]
    ↓ Linear Projection → [batch, seq_len, d_model]  
    ↓ Positional Encoding → [batch, seq_len, d_model]
    ↓ SqueezeformerBlock × n_layers → [batch, seq_len, d_model]
    ↓ Masked Global Pooling → [batch, d_model]
IMU Features [batch, d_model]
```

#### MiniRocket Branch
```
MiniRocket Features [batch, ~10000]
    ↓ MiniRocketBranch (MLP)
      ├─ Linear(~10000 → d_model*4) + LayerNorm + SiLU + Dropout
      ├─ Linear(d_model*4 → d_model*2) + LayerNorm + SiLU + Dropout  
      ├─ Linear(d_model*2 → d_model) + LayerNorm + SiLU + Dropout
      └─ Linear(d_model → d_model)
    ↓
MiniRocket Features [batch, d_model]
```

#### Feature Fusion
```
IMU Features [batch, d_model] ─┐
                                ├─ HybridFeatureFusion
MiniRocket Features [batch, d_model] ─┘
    ↓
    ├─ Concatenation: concat → Linear(2*d_model → fusion_dim)
    ├─ Addition: project_both → add → LayerNorm + SiLU  
    └─ Attention: Q=imu, K=both, V=both → weighted_sum
    ↓
Fused Features [batch, fusion_dim]
```

#### Demographics Integration
```
Demographics {age, sex, height, shoulder_to_wrist, elbow_to_wrist}
    ↓ DemographicsEmbedding
    ├─ Categorical: sex → Embedding(2 → emb_dim)
    ├─ Numerical: age, height, s2w, e2w → Normalize → Linear
    └─ Combine → [batch, demo_dim]
    ↓
Final Features = concat([Fused Features, Demographics]) 
    → [batch, fusion_dim + demo_dim]
```

#### Classification Heads
```
Final Features [batch, fusion_dim + demo_dim] ─┐
                                               ├─ Multiclass Head
    LayerNorm → Dropout → Linear(→ fusion_dim//2) ─┤  ├─ Linear(→ 18 classes)
    → SiLU → Dropout → Linear                      ─┘
                                               ┐
                                               ├─ Binary Head  
    LayerNorm → Dropout → Linear(→ fusion_dim//4) ─┤  ├─ Linear(→ 1)
    → SiLU → Dropout → Linear                      ─┘
```

### Loss Computation
```
Multiclass Logits [batch, 18] → CrossEntropyLoss / ACLS / LabelSmoothing
Binary Logits [batch, 1] → BCEWithLogitsLoss / ACLSBinary / LabelSmoothingBCE

Total Loss = α * Multiclass Loss + (1-α) * Binary Loss
```

### Model Parameters
- **d_model**: 512 (unified feature dimension)
- **fusion_dim**: 1024 (after feature fusion)  
- **n_layers**: 6 (Squeezeformer blocks)
- **n_heads**: 8 (attention heads)
- **d_ff**: 2048 (feedforward dimension)
- **fusion_method**: "concatenation" | "addition" | "attention"

### Key Innovations
1. **Dual-Branch Design**: Combines sequential (Squeezeformer) and static (MiniRocket) features
2. **Flexible Fusion**: Multiple fusion strategies for optimal feature combination
3. **Full exp013 Compatibility**: Maintains all existing functionality
4. **Efficient Caching**: MiniRocket features pre-computed and cached
"""

    return architecture_desc


if __name__ == "__main__":
    print("=== exp014 Architecture Diagram Creation ===")

    # Try to create torchviz diagram
    diagram_path, simplified_path = create_architecture_diagram()

    # Create manual description
    manual_desc = create_manual_architecture_description()

    if diagram_path and simplified_path:
        print("\n✅ Architecture diagrams created:")
        print(f"  - Full diagram: {diagram_path}")
        print(f"  - Simplified diagram: {simplified_path}")
    else:
        print("\n⚠️  Torchviz diagrams not created, but manual description available")

    print("\n📝 Manual architecture description created")
    print("\nArchitecture information ready for report integration!")
