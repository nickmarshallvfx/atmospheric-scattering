# AOV Implementation Findings

## Current State (2024-01-09)

### What Works
| Component | Status | Notes |
|-----------|--------|-------|
| **Transmittance AOV** | ✅ WORKING | Correct values from t_final in Step 2.4c |
| **Node naming** | ✅ WORKING | All nodes found by name: d, mu, r, r_p, mu_p, t_rgb, t_times_spt |
| **Step 2.4 standalone** | ✅ WORKING | Combined pass correct when run alone |

### What's Broken
| Component | Status | Notes |
|-----------|--------|-------|
| **Combined pass** | ❌ BROKEN | After Step 2.4c runs |
| **Rayleigh AOV** | ❌ BROKEN | Shows wrong values (possibly combined Rayleigh+Mie?) |
| **Mie AOV** | ⚠️ PARTIAL | Has values, correctness unclear |

## Root Cause Analysis

### The Problem
Step 2.4c modifies the node graph created by Step 2.4:
1. Removes connection: `t_rgb -> t_times_spt`
2. Creates new LUT transmittance nodes
3. Connects: `t_final -> t_times_spt`

This modification **breaks the inscatter calculation** because:
- The inscatter formula is: `S_cam - T × S_pt`
- `t_times_spt` computes `T × S_pt`
- When we disconnect `t_rgb`, the inscatter chain is broken

### Key Insight
The Rayleigh and Mie AOVs are added in Step 2.4 **before** Step 2.4c modifies anything.
But they tap into `ray_r`, `ray_g`, `ray_b`, and `mie_result` which come **after** the inscatter calculation.

The inscatter node (`inscatter`) outputs `S_cam - T × S_pt`.
Then that feeds into:
- `sep_rgb` (separate RGB)
- `clamp_r/g/b` 
- `ray_r/g/b` (multiply by Rayleigh phase)
- `mie_result` (multiply by Mie phase)

So if the inscatter calculation is broken, ray_r/g/b and mie_result are also broken.

## Solution Options

### Option A: Create Step 2.4c as standalone (no modification)
- Duplicate Step 2.4 code but with LUT transmittance built-in
- Pros: No fragile modifications
- Cons: Code duplication

### Option B: Fix the modification logic
- Current code removes t_rgb connection but something else is breaking
- Need to trace exactly what's wrong

### Option C: Different architecture
- Have Step 2.4 NOT create AOVs
- Have Step 2.4c create ALL AOVs after modifications complete
- This ensures AOVs tap into the final, correct node graph

## Named Nodes in Step 2.4
```
Helios_D        - distance (d)
Helios_R        - camera radius (r)
Helios_Mu       - view zenith cosine (mu)
Helios_R_P      - point radius (r_p)
Helios_Mu_P     - point zenith cosine (mu_p)
Helios_T_RGB    - transmittance as RGB (t_rgb)
Helios_TxSpt    - T × S_pt mix node (t_times_spt)
```

## Implementation History

### Attempt 1: AOVs in Step 2.4
- Added Rayleigh and Mie AOVs in Step 2.4 using direct node refs
- Result: AOVs created but Step 2.4c modification broke them
- Problem: AOVs tap into nodes that get corrupted when t_rgb disconnected

### Attempt 2: Named nodes + AOVs in Step 2.4
- Named all geometry nodes: d, mu, r, r_p, mu_p, t_rgb, t_times_spt
- Named output nodes: ray_r, ray_g, ray_b, mie_result
- Result: Transmittance AOV correct, Rayleigh/Combined broken
- Problem: AOVs still created before modifications complete

### Attempt 3: ALL AOVs in Step 2.4c (CURRENT)
- Removed AOV code from Step 2.4
- Added ALL AOVs in Step 2.4c AFTER LUT transmittance modification
- Nodes found by name: ray_r, ray_g, ray_b, mie_result, t_final
- Theory: AOVs now tap into the FINAL node graph state

## Named Nodes (Complete List)
```
# Geometry (in Step 2.4)
Helios_D         - distance (d)
Helios_R         - camera radius (r)
Helios_Mu        - view zenith cosine (mu)
Helios_R_P       - point radius (r_p)
Helios_Mu_P      - point zenith cosine (mu_p)

# Transmittance (in Step 2.4)
Helios_T_RGB     - transmittance as RGB (t_rgb)
Helios_TxSpt     - T × S_pt mix node (t_times_spt)

# Output components (in Step 2.4)
Helios_Ray_R     - Rayleigh R channel with phase
Helios_Ray_G     - Rayleigh G channel with phase
Helios_Ray_B     - Rayleigh B channel with phase
Helios_Mie_Result - Mie with phase (grayscale)
```

## Next Steps
1. Test current implementation (ALL AOVs in Step 2.4c)
2. If still broken, investigate WHY the inscatter calculation fails
