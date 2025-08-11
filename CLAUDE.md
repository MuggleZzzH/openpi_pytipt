---
alwaysApply: true
---
You are a senior engineer with deep experience building production-grade AI agents, automations, and workflow systems. Every task you execute must follow this procedure without exception:
	
1.Clarify Scope First
•Before writing any code, map out exactly how you will approach the task.
•Confirm your interpretation of the objective.
•Write a clear plan showing what functions, modules, or components will be touched and why.
•Do not begin implementation until this is done and reasoned through.
	
2.Locate Exact Code Insertion Point
•Identify the precise file(s) and line(s) where the change will live.
•Never make sweeping edits across unrelated files.
•If multiple files are needed, justify each inclusion explicitly.
•Do not create new abstractions or refactor unless the task explicitly says so.
	
3.Minimal, Contained Changes
•Only write code directly required to satisfy the task.
•Avoid adding logging, comments, tests, TODOs, cleanup, or error handling unless directly necessary.
•No speculative changes or “while we’re here” edits.
•All logic should be isolated to not break existing flows.
	
4.Double Check Everything
•Review for correctness, scope adherence, and side effects.
•Ensure your code is aligned with the existing codebase patterns and avoids regressions.
•Explicitly verify whether anything downstream will be impacted.
	
5.Deliver Clearly
•Summarize what was changed and why.
•List every file modified and what was done in each.
•If there are any assumptions or risks, flag them for review.
	
Reminder: You are not a co-pilot, assistant, or brainstorm partner. You are the senior engineer responsible for high-leverage, production-safe changes. Do not improvise. Do not over-engineer. Do not deviate
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a simplified PyTorch implementation of LeRobot's PI0 and PI0-fast Vision-Language-Action (VLA) models, based on Physical Intelligence's OpenPI checkpoints. The repository provides fixes, optimizations, comprehensive usage documentation, and a complete RIPT (Reinforcement Learning with Image-based Trajectory Prediction) framework for distributed RL fine-tuning.

**Critical Note**: This repository includes extensive debugging and testing files that help diagnose common issues. Many files starting with `test_` are diagnostic tools for specific problems that arose during development.

## Recent Major Enhancements

The repository has recently been enhanced with production-ready features including:

- **Advanced Parallel Environment Support**: Intelligent parallel environment management with automatic memory optimization and batch processing fallbacks to prevent GPU memory overflow
- **Enhanced Testing Infrastructure**: Comprehensive test suites (`test_enhanced_features.py`, `quick_single_gpu_test.py`) that validate all system components
- **File-based Distributed Coordination**: `FileGlobalCounter` system for robust multi-process coordination using file locks
- **SubprocVectorEnv Optimization**: Smart detection and mitigation of model serialization issues in parallel environments
- **Intelligent Memory Management**: Automatic GPU memory analysis and strategy selection for parallel training

## Staged Development Architecture

The codebase follows a **11-stage progressive development approach** from basic inference to production-ready distributed training:

- **Stages 1-6**: `scripts/` directory - Basic inference and environment integration
- **Stage 7**: `7_train_with_rl_core.py` - Core RL training implementation
- **Stage 8**: `8_train_with_epochs.py` - Smart sampling and batch processing
- **Stage 9**: `9_train_with_config.py` - YAML configuration system
- **Stage 10**: `10_train_with_distributed.py` - Multi-GPU distributed training
- **Stage 11**: `11_train_ript_vla_style.py` - RIPT-VLA style simplified training with enhanced parallel environment support

Each stage builds incrementally on the previous ones, maintaining backward compatibility while adding sophisticated features.

## Core Architecture

### Key Components

1. **PI0 Policy Models** (`pi0/` directory):
   - `modeling_pi0.py`: Main PI0 policy implementation with Flow Matching and critical image format fixes
   - `modeling_pi0fast.py`: Optimized PI0-fast variant
   - `paligemma_with_expert.py`: Vision-language model backbone

2. **RIPT RL Training Framework** (`pi0/ript/` directory):
   - `scripts/train_ript_pi0.py`: Distributed RL training script using CFG-style advantage weighting
   - `algos/rl_optimizers/`: Core RL optimization algorithms
     - `pi0_cfg_interface.py`: Critical adapter layer between PI0Policy and RIPT framework
     - `rl_optimizer_pi0_cfg.py`: CFG-style RL optimizer with Leave-One-Out advantage computation
     - `rollout_generator.py`: Trajectory generation and data collection
     - `file_counter.py`: Distributed coordination for multi-process training with atomic file operations
     - `enhanced_rollout_generator.py`: Advanced rollout generation with smart sampling and early stopping
   - `env/`: Environment integration with advanced parallel support
     - `pi0_libero_runner.py`: Enhanced LIBERO environment runner with intelligent parallel environment management
     - `pi0_libero_runner_ript_vla.py`: RIPT-VLA style runner with action queue management
     - `parallel_env_factory.py`: Independent environment factory for solving SubprocVectorEnv serialization issues
   - `utils/`: LIBERO-specific utilities and policy wrappers
   - `config/`: Training configuration files with feature toggles and parallel environment settings

3. **Model Conversion** (`convert_pi0_to_hf_lerobot.py`):
   - Converts JAX checkpoints from OpenPI to PyTorch format
   - Preserves normalization statistics essential for inference

4. **Environment Integration**:
   - **CleanDiffuser/**: Primary LIBERO environment integration (used by RIPT training)
   - **LIBERO/**: Official LIBERO benchmark tasks and utilities  
   - **lerobot/**: Extended LeRobot framework with backward compatibility layers

5. **Testing and Debugging Infrastructure**:
   - Multiple `test_*.py` files for diagnosing specific issues
   - `2_test_pi0_on_libero.py`: Reference implementation for LIBERO environment testing (now with debug checkpoints)
   - `analyze_*.py`: Scripts for performance analysis and debugging
   - **RIPT_QUALITY_DEBUG_GUIDE.md**: Comprehensive debugging guide for quality issues
   - `test_enhanced_features.py`: Production-ready test suite for RIPT-VLA level functionality validation
   - `quick_single_gpu_test.py`: Fast system validation with configurable testing modes
   - `test_file_counter_integration.py`: Distributed coordination testing and validation

## Essential Development Commands

### Prerequisites and Installation
Install core dependencies:
```bash
pip install -r requirements.txt
```

For RIPT training, additional dependencies are required (already included in requirements.txt):
- hydra-core, robosuite, gym for RL training
- bddl for LIBERO environment tasks
- moviepy, imageio for video generation and analysis

### Model Conversion
Convert OpenPI JAX checkpoints to PyTorch:
```bash
python convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir /path/to/jax/checkpoint/params \
    --output_path /path/to/pytorch/checkpoint
```

**Important**: Always preserve the original JAX checkpoint directory as it contains `norm_stats.json` which is critical for proper model inference.

### Running Inference
Basic inference test:
```bash
python 1_e2e_inference.py
```

Libero environment demos:
```bash
python libero_demo_lerobot.py
python 2_test_pi0_on_libero.py
```

### RIPT RL Training
Train PI0 with reinforcement learning using RIPT framework:

**Single GPU Training:**
```bash
cd /zhaohan/ZJH/openpi_pytorch
python pi0/ript/scripts/train_ript_pi0.py --config_path pi0/ript/config/train_pi0_cfg_rl.yaml
```

**Multi-GPU Distributed Training (NEW - Stage 10):**
```bash
# Quick 2GPU test
./scripts/quick_distributed_test.sh

# Standard 4GPU training
./scripts/launch_distributed_training.sh \
    --config pi0/ript/config/distributed_train_pi0.yaml --gpus 4

# Large-scale 8GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/launch_distributed_training.sh \
    --config pi0/ript/config/distributed_train_pi0.yaml --gpus 8
```

**Configuration-based Training (Stage 9-11):**
```bash
# YAML configuration system
python 9_train_with_config.py --config_path pi0/ript/config/debug_train_stage9.yaml

# Distributed configuration system  
python 10_train_with_distributed.py --config_path pi0/ript/config/distributed_train_pi0.yaml

# RIPT-VLA style simplified training with enhanced parallel environment support
python 11_train_ript_vla_style.py --config_path pi0/ript/config/stage11_ript_vla.yaml
```

Debug training with minimal configuration:
```bash
python pi0/ript/scripts/train_ript_pi0.py --config_path pi0/ript/config/debug_train_pi0.yaml
```

### Testing and Validation Commands
```bash
# Environment setup validation
export DEBUG_SAVE_PROCESSED=1      # Save processed images for debugging
export PI0_DEBUG_SAVE_VIDEO=true   # Enable rollout video generation
export TOKENIZERS_PARALLELISM=false # Fix tokenizer warnings

# Comprehensive system validation (RECOMMENDED for new setups)
python test_enhanced_features.py   # Full RIPT-VLA functionality test
python quick_single_gpu_test.py --config_path pi0/ript/config/single_gpu_test.yaml

# Core system validation
python 1_e2e_inference.py          # Basic inference test
python test_ript_normalization.py   # Test normalization consistency
python test_ript_training_quick.py  # Test core training components
python check_libero_format.py       # Verify LIBERO image format

# Component import validation
python -c "from pi0.modeling_pi0 import PI0Policy; print('✓ PI0 import successful')"
python -c "from pi0.ript.reward_function import BinarySuccessReward; print('✓ RIPT import successful')"

# Parallel environment and distributed coordination testing
python test_file_counter_integration.py   # Test distributed coordination
python test_parallel_envs.py              # Test parallel environment functionality

# Performance analysis (when investigating issues)
python test_ript_vs_original.py     # Compare RIPT vs reference implementation
python analyze_inference_differences.py # Detailed inference analysis
```

### Development Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up LIBERO environment (for RIPT training)
# CleanDiffuser and LIBERO should be included in requirements.txt

# Verify GPU setup for distributed training
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Distributed Training Setup
```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python pi0/ript/scripts/train_ript_pi0.py \
    --config_path pi0/ript/config/debug_train_pi0.yaml

# Multi-GPU distributed training  
export NCCL_TIMEOUT=108000  # Extended timeout for distributed operations
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    pi0/ript/scripts/train_ript_pi0.py --config_path pi0/ript/config/train_pi0_cfg_rl.yaml
```

### Debugging Quality Issues
For systematic debugging of quality differences between RIPT and reference implementations:
```bash
# Run reference with debug checkpoints
python 2_test_pi0_on_libero.py

# Run RIPT with debug output  
python pi0/ript/scripts/train_ript_pi0.py --config_path pi0/ript/config/train_pi0_cfg_rl.yaml

# Compare debug outputs
python -c "
import json, numpy as np
ref = json.load(open('debug_2test_raw_obs.json'))
ript = json.load(open('ript/debug_analysis/session_*/checkpoint2_obs_step0.json'))
# Detailed comparison logic...
"
```

## Critical Architecture Details

### RIPT System Integration
The RIPT framework implements CFG-style reinforcement learning for PI0 policies with a sophisticated multi-stage architecture:

1. **Flow Matching + CFG-RL**: PI0 uses time-based interpolation for action generation, optimized with classifier-free guidance style advantage weighting
2. **Distributed Training**: Multi-process coordination through file-based counters and PyTorch DDP
3. **LIBERO Integration**: 32D policy actions mapped to 7D LIBERO environment actions (6-DOF pose + gripper)
4. **Advantage Computation**: Leave-One-Out baseline for trajectory-level advantage estimation without value networks
5. **Smart Sampling**: Enhanced sampling system with state history tracking and intelligent state selection

### Progressive Training Pipeline Evolution
The training system evolved through multiple stages:

- **Stage 7-8**: Core RL training with smart sampling and batch processing
- **Stage 9**: Configuration management with YAML support and parameter validation
- **Stage 10**: Production-ready distributed training with PyTorch DDP

Each stage maintains full backward compatibility while adding sophisticated features.

### Import Path Resolution
The repository maintains compatibility across different frameworks:
- `/zhaohan/ZJH/lerobot/lerobot/constants.py` - Re-exports from `common.constants`
- `/zhaohan/ZJH/lerobot/lerobot/policies/` - Compatibility layer for policy imports
- `lerobot_constants.py` - Local constants file for standalone operation
- **RIPT imports**: All internal RIPT imports use `pi0.ript.*` path structure

### Model Loading Requirements
1. **PyTorch Checkpoint**: Converted from JAX using conversion script
2. **Normalization Statistics**: Essential `norm_stats.json` from original JAX checkpoint
3. **Dataset Statistics**: Must include image feature keys for proper model initialization
4. **RIPT Dependencies**: Additional packages for RL training (hydra-core, robosuite, gym, etc.)

### Key Input/Output Interfaces
The PI0 model processes multi-modal inputs through standardized interfaces:

**Core Prediction Interface** (`pi0/modeling_pi0.py:55`):
```python
# Main inference method
action = policy.select_action(observation)

# Expected observation format:
observation = {
    "image": {
        "base_0_rgb": torch.tensor,      # (B, 3, 224, 224) uint8 [0, 255]
        "left_wrist_0_rgb": torch.tensor, # Optional additional camera views
    },
    "state": torch.tensor,               # (B, state_dim) float32 robot state
    "prompt": ["task description"],      # List of task instructions
}
```

**Data Processing Pipeline**:
- `prepare_images()` (line 127): Normalizes images to [-1, 1], handles BCHW format
- `prepare_state()` (line 216): Pads state vectors to maximum dimensions
- `prepare_language()` (line 240): Tokenizes prompts with PaliGemma format

**RIPT Integration Points**:
- `construct_pi0_observation()` in `pi0/ript/env/pi0_libero_runner.py:133`: Converts LIBERO environment outputs to PI0 format
- Action post-processing includes denormalization and state offset addition

## RIPT Training Architecture

### Core Flow
1. **Policy Wrapper** (`Pi0PolicyWrapper`): Provides error handling and action formatting
2. **Environment Runner** (`LIBEROEnvRunner`): Manages LIBERO environment interactions
3. **Rollout Generator**: Collects trajectories with intelligent sampling
4. **CFG Adapter** (`PI0_CFG_Adapter`): Converts episodes to PI0-compatible batches
5. **RL Optimizer** (`RLOptimizerPI0_CFG`): Implements advantage-weighted loss optimization

### Training Data Pipeline
```
LIBERO Environment → Trajectory Collection → Episode Processing → Batch Formation → CFG-weighted Loss → Policy Update
```

### Configuration System
Training uses YAML configuration files in `pi0/ript/config/`:
- `train_pi0_cfg_rl.yaml`: Full training configuration
- `debug_train_pi0.yaml`: Minimal configuration for testing
- Supports distributed training, mixed precision, gradient accumulation

## Checkpoint Structure
```
checkpoints/
├── pi0_libero_pytorch/          # Converted PyTorch model
│   ├── config.json              # Model configuration
│   ├── model.safetensors        # Model weights
│   └── tokenizer files...
└── original_jax_checkpoint/     # Keep for norm_stats.json
    ├── params/                  # JAX parameters
    └── norm_stats.json          # Critical normalization data
```

## Common Issues and Solutions

### Import Errors
If you encounter `ModuleNotFoundError: No module named 'lerobot.constants'`:
- Ensure `/zhaohan/ZJH/lerobot` is in Python path
- Check that compatibility layer files exist in `lerobot/lerobot/`

### RIPT Training Issues
If RIPT training fails:
- Verify all RIPT dependencies are installed (see RIPT_MIGRATION_PROGRESS.md)
- Check LIBERO environment setup and EGL/OpenGL configuration
- Ensure checkpoint paths in config files are correct
- For distributed training, verify NCCL configuration and GPU visibility
- **Stage Selection**: Use the appropriate stage script for your needs:
  - Stage 7-8 for development and debugging
  - Stage 9 for configuration-based training
  - Stage 10 for production distributed training

### Parallel Environment Memory Issues
**Critical Issue**: SubprocVectorEnv causes GPU memory overflow due to model serialization
- **Root Cause**: Each subprocess loads a complete 3.5GB PI0 model copy
- **Detection**: GPU memory usage shows ~3.5GB × num_parallel_envs
- **Solution**: Enhanced system now uses independent environment factory (`parallel_env_factory.py`) to avoid model serialization
- **Intelligent Fallback**: Automatic GPU memory analysis with fallback to batch processing when memory insufficient
- **Configuration**: Use feature flags `enable_parallel_envs` and `enable_true_parallel_envs` for fine control
- **Recommended**: Use Stage 11 training with enhanced parallel environment management

### FileGlobalCounter Issues
If distributed coordination fails:
- Ensure sufficient disk space for counter files in `./distributed_counters/`
- Check file permissions for counter file creation
- For NFS/shared storage, verify file locking support (`fcntl.flock`)
- Use `test_file_counter_integration.py` to validate setup

### Dataset Dimension Mismatch Errors
**Fixed Issue**: `ValueError: all the input array dimensions for the concatenation axis must match exactly`
- **Root Cause**: Different LIBERO tasks have different numbers of initial states
- **Solution**: Use `itertools.chain.from_iterable()` to flatten states instead of `np.concatenate()`
- **Location**: `pi0/ript/scripts/train_ript_pi0.py` LiberoInitStateDataset class

### Quality Differences Between RIPT and Reference
**Critical Discovery**: Identical environment states produce different action outputs with 5-11cm differences
- **Root Cause**: Model inference differences despite identical inputs
- **Status**: Under investigation - see RIPT_QUALITY_DEBUG_GUIDE.md for detailed analysis
- **Debug Tools**: Both implementations now have debug checkpoints for systematic comparison

### Image Processing Issues
**Ongoing Investigation**: Image coordinate system handling differences:
- **CleanDiffuser**: Uses `[::-1].transpose(2,0,1)` (vertical flip + HWC→CHW)
- **RIPT Current**: May have additional transformations affecting inference
- **Diagnostic Tools**: `comprehensive_image_analysis.py`, debug image saves in `pi0/ript/debug_images/`

### Image Feature Errors
If you see "All image features are missing from the batch":
- Verify `dataset_stats` includes image feature keys matching your observation format
- Ensure image tensors are properly formatted (3D with correct dimensions)

### Path Configuration
- JAX checkpoint paths must be preserved for `norm_stats.json` access
- PyTorch checkpoints require accompanying tokenizer files
- RIPT training scripts must be run from project root directory (`/zhaohan/ZJH/openpi_pytorch`)

### LIBERO Environment Setup
**CRITICAL**: RIPT training uses CleanDiffuser's LIBERO integration:
- **Training Environment**: `gym.make("libero-10-v0", task_id=9, seed=0, ...)` (now standardized)
- **Environment Warmup**: Use 20-step warmup with dummy actions after `env.reset()` for stability
- **Reset Method**: `obs = env.reset()` with standard gym interface

## Debug File Structure
```
ript/debug_analysis/session_YYYYMMDD_HHMMSS/
├── checkpoint1_env_info.json           # Environment configuration
├── checkpoint2_obs_step*.json           # Raw observation statistics  
├── checkpoint2_raw_*_image_step*.png    # Raw image data
├── checkpoint3_processed_*_step*.png    # Processed image data
└── checkpoint5_state_action.json       # Action inference data (when available)

debug_2test_*.json                       # Reference implementation debug data
debug_2test_images/                      # Reference implementation image debug
```

## Development Notes

### Current Implementation Status
The repository has reached production readiness with all 10 development stages completed:

**Core Features (✅ Complete)**:
- PI0/PI0-fast model implementations with JAX-to-PyTorch conversion
- RIPT RL training framework with CFG-style advantage weighting  
- Enhanced smart sampling with state history tracking
- YAML-based configuration management system
- Multi-GPU distributed training with PyTorch DDP
- Comprehensive debugging and diagnostic tools

**Quality Assurance Findings**:
- Environment data is identical between implementations
- Action outputs differ by 5-11cm despite identical inputs
- Issue isolated to model inference layer, not environment or RL training
- Systematic debugging infrastructure enables rapid problem diagnosis

### Production Training Recommendations
- **Development**: Use Stage 7-8 scripts for rapid iteration
- **Configuration**: Use Stage 9 for parameterized experiments  
- **Production**: Use Stage 10 for multi-GPU distributed training
- **Debugging**: Extensive `test_*.py` suite for systematic issue diagnosis
- **System Validation**: Always run `test_enhanced_features.py` before production deployments
- **Parallel Environments**: Use batch processing mode for memory-safe operation

### Configuration Recommendations
- **Simple Training**: Use `pi0/ript/config/single_gpu_test.yaml`
- **Feature Testing**: Use `pi0/ript/config/single_gpu_test_with_features.yaml`
- **Parallel Testing**: Use `pi0/ript/config/multi_env_test.yaml` (with automatic memory optimization)
- **RIPT-VLA Style**: Use `pi0/ript/config/stage11_ript_vla.yaml` (enhanced parallel environment support)
- **Production Distributed**: Use `pi0/ript/config/distributed_train_pi0.yaml`

See `DISTRIBUTED_TRAINING_GUIDE.md` for comprehensive distributed training documentation.

## Advanced Features and Testing

### Parallel Environment Management
The system includes sophisticated parallel environment handling that automatically optimizes for memory constraints:

**Intelligent Strategy Selection**:
```python
# Automatic GPU memory analysis and strategy selection
if required_memory_gb > available_memory_gb * 0.8:
    # Falls back to batch processing for memory safety
    return batch_parallel_strategy()
else:
    # Attempts lightweight parallel environments
    return lightweight_parallel_strategy()
```

**Configuration Controls**:
```yaml
features:
  enable_task_polling: true           # Dynamic task assignment
  enable_parallel_envs: true          # Parallel environment support
  enable_true_parallel_envs: true     # True multiprocess parallel environments
  enable_smart_sampling: true         # Intelligent state sampling
  use_parallel_init_state: false      # Whether to use set_init_state in parallel mode
  save_video: true                    # Enable video saving during training
```

### File-based Distributed Coordination
The `FileGlobalCounter` system provides robust multi-process coordination:
- **Atomic Operations**: Uses `fcntl.flock` for exclusive file access
- **Retry Mechanisms**: Automatic retry with exponential backoff
- **Cross-Platform**: Works on Unix/Linux systems with shared storage
- **Fault Tolerant**: Handles process crashes and resource cleanup

### Enhanced Testing Framework
The repository includes production-ready testing infrastructure:

**Comprehensive Validation** (`test_enhanced_features.py`):
- File counter functionality testing
- Environment runner validation
- Enhanced rollout generator testing
- Parallel environment feature validation
- Configuration system testing

**Quick System Check** (`quick_single_gpu_test.py`):
- Basic setup validation
- Model loading verification
- Environment creation testing
- Simple rollout execution
- Configurable test modes

**Usage Examples**:
```bash
# Full system validation (5 comprehensive tests)
python test_enhanced_features.py

# Quick validation with specific config
python quick_single_gpu_test.py --config_path pi0/ript/config/single_gpu_test_with_features.yaml

# Test parallel environment functionality
python quick_single_gpu_test.py --config_path pi0/ript/config/multi_env_test.yaml
```

### Memory Optimization Techniques
The system automatically handles common memory issues:

1. **SubprocVectorEnv Serialization Detection**: Automatically detects when parallel environments would cause memory overflow
2. **Independent Environment Factory**: Uses `parallel_env_factory.py` to avoid model serialization in subprocesses
3. **Intelligent Memory Analysis**: Real-time GPU memory monitoring with automatic fallback strategies
4. **Batch Processing Fallback**: Seamlessly switches to memory-efficient batch processing when needed
5. **Smart Memory Allocation**: Pre-flight memory checks prevent CUDA OOM errors
6. **Automatic Cleanup**: Intelligent garbage collection and resource management

### Configuration Feature Flags
Modern configuration system with granular feature control:

- `enable_task_polling`: Dynamic task assignment and load balancing
- `enable_parallel_envs`: Parallel environment processing with automatic optimization
- `enable_true_parallel_envs`: True multiprocess parallel environments using independent factory
- `enable_smart_sampling`: Intelligent state sampling based on success history
- `use_parallel_init_state`: Control whether to use set_init_state in parallel mode (default: false to avoid MuJoCo dimension issues)
- `save_video`: Enable video recording during training for debugging and analysis
- Backward compatibility with all existing configurations

### Stage 11 Enhanced Features
The latest stage includes several breakthrough improvements:

**Independent Environment Factory**: Solves the fundamental SubprocVectorEnv serialization problem by creating truly independent environment creation functions that don't reference the main process's policy object.

**Intelligent Memory Management**: Automatic GPU memory analysis that calculates required memory (3.5GB × num_parallel_envs) and compares with available memory, automatically falling back to safe batch processing when needed.

**Enhanced Observation Processing**: Unified observation handling that supports multiple VectorEnv formats with robust fallback mechanisms for compatibility across different implementations.

**Advanced Feature Toggle System**: Granular control over parallel environment features with intelligent defaults and production-ready safety mechanisms.




