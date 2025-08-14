# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a comprehensive Vision-Language-Action (VLA) research repository containing multiple implementations and frameworks for robotics AI:

- **OpenPI PyTorch** (`openpi_pytipt/`): Main simplified PyTorch implementation of PI0/PI0-fast models with RIPT RL framework
- **RIPT-PI0** (`ript-pi0/`): RIPT framework integration with PI0 models  
- **RIPT-VLA Original** (`ript-vla_ori/`): Original RIPT-VLA implementation for comparison
- **QueST** (`QueST/`): QueST algorithm implementation for robotics
- **CFG-RL** (`cfgrl/`): Classifier-Free Guidance for Reinforcement Learning
- **LeRobot** (`lerobot/`): Extended LeRobot framework
- **ConRFT** (`conrft/`): Contact-Aware Robot Fine-Tuning framework

## Main Working Directory

**Primary development focus**: `openpi_pytipt/` - This contains the most active and complete implementation with production-ready features including CFG integration, distributed training, and comprehensive testing infrastructure.

**Important**: The `openpi_pytipt/` directory contains its own detailed CLAUDE.md file with extensive technical documentation. Always reference that file for specific implementation details, debugging guides, and production deployment instructions.

## Quick Start Commands

### Working with OpenPI PyTorch (Primary)
```bash
cd openpi_pytipt
pip install -r requirements.txt

# Basic model conversion and inference
python convert_pi0_to_hf_lerobot.py --checkpoint_dir /path/to/jax/params --output_path /path/to/pytorch
python 1_e2e_inference.py

# System validation (recommended first step)
python test_enhanced_features.py
python quick_single_gpu_test.py

# RIPT RL training (staged approach)
python 11_train_ript_vla_style.py --config_path pi0/ript/config/stage11_ript_vla.yaml
```

### Working with RIPT-VLA Original
```bash
cd ript-vla_ori
python train_ript.py config/task/libero_goal.yaml
```

### Working with QueST
```bash
cd QueST
python train.py config/train_base.yaml task=libero_base algo=quest
```

### LIBERO Evaluation
```bash
cd openpi_pytipt
./run_libero10_eval.sh 0 20  # GPU 0, 20 rollouts per task
```

## Repository Architecture

### OpenPI PyTorch (`openpi_pytipt/`) - Primary Implementation
- **12-Stage Progressive Development**: From basic inference to production distributed training
- **CFG Integration**: Classifier-Free Guidance for improved model quality  
- **RIPT RL Framework**: Complete reinforcement learning training system
- **Production Features**: Distributed training, parallel environments, comprehensive testing
- **Advanced Memory Management**: Intelligent parallel environment handling with automatic fallbacks

### Other Frameworks
- **RIPT-PI0** (`ript-pi0/`): Alternative RIPT integration with PI0 models
- **RIPT-VLA Original** (`ript-vla_ori/`): Reference implementation for comparison studies
- **QueST** (`QueST/`): QueST algorithm with autoencoder training stages
- **CFG-RL** (`cfgrl/`): Standalone CFG-RL implementation with PI0 integration
- **LeRobot** (`lerobot/`): Extended framework with additional robot support
- **ConRFT** (`conrft/`): Contact-aware robot fine-tuning framework

## Key Technical Features

### Model Conversion System
- **JAX-to-PyTorch**: Converts OpenPI checkpoints while preserving critical normalization statistics
- **Essential Files**: Always preserve `norm_stats.json` from original JAX checkpoints
- **HuggingFace Integration**: Compatible with LeRobot policy loading system

### RIPT Training Pipeline
```
LIBERO Environment → Trajectory Collection → Episode Processing → 
CFG Batch Formation → Advantage-Weighted Loss → Policy Update
```

### Configuration Management
- **YAML-based**: Comprehensive configuration system with feature toggles
- **Staged Configs**: Different configurations for development, testing, and production
- **Parallel Environment Control**: Intelligent memory management with automatic optimization

## Critical Development Notes

### Memory Management
- **SubprocVectorEnv Issue**: Can cause GPU memory overflow (3.5GB × num_parallel_envs)
- **Solution**: Enhanced system uses independent environment factory with automatic fallback
- **Intelligent Detection**: Real-time GPU memory analysis with fallback strategies

### Import Path Compatibility  
The repository maintains compatibility across multiple frameworks:
- LeRobot imports: `/zhaohan/ZJH/lerobot/lerobot/constants.py`
- RIPT imports: Use `pi0.ript.*` path structure  
- Local constants: `lerobot_constants.py` for standalone operation

### Testing Strategy
- **System Validation**: `test_enhanced_features.py` for comprehensive testing
- **Quick Checks**: `quick_single_gpu_test.py` for rapid validation
- **Component Testing**: Individual test files for specific functionality
- **Debug Infrastructure**: Extensive debugging tools with image analysis capabilities

## Development Workflow

### For OpenPI PyTorch Development (Primary)
1. **System Validation**: Run `test_enhanced_features.py` to verify system health
2. **Model Setup**: Convert JAX checkpoints while preserving `norm_stats.json`
3. **Configuration Selection**: Choose appropriate YAML config for your use case
4. **Progressive Training**: Start with single-GPU debug configs, scale to distributed as needed
5. **Quality Assurance**: Use extensive debug infrastructure when issues arise

### Common Development Tasks
- **Model Conversion**: `python convert_pi0_to_hf_lerobot.py` with proper paths
- **Quick Testing**: `python quick_single_gpu_test.py` for rapid validation
- **Full Training**: Use staged approach (Stage 7 → 8 → 9 → 10 → 11)
- **Distributed Training**: Set `CUDA_VISIBLE_DEVICES` and use appropriate configs
- **Debugging**: Leverage comprehensive debug tools and image analysis

## Repository Status and Recommendations

### Production Ready Components
- **OpenPI PyTorch**: ✅ Complete with CFG integration and distributed training
- **Model Conversion**: ✅ Reliable JAX-to-PyTorch conversion with statistics preservation
- **RIPT Framework**: ✅ Production-ready RL training with intelligent memory management
- **Testing Infrastructure**: ✅ Comprehensive validation and debugging tools

### Development Focus
- **Primary**: Use `openpi_pytipt/` for all new development work
- **Reference**: Use `ript-vla_ori/` for comparison studies and validation
- **Experimentation**: Other frameworks available for specific research needs

**For detailed technical documentation, always reference `openpi_pytipt/CLAUDE.md` which contains comprehensive implementation details, debugging guides, and production deployment instructions.**