"""
使用OpenPI-RIPT兼容数据包装器的训练示例
展示如何在保持OpenPI标准的同时支持RIPT功能
"""

import lightning as L
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from pi0.modeling_pi0 import PI0Policy
from lerobot.configs.policies import PreTrainedConfig
from utils.openpi_ript_dataset_wrapper import create_openpi_ript_dataset


def to_device_dtype(d, device, dtype):
    """将数据批次转移到指定设备和数据类型"""
    for key, value in d.items():
        if isinstance(value, dict):
            to_device_dtype(value, device, dtype)
        elif isinstance(value, torch.Tensor):
            if key not in ["action_is_pad"]:
                d[key] = value.to(device=device, dtype=dtype)
            else:
                d[key] = value.to(device=device)
        else:
            pass
    return d


class LightningTrainingWrapperRiptCompatible(L.LightningModule):
    """
    兼容RIPT的Lightning训练包装器
    """
    
    def __init__(self, config, ckpt_path, enable_ript_training=False):
        super().__init__()
        self.policy = None
        self.config = config
        self.ckpt_path = ckpt_path
        self.enable_ript_training = enable_ript_training
        
        print(f"🚀 初始化训练包装器")
        print(f"   - RIPT训练: {'启用' if enable_ript_training else '禁用'}")

    def configure_model(self):
        if self.policy is None:
            self.policy = PI0Policy.from_pretrained(self.ckpt_path, config=self.config)
            print(f"✅ 模型加载完成: {self.ckpt_path}")

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        """
        训练步骤，支持标准训练和RIPT训练
        """
        if self.enable_ript_training:
            # RIPT训练模式：使用优势加权
            loss = self._ript_training_step(batch, batch_idx)
        else:
            # 标准训练模式
            loss = self.policy(batch)[0]
            
        self.log("train_loss", loss, prog_bar=True)
        
        # 记录额外指标
        if self.enable_ript_training and "advantages" in batch:
            advantages = batch["advantages"]
            self.log("advantages/mean", advantages.mean(), prog_bar=False)
            self.log("advantages/std", advantages.std(), prog_bar=False)
            
        return loss
    
    def _ript_training_step(self, batch, batch_idx):
        """
        RIPT训练步骤，使用优势加权
        """
        # 获取基础损失
        base_loss = self.policy(batch)[0]
        
        # 如果有优势值，进行加权
        if "advantages" in batch:
            advantages = batch["advantages"]
            
            # 简单的优势加权（后续可以替换为更复杂的CFG逻辑）
            # 归一化优势值
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            adv_weights = torch.nn.functional.softplus(adv_norm)
            
            # 加权损失
            weighted_loss = (base_loss * adv_weights.mean()).mean()
            
            return weighted_loss
        else:
            return base_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy.get_optim_params(), 
            lr=5e-5, 
            weight_decay=1e-2, 
            eps=1e-6
        )
        scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def create_compatible_dataloader(
    repo_id: str = "ZibinDong/so100_grab_screwdriver",
    batch_size: int = 4,
    enable_ript: bool = False,
    **kwargs
):
    """
    创建兼容的数据加载器
    
    Args:
        repo_id: 数据集ID
        batch_size: 批次大小
        enable_ript: 是否启用RIPT功能
        **kwargs: 其他参数
    """
    print(f"🔄 创建数据加载器...")
    print(f"   - 数据集: {repo_id}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - RIPT模式: {'启用' if enable_ript else '禁用'}")
    
    # 创建兼容数据集
    dataset = create_openpi_ript_dataset(
        repo_id=repo_id,
        enable_ript=enable_ript,
        action_chunk_size=50,
        target_state_dim=9,  # 根据实际模型需求调整
        **kwargs
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        persistent_workers=True
    )
    
    print(f"✅ 数据加载器创建完成")
    return dataloader


def train_openpi_standard():
    """
    标准OpenPI训练（向后兼容）
    """
    print("🎯 开始标准OpenPI训练...")
    
    # 创建数据加载器（RIPT功能禁用）
    dataloader = create_compatible_dataloader(
        enable_ript=False,
        batch_size=4
    )
    
    # 创建检查点回调
    callback = ModelCheckpoint(
        dirpath="/tmp/pi0_standard_checkpoints",
        filename="{epoch}-{step}",
        save_top_k=-1,
        every_n_epochs=4,
    )
    
    # 创建训练器
    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,  # 单GPU用于测试
        max_epochs=2,  # 少量epoch用于测试
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        callbacks=[callback],
    )
    
    # 初始化模型
    with trainer.init_module():
        ckpt_path = "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
        config = PreTrainedConfig.from_pretrained(ckpt_path)
        config.device = "cpu"
        config.freeze_vision_encoder = True
        config.train_expert_only = True
        config.train_state_proj = True
        
        training_policy = LightningTrainingWrapperRiptCompatible(
            config, 
            ckpt_path, 
            enable_ript_training=False
        )
    
    # 开始训练
    trainer.fit(training_policy, dataloader)
    print("✅ 标准训练完成")


def train_openpi_ript_compatible():
    """
    RIPT兼容训练（展示新功能）
    """
    print("🎯 开始RIPT兼容训练...")
    
    # 创建数据加载器（RIPT功能启用）
    dataloader = create_compatible_dataloader(
        enable_ript=True,
        batch_size=4
    )
    
    # 创建检查点回调
    callback = ModelCheckpoint(
        dirpath="/tmp/pi0_ript_checkpoints",
        filename="{epoch}-{step}",
        save_top_k=-1,
        every_n_epochs=4,
    )
    
    # 创建训练器
    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,  # 单GPU用于测试
        max_epochs=2,  # 少量epoch用于测试
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        callbacks=[callback],
    )
    
    # 初始化模型
    with trainer.init_module():
        ckpt_path = "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
        config = PreTrainedConfig.from_pretrained(ckpt_path)
        config.device = "cpu"
        config.freeze_vision_encoder = True
        config.train_expert_only = True
        config.train_state_proj = True
        
        training_policy = LightningTrainingWrapperRiptCompatible(
            config, 
            ckpt_path, 
            enable_ript_training=True
        )
    
    # 开始训练
    trainer.fit(training_policy, dataloader)
    print("✅ RIPT兼容训练完成")


def test_data_compatibility():
    """
    测试数据兼容性
    """
    print("🧪 测试数据兼容性...")
    
    # 测试1：标准模式
    print("\n--- 测试标准模式 ---")
    dataset_standard = create_openpi_ript_dataset(
        repo_id="ZibinDong/so100_grab_screwdriver",
        enable_ript=False
    )
    sample_standard = dataset_standard[0]
    print(f"标准模式样本字段: {list(sample_standard.keys())}")
    
    # 测试2：RIPT模式
    print("\n--- 测试RIPT模式 ---")
    dataset_ript = create_openpi_ript_dataset(
        repo_id="ZibinDong/so100_grab_screwdriver",
        enable_ript=True
    )
    sample_ript = dataset_ript[0]
    print(f"RIPT模式样本字段: {list(sample_ript.keys())}")
    
    # 验证字段兼容性
    common_fields = ["image", "state", "action", "action_is_pad", "prompt"]
    for field in common_fields:
        assert field in sample_standard, f"标准模式缺少字段: {field}"
        assert field in sample_ript, f"RIPT模式缺少字段: {field}"
    
    # 验证RIPT特有字段
    ript_fields = ["advantages", "init_hash"]
    for field in ript_fields:
        assert field not in sample_standard, f"标准模式不应有RIPT字段: {field}"
        assert field in sample_ript, f"RIPT模式缺少字段: {field}"
    
    print("✅ 数据兼容性测试通过")


if __name__ == "__main__":
    # 测试数据兼容性
    test_data_compatibility()
    
    # 注释掉实际训练（需要GPU和检查点）
    # train_openpi_standard()
    # train_openpi_ript_compatible()
    
    print("🎉 示例代码运行完成")
