import numpy as np
import pytorch_lightning as L
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from termcolor import cprint
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize

from pi0.modeling_pi0 import PI0Policy
from utils.normalizers import Normalizer
from utils.server import PolicyServer


def to_device_dtype(d, device, dtype):
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


class LightningTrainingWrapper(L.LightningModule):
    def __init__(self, config, ckpt_path):
        super().__init__()
        # load model in `configure_model` to accelerate model loading
        self.policy = None
        self.config = config
        self.ckpt_path = ckpt_path

    def configure_model(self):
        if self.policy is None:
            self.policy = PI0Policy.from_pretrained(self.ckpt_path, config=self.config)

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss


class SO100Policy:
    def __init__(
        self,
        ckpt_path: str,
        pi0_ckpt_path: str,
        repo_id: str = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        # load policy
        cprint("Loading SO100 Policy...", "yellow")
        config = PreTrainedConfig.from_pretrained(pi0_ckpt_path)
        training_policy = LightningTrainingWrapper(config, pi0_ckpt_path)
        training_policy.load_state_dict(
            torch.load(ckpt_path, map_location="cpu")["state_dict"]
        )
        self.policy = policy.to(device=device, dtype=dtype).eval()
        cprint("SO100 Policy loaded successfully!", "green")

        cprint("Prepareing norm stats...", "yellow")
        dataset = LeRobotDataset(repo_id=repo_id)
        self.normalizer = Normalizer(
            norm_stats=dataset.meta.stats,
            norm_type={
                "observation.images.base": "identity",
                "observation.images.wrist": "identity",
                "observation.state": "meanstd",
                "action": "std",
            },
        )
        cprint("Norm stats prepared successfully!", "green")

        self.resize = Resize((224, 224))

        cprint("Ready to use SO100 Policy!", "green")

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        """
        obs: {
            "base": uint8 (H, W, C),
            "wrist": uint8 (H, W, C),
            "state": float32 (state_dim,),
            "prompt": str
        }
        """
        obs = self.normalizer.normalize(
            {
                "observation.images.base": obs["base"],
                "observation.images.wrist": obs["wrist"],
                "observation.state": obs["state"],
                "prompt": obs["prompt"],
            }
        )

        base_image = torch.tensor(
            obs["observation.images.base"], dtype=torch.uint8, device=self.device
        )
        wrist_image = torch.tensor(
            obs["observation.images.wrist"], dtype=torch.uint8, device=self.device
        )
        base_image = base_image.permute(2, 0, 1)[None]
        wrist_image = wrist_image.permute(2, 0, 1)[None]
        base_image = self.resize(base_image)
        wrist_image = self.resize(wrist_image)
        state = torch.tensor(
            obs["observation.state"], dtype=self.dtype, device=self.device
        )[None]
        prompt = obs["prompt"]
        action = self.policy.select_action(
            {
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                },
                "state": state,
                "prompt": prompt,
            }
        )
        action = action[:, :, :6]
        action = action.float().cpu().numpy()
        state = state.float().cpu().numpy()
        state_action = self.normalizer.unnormalize(
            {"observation.state": state, "action": action}
        )
        state = state_action["observation.state"]
        action = state_action["action"]
        action = action + state
        return action

    def __call__(self, obs: np.ndarray):
        return self.act(obs)


if __name__ == "__main__":
    policy = SO100Policy(
        ckpt_path="/mnt/20T/dzb/pi0_so100_checkpoints/epoch=39-step=29760.ckpt",
        pi0_ckpt_path="/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch",
        repo_ids="ZibinDong/so100_grab_screwdriver",
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    server = PolicyServer(policy, host="0.0.0.0", port=12346)
    server.run()
