"""Regression test: SharedDiTWAM.infer_joint must feed the CLEAN video as the
DreamZero clean-context stream, not the noisy latents being denoised.

The shared-DiT core treats ``clean_video`` as a clean (zero-timestep) context
stream.  Training already passes the clean VAE encoding.  Inference used to pass
the *noisy* latents, a train/inference divergence that corrupts the clean
conditioning stream.  See the one-line fix in ``shared_dit_wam.py``.
"""

from types import SimpleNamespace

import torch

from streamingwam.wam.shared_dit_wam import SharedDiTWAM


class _TinySchedulerConfig:
    num_train_timesteps = 1000
    train_shift = 5.0
    infer_shift = 5.0


class _RecordingSharedDiT:
    """Fake shared-DiT core that records what clean/noisy video it receives."""

    num_layers = 2

    def __init__(self, backbone) -> None:
        self.backbone = backbone

    def __call__(
        self,
        noisy_video=None,
        video_timestep=None,
        context=None,
        noisy_action=None,
        action_timestep=None,
        state=None,
        context_mask=None,
        clean_video=None,
        state_is_pad=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.backbone.calls.append(
            {
                "clean_is_noisy": bool(torch.equal(clean_video, noisy_video)),
                "clean": clean_video.detach().clone(),
            }
        )
        video_pred = torch.zeros(noisy_video.shape)
        action_pred = torch.zeros(noisy_action.shape)
        return video_pred, action_pred


class _TinyBackbone:
    def __init__(self) -> None:
        self.info = SimpleNamespace(
            hidden_dim=8,
            num_heads=2,
            attn_head_dim=4,
            freq_dim=8,
            eps=1e-6,
            ffn_dim=32,
            text_dim=8,
            in_channels=3,
            patch_size=(1, 2, 2),
        )
        self.calls = []

    def build_shared_dit_core(self, config, **kwargs):
        return _RecordingSharedDiT(self)

    def encode_video(self, video) -> torch.Tensor:
        return torch.full((video.shape[0], 3, video.shape[2], 2, 2), 0.5)

    def decode_latents(self, latents) -> torch.Tensor:
        return latents


def _make_model() -> SharedDiTWAM:
    model = object.__new__(SharedDiTWAM)
    model.chunk_size = 32
    model.action_dim = 7
    model.state_dim = 8
    model.action_tokens_per_state = 4
    model.config = SimpleNamespace(
        chunk_size=32,
        action_dim=7,
        proprio_dim=8,
        max_state_dim=8,
        shared_dit_clean_context="full_video",
        shared_dit_pin_first_latent_step=True,
        shared_dit_checkpoint_blocks=False,
        num_frame_per_block=2,
        num_action_per_block=32,
        num_state_per_block=1,
        video_scheduler=_TinySchedulerConfig(),
        action_scheduler=_TinySchedulerConfig(),
    )
    model.backbone = _TinyBackbone()
    model.shared_dit = model.backbone.build_shared_dit_core(model.config)
    return model


def test_shared_dit_infer_joint_feeds_clean_video_as_clean_context() -> None:
    model = _make_model()
    input_image = torch.randn(1, 3, 8, 8)
    context = torch.randn(1, 4, 8)
    context_mask = torch.ones(1, 4, dtype=torch.bool)
    state = torch.randn(1, 8)

    model.infer_joint(
        input_image=input_image,
        context=context,
        context_mask=context_mask,
        num_video_frames=9,
        action_horizon=32,
        num_inference_steps=2,
        action_num_inference_steps=2,
        proprio=state,
    )

    assert model.backbone.calls, "shared-DiT core was never invoked"
    assert all(not call["clean_is_noisy"] for call in model.backbone.calls)
