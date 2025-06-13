import torch
from toolkit.pixel_shuffle_encoder import AutoencoderPixelMixer


def apply_flex2_patch(sd, downscale_factor=16):
    """Adjust Flex2 models for mismatched x_embedder input dimensions."""
    if not hasattr(sd, "unet") or not hasattr(sd.unet, "x_embedder"):
        return

    latent_channels = getattr(sd.vae.config, "latent_channels", None)
    if latent_channels is None:
        return

    expected_in = latent_channels * 4
    current_in = sd.unet.x_embedder.in_features
    if expected_in == current_in:
        return

    # Replace VAE with pixel mixer so latents match transformer expectations
    sd.vae = AutoencoderPixelMixer(in_channels=3, downscale_factor=downscale_factor)
    if hasattr(sd, "pipeline") and hasattr(sd.pipeline, "vae"):
        sd.pipeline.vae = sd.vae

    latent_channels = sd.vae.config.latent_channels
    new_in = latent_channels * 4

    sd.unet.x_embedder = torch.nn.Linear(new_in, sd.unet.x_embedder.out_features, bias=True)
    sd.unet.proj_out = torch.nn.Linear(sd.unet.proj_out.in_features, new_in, bias=True)
