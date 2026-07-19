import math
import os

import torch
import torch.nn.functional as F

try:
    import timm
except Exception:  # timm is optional for non-timm backbones
    timm = None

try:
    from diffusers import StableDiffusionPipeline
except Exception:  # diffusers is optional for non-SD backbones
    StableDiffusionPipeline = None

from peca.utils import segment_pooling


def _ensure_three_channels(images):
    if images is None:
        return images
    if images.dim() == 5:
        c = images.shape[2]
        if c == 3:
            return images
        if c > 3:
            return images[:, :, :3]
        repeat = 3 // c + (1 if 3 % c else 0)
        return images.repeat(1, 1, repeat, 1, 1)[:, :, :3]
    if images.dim() == 4:
        c = images.shape[1]
        if c == 3:
            return images
        if c > 3:
            return images[:, :3]
        repeat = 3 // c + (1 if 3 % c else 0)
        return images.repeat(1, repeat, 1, 1)[:, :3]
    if images.dim() == 3:
        c = images.shape[0]
        if c == 3:
            return images
        if c > 3:
            return images[:3]
        repeat = 3 // c + (1 if 3 % c else 0)
        return images.repeat(repeat, 1, 1)[:3]
    return images


def _unwrap_timm_feats(feats):
    if isinstance(feats, dict):
        for key in ("x", "feat", "features", "last"):
            if key in feats:
                return feats[key]
        if feats:
            return next(iter(feats.values()))
    if isinstance(feats, (list, tuple)):
        for item in reversed(feats):
            if hasattr(item, "dim") and item.dim() in (3, 4):
                return item
        if feats:
            return feats[-1]
    return feats


class BackboneOnlyModel:
    def __init__(
        self,
        dino_model_type,
        dino_input_size,
        segment_pool_size,
        device,
        load_dino,
        dino_repository="facebookresearch/dinov2:main",
        timm_model_type=None,
        timm_input_size=None,
        timm_pretrained=True,
        load_timm=False,
        sam2_model_id=None,
        sam2_input_size=None,
        sam2_amp_dtype=None,
        load_sam2=False,
        sd_model_id=None,
        sd_input_size=None,
        sd_prompt=None,
        sd_timestep=None,
        sd_timestep_ratio=None,
        sd_up_block_index=0,
        sd_precision="fp16",
        load_sd=False,
    ):
        self.device = device
        self.segment_pool_size = segment_pool_size
        self.dino_input_size = dino_input_size
        self.dino = None
        self.dino_dim = None
        self.dino_repository = dino_repository
        self.timm = None
        self.timm_input_size = timm_input_size or dino_input_size
        self.timm_model_type = timm_model_type
        self.timm_pretrained = timm_pretrained
        self.sam2_predictor = None
        self.sam2_input_size = sam2_input_size or [1024, 1024]
        self.sam2_model_id = sam2_model_id
        self.sam2_amp_dtype = sam2_amp_dtype
        self.sd_pipe = None
        self.sd_unet = None
        self.sd_vae = None
        self.sd_scheduler = None
        self.sd_tokenizer = None
        self.sd_text_encoder = None
        self.sd_prompt_embeds = None
        self.sd_model_id = sd_model_id or "sd2-community/stable-diffusion-2-1"
        self.sd_input_size = sd_input_size or [768, 768]
        self.sd_prompt = sd_prompt or "a photo of an anime character."
        self.sd_timestep = sd_timestep
        self.sd_timestep_ratio = sd_timestep_ratio
        self.sd_up_block_index = int(sd_up_block_index)
        self.sd_precision = sd_precision
        if load_dino:
            self._load_dino(dino_model_type)
        if load_timm:
            self._load_timm(timm_model_type, timm_pretrained)
        if load_sam2:
            self._load_sam2()
        if load_sd:
            self._load_sd()

    def _load_dino(self, model_type):
        if self.dino is not None:
            return
        print(f"Loading DINOv2 model '{model_type}' from PyTorch Hub...")
        self.dino = torch.hub.load(
            self.dino_repository,
            model_type,
            trust_repo=True,
        )
        self.dino.eval().to(self.device)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino_dim = getattr(self.dino, "embed_dim", None)
        print(f"DINOv2 model '{model_type}' loaded successfully.")

    def _load_timm(self, model_type, pretrained=True):
        if self.timm is not None:
            return
        if timm is None:
            raise RuntimeError("timm is not available; install timm to use timm backbones.")
        if not model_type:
            raise ValueError("timm_model_type is required for timm backbones.")
        print(f"Loading timm model '{model_type}' (pretrained={pretrained})...")
        self.timm = timm.create_model(model_type, pretrained=pretrained)
        self.timm.eval().to(self.device)
        for param in self.timm.parameters():
            param.requires_grad = False
        print(f"timm model '{model_type}' loaded successfully.")

    def _resolve_sam2_amp_dtype(self):
        if self.sam2_amp_dtype:
            key = str(self.sam2_amp_dtype).strip().lower()
            if key in ("bf16", "bfloat16"):
                return torch.bfloat16
            if key in ("fp16", "float16", "half"):
                return torch.float16
            if key in ("fp32", "float32"):
                return torch.float32
        if self.device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _load_sam2(self):
        if self.sam2_predictor is not None:
            return
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(
                "sam2 is not available; install SAM2 from https://github.com/facebookresearch/sam2 "
                "or via pip, and ensure dependencies are available."
            ) from exc
        if not self.sam2_model_id:
            raise ValueError("sam2_model_id is required for the SAM2 backbone.")
        print(f"Loading SAM2 from Hugging Face '{self.sam2_model_id}'...")
        predictor = SAM2ImagePredictor.from_pretrained(self.sam2_model_id)
        predictor.model.eval().to(self.device)
        for param in predictor.model.parameters():
            param.requires_grad = False
        self.sam2_predictor = predictor
        print("SAM2 model loaded successfully.")

    def _resolve_sd_dtype(self):
        key = str(self.sd_precision or "fp16").strip().lower()
        if key in ("bf16", "bfloat16"):
            return torch.bfloat16
        if key in ("fp32", "float32"):
            return torch.float32
        if key in ("fp16", "float16", "half"):
            if self.device.type == "cuda":
                return torch.float16
            return torch.float32
        if key == "auto":
            if self.device.type == "cuda":
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
        return torch.float32

    def _encode_sd_prompt(self):
        if self.sd_tokenizer is None or self.sd_text_encoder is None:
            raise RuntimeError("Stable Diffusion tokenizer/text encoder is not initialized.")
        text_inputs = self.sd_tokenizer(
            [self.sd_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.sd_tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = getattr(text_inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            if attention_mask is not None:
                prompt_embeds = self.sd_text_encoder(input_ids, attention_mask=attention_mask)[0]
            else:
                prompt_embeds = self.sd_text_encoder(input_ids)[0]
        self.sd_prompt_embeds = prompt_embeds

    def _load_sd(self):
        if self.sd_pipe is not None:
            return
        if StableDiffusionPipeline is None:
            raise RuntimeError(
                "diffusers is not available; install diffusers to use Stable Diffusion features."
            )
        dtype = self._resolve_sd_dtype()
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        pipe = None
        load_errors = []
        for model_id in [self.sd_model_id]:
            print(f"Loading Stable Diffusion '{model_id}' (dtype={dtype})...")
            kwargs = {
                "torch_dtype": dtype,
                "safety_checker": None,
            }
            if token:
                kwargs["token"] = token
            try:
                kwargs["requires_safety_checker"] = False
                pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
                self.sd_model_id = model_id
                break
            except TypeError:
                kwargs.pop("requires_safety_checker", None)
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
                    self.sd_model_id = model_id
                    break
                except Exception as exc:  # noqa: PERF203
                    load_errors.append((model_id, exc))
                    if "No module named 'kornia_rs'" in str(exc):
                        break
            except Exception as exc:  # noqa: PERF203
                load_errors.append((model_id, exc))
                if "No module named 'kornia_rs'" in str(exc):
                    break

        if pipe is None:
            msg = "; ".join([f"{mid}: {err}" for mid, err in load_errors]) or "unknown error"
            dep_hint = ""
            if any("No module named 'kornia_rs'" in str(err) for _, err in load_errors):
                dep_hint = " Install missing dependency: `pip install kornia-rs`."
            raise RuntimeError(
                "Failed to load Stable Diffusion model. "
                f"Tried '{self.sd_model_id}'. Last errors: {msg}. "
                "If using StabilityAI official repo, run `huggingface-cli login` (or set HF_TOKEN) "
                f"and accept model terms on the model page.{dep_hint}"
            )
        pipe = pipe.to(self.device)
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        for module in (pipe.unet, pipe.vae, pipe.text_encoder):
            for param in module.parameters():
                param.requires_grad = False
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        self.sd_pipe = pipe
        self.sd_unet = pipe.unet
        self.sd_vae = pipe.vae
        self.sd_scheduler = pipe.scheduler
        self.sd_tokenizer = pipe.tokenizer
        self.sd_text_encoder = pipe.text_encoder
        self._encode_sd_prompt()
        print(f"Stable Diffusion '{self.sd_model_id}' loaded successfully.")

    def _resolve_sd_timestep(self):
        if self.sd_scheduler is None:
            return 261
        max_step = int(getattr(getattr(self.sd_scheduler, "config", None), "num_train_timesteps", 1000)) - 1
        if max_step < 1:
            max_step = 999
        if self.sd_timestep is not None:
            t = int(self.sd_timestep)
        elif self.sd_timestep_ratio is not None:
            t = int(round(float(self.sd_timestep_ratio) * float(max_step)))
        else:
            t = 261
        return max(0, min(max_step, t))

    def eval(self):
        return self

    def get_segment_feats(self, feats_map, seg_image, seg_num):
        return segment_pooling(feats_map, seg_image, seg_num, self.segment_pool_size)

    def get_seg_cos_sim(self, seg_feats_src, seg_feats_tgt):
        seg_feats_src = F.normalize(seg_feats_src, p=2, dim=-1)
        seg_feats_tgt = F.normalize(seg_feats_tgt, p=2, dim=-1)

        B, S_src, L_src, C = seg_feats_src.shape
        seg_feats_src = seg_feats_src.view(B, S_src * L_src, C)

        B, S_tgt, L_tgt, C = seg_feats_tgt.shape
        seg_feats_tgt = seg_feats_tgt.view(B, S_tgt * L_tgt, C)

        return torch.matmul(seg_feats_tgt, seg_feats_src.transpose(-1, -2))

    def get_dino_feats_map(self, images):
        if self.dino is None:
            raise RuntimeError("DINO backbone is not initialized.")
        images = _ensure_three_channels(images)
        B, S, C, H, W = images.shape
        flat = images.view(B * S, C, H, W)
        flat = F.interpolate(flat, size=self.dino_input_size, mode="bilinear", align_corners=False)
        dino_output = self.dino.get_intermediate_layers(flat, n=1, return_class_token=False)
        patch_tokens = dino_output[0]
        feat_h = self.dino_input_size[0] // 14
        feat_w = self.dino_input_size[1] // 14
        return patch_tokens.permute(0, 2, 1).view(B, S, self.dino_dim, feat_h, feat_w)

    def _infer_timm_grid(self, tokens, input_size):
        if tokens.dim() != 3:
            return None, None, False
        n_tokens = tokens.shape[1]
        patch_embed = getattr(self.timm, "patch_embed", None)
        if patch_embed is not None:
            grid = getattr(patch_embed, "grid_size", None)
            if grid is not None and len(grid) >= 2:
                grid_h, grid_w = int(grid[0]), int(grid[1])
                if n_tokens == grid_h * grid_w + 1:
                    return grid_h, grid_w, True
                if n_tokens == grid_h * grid_w:
                    return grid_h, grid_w, False
            patch_size = getattr(patch_embed, "patch_size", None)
            if patch_size is not None:
                if isinstance(patch_size, (list, tuple)):
                    ph, pw = int(patch_size[0]), int(patch_size[1])
                else:
                    ph = pw = int(patch_size)
                grid_h = int(input_size[0] // ph)
                grid_w = int(input_size[1] // pw)
                if n_tokens == grid_h * grid_w + 1:
                    return grid_h, grid_w, True
                if n_tokens == grid_h * grid_w:
                    return grid_h, grid_w, False
        # fallback: infer from perfect square
        if n_tokens > 1:
            sq = int(math.isqrt(n_tokens - 1))
            if sq * sq == n_tokens - 1:
                return sq, sq, True
        sq = int(math.isqrt(n_tokens))
        if sq * sq == n_tokens:
            return sq, sq, False
        return None, None, False

    def get_timm_feats_map(self, images):
        if self.timm is None:
            raise RuntimeError("timm backbone is not initialized.")
        images = _ensure_three_channels(images)
        B, S, C, H, W = images.shape
        flat = images.view(B * S, C, H, W)
        flat = F.interpolate(flat, size=self.timm_input_size, mode="bilinear", align_corners=False)
        if hasattr(self.timm, "forward_features"):
            feats = self.timm.forward_features(flat)
        else:
            feats = self.timm(flat)
        feats = _unwrap_timm_feats(feats)
        if not hasattr(feats, "dim"):
            raise RuntimeError("timm backbone did not return a tensor feature map.")
        if feats.dim() == 4:
            _, C_feat, H_feat, W_feat = feats.shape
            return feats.view(B, S, C_feat, H_feat, W_feat)
        if feats.dim() == 3:
            grid_h, grid_w, has_cls = self._infer_timm_grid(feats, self.timm_input_size)
            if grid_h is None or grid_w is None:
                raise RuntimeError("Failed to infer token grid size for timm backbone.")
            if has_cls:
                feats = feats[:, 1:]
            feats = feats.permute(0, 2, 1).contiguous()
            return feats.view(B, S, feats.shape[1], grid_h, grid_w)
        raise RuntimeError(f"Unsupported timm feature shape: {tuple(feats.shape)}")

    def _extract_sam2_embedding(self):
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 predictor is not initialized.")
        predictor = self.sam2_predictor
        if hasattr(predictor, "get_image_embedding"):
            emb = predictor.get_image_embedding()
        else:
            features = getattr(predictor, "_features", None)
            emb = None
            if isinstance(features, dict):
                emb = features.get("image_embed") or features.get("image_embedding")
            if emb is None:
                raise RuntimeError("SAM2 predictor did not expose image embeddings.")
        if hasattr(emb, "dim") and emb.dim() == 4 and emb.shape[0] == 1:
            emb = emb[0]
        return emb

    def get_sam2_feats_map(self, images):
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 backbone is not initialized.")
        images = _ensure_three_channels(images)
        B, S, C, H, W = images.shape
        flat = images.view(B * S, C, H, W)
        feats_list = []
        dtype = self._resolve_sam2_amp_dtype()
        use_amp = self.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        for idx in range(flat.shape[0]):
            img = flat[idx : idx + 1]
            img = F.interpolate(img, size=self.sam2_input_size, mode="bilinear", align_corners=False)
            img = img.clamp(0, 1).mul(255).byte().squeeze(0)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast("cuda", dtype=dtype):
                        self.sam2_predictor.set_image(img_np)
                        emb = self._extract_sam2_embedding()
                else:
                    self.sam2_predictor.set_image(img_np)
                    emb = self._extract_sam2_embedding()
            if not hasattr(emb, "dim"):
                raise RuntimeError("SAM2 embedding is not a tensor.")
            if emb.dim() == 4 and emb.shape[0] == 1:
                emb = emb[0]
            if emb.dim() != 3:
                raise RuntimeError(f"Unsupported SAM2 embedding shape: {tuple(emb.shape)}")
            feats_list.append(emb.to(self.device, dtype=torch.float32))
        feats = torch.stack(feats_list, dim=0)
        return feats.view(B, S, feats.shape[1], feats.shape[2], feats.shape[3])

    def get_sd_feats_map(self, images):
        if self.sd_pipe is None:
            raise RuntimeError("Stable Diffusion backbone is not initialized.")
        images = _ensure_three_channels(images)
        B, S, C, H, W = images.shape
        flat = images.view(B * S, C, H, W)
        flat = F.interpolate(flat, size=self.sd_input_size, mode="bilinear", align_corners=False)
        flat = (flat * 2.0 - 1.0).clamp(-1.0, 1.0)

        timestep = self._resolve_sd_timestep()
        t = torch.full((flat.shape[0],), int(timestep), dtype=torch.long, device=self.device)
        prompt_embeds = self.sd_prompt_embeds.to(self.device)
        if prompt_embeds.shape[0] != flat.shape[0]:
            prompt_embeds = prompt_embeds.expand(flat.shape[0], -1, -1)

        captured = []

        def _hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(out, torch.Tensor):
                captured.append(out)

        if self.sd_up_block_index < 0 or self.sd_up_block_index >= len(self.sd_unet.up_blocks):
            raise ValueError(
                f"sd_up_block_index={self.sd_up_block_index} out of range for {len(self.sd_unet.up_blocks)} up blocks."
            )
        hook_handle = self.sd_unet.up_blocks[self.sd_up_block_index].register_forward_hook(_hook)

        sd_dtype = self._resolve_sd_dtype()
        use_amp = self.device.type == "cuda" and sd_dtype in (torch.float16, torch.bfloat16)
        try:
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast("cuda", dtype=sd_dtype):
                        latents = self.sd_vae.encode(flat).latent_dist.sample()
                        scale = float(getattr(self.sd_vae.config, "scaling_factor", 0.18215))
                        latents = latents * scale
                        noise = torch.randn_like(latents)
                        noisy_latents = self.sd_scheduler.add_noise(latents, noise, t)
                        _ = self.sd_unet(
                            noisy_latents,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )
                else:
                    latents = self.sd_vae.encode(flat).latent_dist.sample()
                    scale = float(getattr(self.sd_vae.config, "scaling_factor", 0.18215))
                    latents = latents * scale
                    noise = torch.randn_like(latents)
                    noisy_latents = self.sd_scheduler.add_noise(latents, noise, t)
                    _ = self.sd_unet(
                        noisy_latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )
        finally:
            hook_handle.remove()

        if not captured:
            raise RuntimeError("Failed to capture Stable Diffusion upsampling block features.")
        feats = captured[-1].float()
        if feats.dim() != 4:
            raise RuntimeError(f"Unsupported Stable Diffusion feature shape: {tuple(feats.shape)}")
        return feats.view(B, S, feats.shape[1], feats.shape[2], feats.shape[3])
