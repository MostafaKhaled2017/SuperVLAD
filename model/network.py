
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel, DeiTModel
#from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.normalization import L2Norm
import model.supervlad_layer as supervlad_layer

from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
import torch.nn.functional as F
import math
from model.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}

class SuperVLADModel(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        super().__init__()
        self.backbone = get_backbone(args, pretrained_foundation, foundation_model_path)
        self.arch_name = args.backbone
        self.crossimage_encoder = args.crossimage_encoder
        self.aggregation = supervlad_layer.SuperVLAD(clusters_num=args.supervlad_clusters, ghost_clusters_num=args.ghost_clusters, dim=args.features_dim,
                                work_with_tokens=args.work_with_tokens)

        if self.crossimage_encoder:
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Cross-image encoder
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim

    @staticmethod
    def _build_token_grid(height, width, device):
        rows = torch.arange(height, device=device)
        cols = torch.arange(width, device=device)
        grid_y, grid_x = torch.meshgrid(rows, cols)
        return torch.stack((grid_y, grid_x), dim=-1).view(-1, 2)

    @staticmethod
    def _make_random_token_mask(mask, keep_count, token_count, sample_seed):
        generator = None
        if sample_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
        kept_tokens = torch.randperm(token_count, generator=generator)[:keep_count]
        mask[kept_tokens.to(mask.device)] = 1.0
        return mask

    @staticmethod
    def _make_center_keep_mask(mask, keep_count, height, width):
        if keep_count >= height * width:
            mask.fill_(1.0)
            return mask
        center_h = min(height, max(1, int(round(math.sqrt(keep_count * height / width)))))
        center_w = min(width, max(1, int(math.ceil(keep_count / center_h))))
        center_h = min(height, max(1, center_h))
        center_w = min(width, max(1, center_w))
        while center_h * center_w < keep_count:
            if center_w < width:
                center_w += 1
            elif center_h < height:
                center_h += 1
            else:
                break
        start_h = max(0, (height - center_h) // 2)
        start_w = max(0, (width - center_w) // 2)
        mask_2d = mask.view(height, width)
        mask_2d[start_h:start_h + center_h, start_w:start_w + center_w] = 1.0
        return mask

    @staticmethod
    def _make_block_keep_mask(mask, keep_count, height, width, sample_seed):
        if keep_count >= height * width:
            mask.fill_(1.0)
            return mask
        block_h = min(height, max(1, int(round(math.sqrt(keep_count * height / width)))))
        block_w = min(width, max(1, int(math.ceil(keep_count / block_h))))
        block_h = min(height, max(1, block_h))
        block_w = min(width, max(1, block_w))
        while block_h * block_w < keep_count:
            if block_w < width:
                block_w += 1
            elif block_h < height:
                block_h += 1
            else:
                break
        max_h = max(0, height - block_h)
        max_w = max(0, width - block_w)
        generator = None
        if sample_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
        start_h = 0 if max_h == 0 else int(torch.randint(max_h + 1, (1,), generator=generator).item())
        start_w = 0 if max_w == 0 else int(torch.randint(max_w + 1, (1,), generator=generator).item())
        mask_2d = mask.view(height, width)
        mask_2d[start_h:start_h + block_h, start_w:start_w + block_w] = 1.0
        return mask

    @staticmethod
    def _make_token_dropout_mask(token_tensor, token_keep_ratio, token_dropout_seed=None, token_dropout_ids=None,
                                 masking_mode="random", token_grid=None):
        if masking_mode == "none" or token_keep_ratio >= 1.0:
            return None
        if token_keep_ratio <= 0.0:
            raise ValueError(f"token_keep_ratio must be > 0, got {token_keep_ratio}")

        batch_size = token_tensor.shape[0]
        token_count = token_tensor.shape[2] * token_tensor.shape[3]
        keep_count = min(token_count, max(1, int(math.ceil(token_count * token_keep_ratio))))
        if keep_count == token_count:
            return None

        mask = torch.zeros((batch_size, token_count), dtype=token_tensor.dtype, device=token_tensor.device)
        if token_dropout_ids is None:
            token_dropout_ids = range(batch_size)
        spatial_modes = {"center", "block"}
        supports_spatial_mask = token_tensor.dim() == 4 and token_grid is not None
        if masking_mode in spatial_modes and not supports_spatial_mask:
            logging.warning("Masking mode '%s' requires spatial token coordinates; falling back to random masking.", masking_mode)
            masking_mode = "random"

        height = token_tensor.shape[2]
        width = token_tensor.shape[3]

        for batch_index, sample_id in enumerate(token_dropout_ids):
            sample_seed = None if token_dropout_seed is None else int(token_dropout_seed) + int(sample_id)
            if masking_mode == "random":
                mask[batch_index] = SuperVLADModel._make_random_token_mask(
                    mask[batch_index], keep_count, token_count, sample_seed
                )
            elif masking_mode == "center":
                mask[batch_index] = SuperVLADModel._make_center_keep_mask(
                    mask[batch_index], keep_count, height, width
                )
            elif masking_mode == "block":
                mask[batch_index] = SuperVLADModel._make_block_keep_mask(
                    mask[batch_index], keep_count, height, width, sample_seed
                )
            else:
                raise ValueError(f"Unsupported masking_mode: {masking_mode}")
        return mask

    def _aggregate(self, x, return_debug=False, low_mass_threshold=1e-3, token_mask=None):
        if isinstance(self.aggregation, nn.Sequential):
            if return_debug:
                pooled, pooling_debug = self.aggregation[0](
                    x, return_debug=True, low_mass_threshold=low_mass_threshold, token_mask=token_mask
                )
            else:
                pooled = self.aggregation[0](x, token_mask=token_mask)
                pooling_debug = None
            for layer in self.aggregation[1:]:
                pooled = layer(pooled)
            return pooled, pooling_debug

        if not return_debug:
            return self.aggregation(x, token_mask=token_mask), None

        pooled, pooling_debug = self.aggregation(
            x, return_debug=True, low_mass_threshold=low_mass_threshold, token_mask=token_mask
        )
        return pooled, pooling_debug

    def forward(self, x, queryflag=0, return_debug=False, pool_mask=None, pool_hook=None,
                low_mass_threshold=1e-3, token_keep_ratio=1.0, token_dropout_seed=None,
                token_dropout_ids=None, masking_mode="none"):
        x = self.backbone(x)
        token_tensor = None
        token_grid = None

        if self.arch_name.startswith("vit"):
            B,P,D = x.last_hidden_state.shape
            W = H = int(math.sqrt(P-1))
            token_tensor = x.last_hidden_state[:, 1:, :].view(B, W, H, D).permute(0, 3, 1, 2)
            token_grid = self._build_token_grid(H, W, token_tensor.device)
        elif self.arch_name.startswith("cct"):
            B,P,D = x.shape
            token_tensor = x.view(-1,24,24,384).permute(0, 3, 1, 2)
            token_grid = self._build_token_grid(24, 24, token_tensor.device)
        elif self.arch_name.startswith("dino"):
            B,P,D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            x0 = x["x_norm_clstoken"]
            token_tensor = x["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)
            token_grid = self._build_token_grid(H, W, token_tensor.device)
        else:
            B, D = x.shape[:2]
            token_tensor = x
            if x.dim() == 4:
                token_grid = self._build_token_grid(x.shape[2], x.shape[3], x.device)
        if pool_mask is None and token_tensor is not None and token_tensor.dim() == 4:
            pool_mask = self._make_token_dropout_mask(
                token_tensor,
                token_keep_ratio=token_keep_ratio,
                token_dropout_seed=token_dropout_seed,
                token_dropout_ids=token_dropout_ids,
                masking_mode=masking_mode,
                token_grid=token_grid,
            )
        if pool_hook is not None:
            hook_output = pool_hook(token_tensor, pool_mask, token_grid)
            if hook_output is not None:
                token_tensor, pool_mask, token_grid = hook_output
        x, pooling_debug = self._aggregate(
            token_tensor, return_debug=return_debug, low_mass_threshold=low_mass_threshold, token_mask=pool_mask
        )

        if self.crossimage_encoder:
            x = self.encoder(x.view(B,-1,D)).view(B,-1)

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        if not return_debug:
            return x

        return {
            "descriptor": x,
            "pool_debug": pooling_debug,
            "tokens_pre_pool": token_tensor,
            "token_positions": token_grid,
            "pool_mask": pool_mask,
            "pool_hook": pool_hook,
        }


def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_backbone(args, pretrained_foundation, foundation_model_path):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = False
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # if args.resize[0] == 224:
        #     backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # elif args.resize[0] == 384:
        #     backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        # else:
        #     raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    elif args.backbone.startswith("dino"):
        backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  
        if pretrained_foundation:
            assert foundation_model_path is not None, "Please specify foundation model path."
            model_dict = backbone.state_dict()
            state_dict = torch.load(foundation_model_path)
            model_dict.update(state_dict.items())
            backbone.load_state_dict(model_dict)

        if args.freeze_te:
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.blocks.named_children():
                if int(name) >= args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True

        args.features_dim = 768#1024
        return backbone
    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone

def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]
