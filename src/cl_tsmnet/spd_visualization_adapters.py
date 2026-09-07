"""Model-specific adapters for extracting validated analysis intermediates."""

import torch
import torch.nn.functional as F


MS_TGC_MODEL_TYPES = {"ms_tgc_spddsbn", "mstgc_augspd_spddsbn"}
SUPPORTED_SPD_VISUALIZATION_MODELS = MS_TGC_MODEL_TYPES | {"tsmnet"}
MSTGC_ALIGNMENT_MODELS = {
    "mstgc_mean_ce",
    "mstgc_dta_cheb_eudsbn",
    "mstgc_dta_cheb_spdmbn",
    "mstgc_dta_cheb_spdbn",
    "mstgc_wo_spddsbn",
    "ms_tgc_spddsbn",
    "mstgc_augspd_spddsbn",
}
SUPPORTED_ALIGNMENT_MODELS = MSTGC_ALIGNMENT_MODELS | {"tsmnet"}


def extract_spd_intermediates(model, windows, domains, model_type):
    """Return logits and named matrices immediately before/after SPDDSBN.

    TSMNet's original forward contract returns optional values in reverse
    request order.  This adapter is the only place that decodes that positional
    tuple; downstream visualization code always consumes named matrices.
    """
    model_type = str(model_type)
    if model_type == "tsmnet":
        output = model(
            windows,
            domains,
            return_latent=False,
            return_prebn=True,
            return_postbn=True,
        )
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError(
                "TSMNet intermediate forward must return (logits, post_bn, pre_bn)"
            )
        logits, post_bn, pre_bn = output
        return logits, {
            "spd_pre_bn": pre_bn,
            "spd_post_bn": post_bn,
        }
    if model_type in MS_TGC_MODEL_TYPES:
        return model(windows, domains, return_intermediates=True)
    raise ValueError(
        "No SPD visualization adapter is registered for model {!r}".format(
            model_type
        )
    )


def visualization_model_metadata(model_type):
    model_type = str(model_type)
    if model_type == "tsmnet":
        return {
            "model_class": "TSMNet.spdnets.models.tsmnet.TSMNet",
            "pre": "TSMNet.spdnet output (BiMap + ReEig), before SPDDSBN",
            "post": "TSMNet.spddsbnorm output, before LogEig",
        }
    if model_type in MS_TGC_MODEL_TYPES:
        return {
            "model_class": "src.cl_tsmnet.ms_tgc_spddsbn.MSTGCSPDDSBN",
            "pre": "GraphSPDManifoldHead.manifold_features (BiMap + ReEig output)",
            "post": "GraphSPDManifoldHead normalization output before LogEig",
        }
    raise ValueError(
        "No SPD visualization metadata is registered for model {!r}".format(
            model_type
        )
    )


def extract_alignment_representation(model, windows, domains, model_type):
    """Extract the trained representation immediately before classification."""
    model_type = str(model_type)
    if model_type == "tsmnet":
        output = model(windows, domains, return_latent=True)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "TSMNet alignment forward must return (logits, latent)"
            )
        _, latent = output
        return latent.float(), {
            "location": "TSMNet LogEig output after configured normalization",
            "representation": "covariance_spd_tangent",
            "normalization": str(getattr(model, "bnorm_", None)),
        }
    if model_type in MSTGC_ALIGNMENT_MODELS:
        return model.extract_alignment_representation(windows, domains)
    raise ValueError(
        "No alignment-representation adapter is registered for model {!r}"
        .format(model_type)
    )


def extract_static_channel_interaction(model, model_type):
    """Return a model-derived symmetric channel interaction matrix.

    MS-TGC returns the actual adjacency used by Chebyshev propagation. TSMNet
    has no graph; its return value is explicitly a spatial-filter channel
    similarity proxy and must not be described as an adaptive adjacency.
    """
    model_type = str(model_type)
    if model_type in MSTGC_ALIGNMENT_MODELS:
        graph = getattr(model, "graph", None)
        if graph is None or str(getattr(graph, "graph_mode", "")) != "adaptive":
            raise ValueError("Graph-pattern analysis requires an adaptive MS-TGC graph")
        adjacency = graph._adjacencies()[0]
        metadata = {
            "interaction_kind": "adaptive_chebyshev_adjacency",
            "used_for_message_passing": True,
        }
    elif model_type == "tsmnet":
        spatial = model.cnn[1]
        weights = spatial.weight[..., 0].permute(2, 0, 1).reshape(
            spatial.weight.shape[2], -1
        )
        normalized = F.normalize(weights, p=2, dim=1, eps=1e-12)
        adjacency = torch.abs(normalized @ normalized.t())
        metadata = {
            "interaction_kind": "tsmnet_spatial_filter_channel_similarity_proxy",
            "used_for_message_passing": False,
        }
    else:
        raise ValueError("Unsupported graph-analysis model: {!r}".format(model_type))
    adjacency = 0.5 * (adjacency + adjacency.t())
    adjacency = adjacency - torch.diag_embed(torch.diagonal(adjacency))
    return adjacency, metadata


def extract_graph_analysis_features(model, windows, domains, model_type):
    """Return pre-interaction temporal maps and per-channel response maps."""
    model_type = str(model_type)
    if model_type in MSTGC_ALIGNMENT_MODELS:
        temporal, response = model.extract_graph_analysis_features(windows)
        metadata = {
            "temporal_location": "shared channel-wise temporal encoder output",
            "response_location": "Chebyshev output before channel reliability and SPD pooling",
            "interaction_kind": "adaptive_chebyshev_adjacency",
        }
        return temporal, response, metadata
    if model_type == "tsmnet":
        temporal = model.cnn[0](
            windows.to(device=model.device_, dtype=torch.float32)[:, None, ...]
        )
        temporal = temporal.permute(0, 2, 1, 3)
        spatial = model.cnn[1].weight[..., 0]
        channel_scale = torch.sqrt(torch.sum(spatial.square(), dim=(0, 1)))
        response = temporal * channel_scale[None, :, None, None]
        metadata = {
            "temporal_location": "TSMNet temporal convolution output",
            "response_location": "temporal response weighted by spatial-kernel channel norm",
            "interaction_kind": "tsmnet_spatial_filter_channel_similarity_proxy",
        }
        return temporal, response, metadata
    raise ValueError("Unsupported graph-analysis model: {!r}".format(model_type))
