"""Model-specific adapters for extracting comparable SPD intermediates."""


MS_TGC_MODEL_TYPES = {"ms_tgc_spddsbn", "mstgc_augspd_spddsbn"}
SUPPORTED_SPD_VISUALIZATION_MODELS = MS_TGC_MODEL_TYPES | {"tsmnet"}


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
