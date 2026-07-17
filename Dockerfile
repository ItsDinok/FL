FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates curl \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cacge-dir \
	torchvision == 0.19.1 \
	flwr == 1.11.1 \
	numpy \
	pandas

ENV T0RCH_HOME=/opt/torch-cache
RUN python -c "from torchvision.models import vit_b_16, ViT_B_16_WEIGHTS; \
	vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)" \
	&& chmod -R a+r /opt/torch-cache

ENV PYTHONUNBUFFERED = 1\
	GRPC_VERBOSITY=ERROR \
	XDG_CACHE_HOME=/workspace/.cache

WORKDIR /workspace

CMD["/bin/bash"]
