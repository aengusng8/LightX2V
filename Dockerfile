FROM lightx2v/lightx2v:26011201-cu128


COPY requirements_animate.txt /app/requirements_animate.txt

WORKDIR /workspace
RUN TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install --no-build-isolation --no-cache-dir -r /app/requirements_animate.txt
RUN pip install --no-cache-dir diffusers==0.37.1 moviepy gradio