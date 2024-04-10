import argparse
import torch
from torchvision.io import read_video
from omegaconf import OmegaConf
from videosaur import configuration, models
from videosaur.data.transforms import CropResize, Normalize
import os
import numpy as np
import imageio
from torchvision import transforms as tvt
from videosaur.visualizations import mix_inputs_with_masks
from videosaur.data.transforms import build_inference_transform


def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, config

def prepare_inputs(video_path: str, transfom_config=None):
    # Load video
    video, _, _ = read_video(video_path)
    video = video.float() / 255.0
    #change size of the video to 224x224
    video_vis = video.permute(0, 3, 1, 2)
    video_vis = tvt.Resize((transfom_config.input_size, transfom_config.input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)

    
    if transfom_config:
        tfs = build_inference_transform(transfom_config)
        video = video.permute(3, 0, 1, 2)
        video = tfs(video).permute(1, 0, 2, 3)
     # Add batch dimension
    inputs = {"video": video.unsqueeze(0), 
              "video_visualization": video_vis.unsqueeze(0)}
    return inputs




def main(config):
    # Load the model from checkpoint
    model, _ = load_model_from_checkpoint(config.checkpoint, config.model_config)
    # Prepare the video dict
    inputs = prepare_inputs(config.input.path, config.input.transforms)
    # Perform inference
    with torch.no_grad():
        outputs = model(inputs)
    if config.output.save_video_path:
        # Save the results
        save_dir = os.path.dirname(config.output.save_video_path)
        os.makedirs(save_dir, exist_ok=True)
        masked_video_frames = mix_inputs_with_masks(inputs, outputs)
        with imageio.get_writer(config.output.save_video_path, fps=10) as writer:
            for frame in masked_video_frames:
                writer.append_data(frame)
        writer.close()
    print("Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on a single MP4 video.")
    parser.add_argument("--config", default="configs/inference/movi_c.yml", help="Configuration to run")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)