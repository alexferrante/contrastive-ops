from pathlib import Path
from src.helper import get_module, embed_images
import logging
import argparse
# import wandb
# from lightning.pytorch.loggers import WandbLogger


def embed(run_id, run_name, save_dir, version='best', loader_param=None, module='contrastive', artifact_fetcher=None):
    """
    General-purpose embedding function.

    Args:
        run_id (str): ID of the run or experiment.
        run_name (str): Name of the run or experiment.
        save_dir (str): Directory where the embeddings should be saved.
        version (str): Version of the model checkpoint (default: 'best').
        loader_param (dict): Parameters for the dataloader (default: None).
        module (str): Module to identify the data loader type (default: 'contrastive').
        artifact_fetcher (function): Function to fetch model artifacts (default: None).

    Returns:
        None
    """
    logging.info(f"Starting embedding for run ID: {run_id}, run name: {run_name}, version: {version}")
    
    if artifact_dir is None:
        artifact_dir = Path(save_dir) / "artifacts" / run_id  # Default location
    logging.info(f"Resolved artifact directory: {artifact_dir}")

    checkpt_path = Path(artifact_dir) / "model.ckpt"
    if not checkpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpt_path}")
    logging.info(f"Loading checkpoint from: {checkpt_path}")
    
    modelname = run_name
    ModelClass = get_module(modelname, 'model')
    DataClass = get_module(module, 'dataloader')
    
    logging.info(f"Loading checkpoint from: {checkpt_path}")
    model = ModelClass.load_from_checkpoint(checkpt_path)
    dm = DataClass.load_from_checkpoint(checkpt_path)

    logging.info(f"Generating embeddings for module: {module}")
    embedding_df = embed_images(model, 
                                dm, 
                                stage='embed', 
                                loader_param=loader_param, 
                                modelname=module)
    
    output_file = Path(save_dir) / f'{run_name}_{run_id}_{version}.pkl'
    embedding_df.to_pickle(output_file)
    logging.info(f"Embeddings saved to: {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding script with run_id and run_name arguments.")
    parser.add_argument("--run_id", type=str, help="The run ID.")
    parser.add_argument("--run_name", type=str, help="The run name.")
    parser.add_argument("-v", "--version", type=str, default="best")
    parser.add_argument("--batch_size", type=int, default=4200, help="Batch size for the loader.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the loader.")
    parser.add_argument("--module", type=str, default="contrastive", help="The module name.")
    args = parser.parse_args()

    loader_param = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    embed(run_id=args.run_id, run_name=args.run_name, version=args.version, loader_param=loader_param, module=args.module)
