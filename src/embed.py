from pathlib import Path
from src.helper import get_module, embed_images
import logging
import argparse
import jsonpickle

def embed(run_name, save_dir, ckpt_path, loader_param=None, module='contrastive'):
    """
    General-purpose embedding function.

    Args:
        run_name (str): Name of the run or experiment.
        save_dir (str): Directory where the embeddings should be saved.
        version (str): Version of the model checkpoint (default: 'best').
        loader_param (dict): Parameters for the dataloader (default: None).
        module (str): Module to identify the data loader type (default: 'contrastive').

    Returns:
        None
    """
    checkpt_path = Path(ckpt_path)
    if not checkpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpt_path}")
    logging.info(f"Loading checkpoint from: {checkpt_path}")
    
    modelname = run_name.split("_")[0]
    ModelClass = get_module(modelname, 'model')
    DataClass = get_module(module, 'dataloader')

    checkpoint = torch.load(ckpt_path)
    hparams = checkpoint['hyper_parameters']
    model = ModelClass.load_from_checkpoint(ckpt_path, **hparams, strict=False)    
    
    dm = DataClass.load_from_checkpoint(checkpt_path)

    logging.info(f"Generating embeddings for module: {module}")
    embedding_df = embed_images(model, dm, stage='embed', loader_param=loader_param, modelname=module)
    
    output_file = Path(save_dir) / f'{run_name}.pkl'
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
