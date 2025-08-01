import argparse


def get_argparser():
    """
    Create and return an argparse.ArgumentParser pre-configured with all command-line options
    required for running exCLIP experiments. The parser includes arguments for configuration files,
    device selection, output directories, checkpoint paths, data loading, model parameters,
    training and evaluation flags, and experiment tracking.
    """
    parser = argparse.ArgumentParser("explainable CLIP (exCLIP)", add_help=False)

    # string arguments
    parser.add_argument("--config", default="./exCLIP/configs/config.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="./debug_dir")
    parser.add_argument("--ckpt_path", type=str, default="none")

    # integer arguments
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--general_hidden_dim", type=int, default=512)
    parser.add_argument("--inference_batch_size_scale", type=int, default=15)

    # float arguments
    parser.add_argument("--train_data_ratio", type=float, default=1.0)

    # store true arguments
    parser.add_argument("--create_ans_label_dicts", action="store_true", default=False)
    parser.add_argument("--weight_norm", action="store_true", default=False)
    parser.add_argument("--init_weights", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--pre_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--load_weights", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument("--only_load_model_weights", action="store_true", default=False)

    # comet_ml related args
    parser.add_argument("--experiment_name", type=str)

    return parser
