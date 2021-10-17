import sys
import json
from detect import DarknetDetector
from train import DarknetTrainer


def configure_json(json_path):
    with open(json_path, 'r') as f:
        s = f.read()
        s = s.replace('\t', '')
        s = s.replace('\n', '')
        s = s.replace(',}', '}')
        s = s.replace(',]', ']')
        return json.loads(s)


def fetch_parameters():
    params = configure_json("params.json")
    detector_params = {
        "images": params["detector_params"]["images_path"],
        "destination": params["detector_params"]["destination_path"],
        "cfg_path": params["detector_params"]["cfg_file_path"],
        "weights_path": params["detector_params"]["weights_file_path"],
        "resolution": params["detector_params"]["resolution"],
        "confidence": params["detector_params"]["confidence"],
        "nms_thresh": params["detector_params"]["nms_threshold"],
        "CUDA": params["detector_params"]["CUDA"],
        "TORCH": params["detector_params"]["use_torch_weights"],
    }
    trainer_params = {
        "cfg_file": params["detector_params"]["cfg_file_path"],
        "weights_file": params["detector_params"]["weights_file_path"],
        "epoch": params["training_params"]["number_of_epoch"],
        "batch_size": params["training_params"]["batch_size"],
        "resolution": params["detector_params"]["resolution"],
        "confidence": params["detector_params"]["confidence"],
        "CUDA": params["detector_params"]["CUDA"],
        "TUNE": params["training_params"]["start_from_checkpoint"],
    }
    validator_params = {
        "annotation_dir": params["training_params"]["valid_annot_dir"],
        "img_dir": params["training_params"]["valid_img_dir"],
    }
    train_params = {
        "annotation_dir": params["training_params"]["train_annot_dir"],
        "img_dir": params["training_params"]["train_img_dir"],
    }
    return detector_params, trainer_params, validator_params, train_params


def main():
    if sys.argv[1] == "detect":
        detector_params, _, _, _ = fetch_parameters()
        detector = DarknetDetector(**detector_params)
        detector()
    elif sys.argv[1] == "train":
        _, trainer_params, valid_params, train_params = fetch_parameters()
        trainer = DarknetTrainer(**trainer_params)
        trainer.get_validator(**valid_params)
        trainer.train(**train_params)
    else:
        raise Exception("Unknown Command Error !!")


if __name__ == "__main__":
    main()
