"""
Optuna hyperparameter search for ReYOLOv8 on the MTevent dataset.

Usage:
    python3 tune_optuna.py \
        --weights reyolov8s_gen1_rps.pt \
        --data vtei_mtevent_50ms.yaml \
        --device 0 \
        --n_trials 30 \
        --epochs 30 \
        --batch 4

Each trial runs a short training run and reports the best fitness
(0.1*mAP50 + 0.9*mAP50-95) to Optuna. Results are stored in an SQLite
database (optuna_reyolov8.db) so the study survives restarts.
"""

import argparse
import copy
import sys
import os
from pathlib import Path
import yaml
import optuna
from optuna.samplers import TPESampler

# ── project root on sys.path ──────────────────────────────────────────────────
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the trainer class (train.py is now importable without side effects)
from train import EventVideoYOLOv8DetectionTrainer  # noqa: E402


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='reyolov8s_gen1_rps.pt')
    p.add_argument('--data',    type=str, default='vtei_mtevent_50ms.yaml')
    p.add_argument('--hyp',     type=str, default='default_gen1.yaml',
                   help='Base hyperparameter YAML (relative to project root)')
    p.add_argument('--device',  type=str, default='0')
    p.add_argument('--epochs',  type=int, default=30,
                   help='Epochs per trial (short for fast search)')
    p.add_argument('--batch',   type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--n_trials', type=int, default=30)
    p.add_argument('--study_name', type=str, default='reyolov8_mtevent')
    p.add_argument('--storage',   type=str, default='sqlite:///optuna_reyolov8.db',
                   help='Optuna storage URL. Use "none" for in-memory.')
    p.add_argument('--patience', type=int, default=15,
                   help='EarlyStopping patience per trial')
    p.add_argument('--val_epoch', type=int, default=2,
                   help='Validate every N epochs inside a trial')
    # fixed video settings
    p.add_argument('--clip_length', type=int, default=11)
    p.add_argument('--channels',    type=int, default=5)
    p.add_argument('--imgsz',       type=int, nargs='+', default=[320])
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def load_base_overrides(hyp_path: str) -> dict:
    """Load base hyperparameters from the YAML file."""
    p = Path(hyp_path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text())


def build_overrides(args, trial: optuna.Trial, base: dict, trial_idx: int) -> dict:
    """Merge base overrides with trial-suggested hyperparameters."""
    ov = copy.deepcopy(base)

    # ── fixed settings ─────────────────────────────────────────────────────
    ov['model']       = str(args.weights)
    ov['data']        = str(args.data)
    ov['device']      = args.device
    ov['epochs']      = args.epochs
    ov['batch']       = args.batch
    ov['workers']     = args.workers
    ov['patience']    = args.patience
    ov['val_epoch']   = args.val_epoch
    ov['clip_length'] = args.clip_length
    ov['clip_stride'] = 5        # kept fixed; can be added to search space
    ov['channels']    = args.channels
    ov['imgsz']       = args.imgsz
    ov['plots']       = False    # skip plots to save time
    ov['save']        = True
    ov['save_period'] = -1
    ov['rect']        = True
    ov['resume']      = False
    ov['pretrained']  = False
    ov['half']        = False
    ov['nbs']         = 64
    ov['project']     = str(ROOT / 'runs' / 'tune')
    ov['name']        = f'trial_{trial_idx:04d}'
    ov['exist_ok']    = True

    # ── search space ───────────────────────────────────────────────────────
    # Learning rate
    ov['lr0']            = trial.suggest_float('lr0', 1e-5, 5e-3, log=True)
    ov['lrf']            = trial.suggest_float('lrf', 0.001, 0.1)
    ov['momentum']       = trial.suggest_float('momentum', 0.85, 0.98)
    ov['weight_decay']   = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    ov['optimizer']      = trial.suggest_categorical('optimizer', ['SGD', 'AdamW'])

    # Warmup
    ov['warmup_epochs']   = trial.suggest_int('warmup_epochs', 1, 5)
    ov['warmup_bias_lr']  = trial.suggest_float('warmup_bias_lr', 0.01, 0.2)
    ov['warmup_momentum'] = trial.suggest_float('warmup_momentum', 0.7, 0.95)

    # Loss weights
    ov['box'] = trial.suggest_float('box', 3.0, 12.0)
    ov['cls'] = trial.suggest_float('cls', 0.5, 4.0)
    ov['dfl'] = trial.suggest_float('dfl', 0.5, 3.0)

    # Augmentation
    ov['flip']               = trial.suggest_float('flip', 0.0, 0.5)
    ov['invert']             = trial.suggest_float('invert', 0.0, 0.3)
    ov['zoom_out']           = trial.suggest_float('zoom_out', 0.0, 0.4)
    ov['max_zoom_out_factor'] = trial.suggest_float('max_zoom_out_factor', 1.1, 2.0)
    ov['min_zoom_out_factor'] = 1.0
    ov['suppress']           = 0.0
    ov['positive']           = 0.0

    # LR schedule
    ov['cos_lr'] = trial.suggest_categorical('cos_lr', [True, False])

    return ov


# ── objective ─────────────────────────────────────────────────────────────────
def make_objective(args, base_overrides):
    def objective(trial: optuna.Trial) -> float:
        ov = build_overrides(args, trial, base_overrides, trial.number)
        try:
            trainer = EventVideoYOLOv8DetectionTrainer(overrides=ov)
            trainer.train()
            # best_fitness is tracked by EarlyStopping (0.1*mAP50 + 0.9*mAP50-95)
            best = trainer.stopper.best_fitness
        except Exception as e:
            print(f'[Trial {trial.number}] FAILED: {e}')
            return 0.0
        finally:
            # free GPU memory between trials
            import torch
            torch.cuda.empty_cache()

        print(f'[Trial {trial.number}] best_fitness={best:.4f}')
        return float(best)

    return objective


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    base_overrides = load_base_overrides(args.hyp)

    storage = None if args.storage.lower() == 'none' else args.storage
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction='maximize',
        load_if_exists=True,
    )

    print(f'Starting Optuna study "{args.study_name}" — {args.n_trials} trials, '
          f'{args.epochs} epochs each.')
    if storage:
        print(f'Results stored in: {storage}')

    study.optimize(
        make_objective(args, base_overrides),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ── report best trial ──────────────────────────────────────────────────
    best = study.best_trial
    print('\n' + '='*60)
    print(f'Best trial #{best.number}  fitness={best.value:.4f}')
    print('Hyperparameters:')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    # Save best params to YAML for easy reuse
    out_yaml = ROOT / 'runs' / 'tune' / 'best_params.yaml'
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, 'w') as f:
        yaml.dump({'best_fitness': best.value, 'params': best.params}, f,
                  default_flow_style=False)
    print(f'\nBest params saved to: {out_yaml}')


if __name__ == '__main__':
    main()
