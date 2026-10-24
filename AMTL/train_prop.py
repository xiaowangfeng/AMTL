import csv
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import time
from typing import Any, Dict, Union, Tuple
import pickle as pk
import os
import torch.distributed as dist

import torch
from jarvis.core.atoms import Atoms

from jarvis.db.jsonutils import dumpjson, loadjson
from torch import nn

import ignite
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from data import get_train_val_loaders
from ignite.metrics import Metric
from models.config import TrainingConfig
import json
import pprint
import torch.nn.functional as F
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers.stores import EpochOutputStore
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    Engine,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from models.potnet import PotNet

import random

plt.switch_backend("agg")

device = torch.device("cuda")
print(device)


def prepare_batch(
        batch, device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    batch = (
        batch.to(device, non_blocking=non_blocking),
        batch.y.to(device, non_blocking=non_blocking),
    )

    return batch


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def reshape_multitask_tensor(tensor, num_tasks):
    if tensor.dim() == 1:
        return tensor.view(-1, num_tasks)
    return tensor


class GradNormMultiTaskLoss(nn.Module):
    def __init__(
            self,
            num_tasks,
            alpha=1.5,
            loss_name="mse",
            ema_beta=0.9,
            reference_beta=0.99,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.loss_name = loss_name
        self.ema_beta = ema_beta
        self.reference_beta = reference_beta
        self.task_weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))
        self.register_buffer("initial_losses", torch.zeros(num_tasks, dtype=torch.float32))
        self.register_buffer("ema_task_losses", torch.zeros(num_tasks, dtype=torch.float32))
        self.register_buffer("loss_reference", torch.zeros(num_tasks, dtype=torch.float32))
        self.register_buffer("ema_grad_norms", torch.zeros(num_tasks, dtype=torch.float32))
        self.initialized = False
        self.gradnorm_initialized = False

    def compute_task_losses(self, preds, targets):
        preds = reshape_multitask_tensor(preds, self.num_tasks)
        targets = reshape_multitask_tensor(targets, self.num_tasks)
        task_losses = []
        for task_idx in range(self.num_tasks):
            if self.loss_name == "mse":
                task_loss = F.mse_loss(preds[:, task_idx], targets[:, task_idx])
            elif self.loss_name == "l1":
                task_loss = F.l1_loss(preds[:, task_idx], targets[:, task_idx])
            elif self.loss_name == "poisson":
                task_loss = F.poisson_nll_loss(
                    preds[:, task_idx],
                    targets[:, task_idx],
                    log_input=False,
                    full=True,
                )
            else:
                raise ValueError(f"Unsupported multitask criterion: {self.loss_name}")
            task_losses.append(task_loss)
        return torch.stack(task_losses)

    def weighted_loss(self, task_losses):
        return torch.sum(self.task_weights * task_losses)

    def maybe_store_initial_losses(self, task_losses):
        if not self.initialized:
            self.initial_losses.copy_(task_losses.detach())
            self.ema_task_losses.copy_(task_losses.detach())
            self.loss_reference.copy_(task_losses.detach())
            self.initialized = True

    def update_ema_losses(self, task_losses):
        detached_losses = task_losses.detach()
        if not self.initialized:
            self.maybe_store_initial_losses(detached_losses)
            return self.ema_task_losses.clone(), self.loss_reference.clone()

        self.ema_task_losses.mul_(self.ema_beta).add_(detached_losses, alpha=1.0 - self.ema_beta)
        self.loss_reference.mul_(self.reference_beta).add_(
            self.ema_task_losses,
            alpha=1.0 - self.reference_beta,
        )
        return self.ema_task_losses.clone(), self.loss_reference.clone()

    def update_ema_grad_norms(self, grad_norms):
        detached_grad_norms = grad_norms.detach()
        if not self.gradnorm_initialized:
            self.ema_grad_norms.copy_(detached_grad_norms)
            self.gradnorm_initialized = True
        else:
            self.ema_grad_norms.mul_(self.ema_beta).add_(
                detached_grad_norms,
                alpha=1.0 - self.ema_beta,
            )
        return self.ema_grad_norms.clone()

    def reset_initial_losses(self):
        self.initial_losses.zero_()
        self.ema_task_losses.zero_()
        self.loss_reference.zero_()
        self.ema_grad_norms.zero_()
        self.initialized = False
        self.gradnorm_initialized = False

    def renormalize_(self):
        with torch.no_grad():
            min_weight = 0.18
            weights = self.task_weights.data.clamp(min=min_weight)
            remaining_weight = self.num_tasks - self.num_tasks * min_weight
            excess_weights = (weights - min_weight).clamp(min=0.0)

            if excess_weights.sum() > 0:
                weights = min_weight + remaining_weight * (
                    excess_weights / excess_weights.sum()
                )
            else:
                weights.fill_(1.0)

            self.task_weights.data.copy_(weights)

    def get_weights(self):
        return self.task_weights.detach().clone()

    def forward(self, preds, targets):
        task_losses = self.compute_task_losses(preds, targets)
        return self.weighted_loss(task_losses)


def get_shared_layer_parameter_groups(model):
    shared_parameter_groups = []
    shared_modules = []

    if hasattr(model, "conv_layers"):
        conv_layers = list(model.conv_layers)
        shared_modules.extend(conv_layers[-2:] if len(conv_layers) >= 2 else conv_layers)
    if hasattr(model, "transformer_conv_layers") and len(model.transformer_conv_layers) > 0:
        shared_modules.append(model.transformer_conv_layers[-1])

    for module in shared_modules:
        params = [param for param in module.parameters() if param.requires_grad]
        if params:
            shared_parameter_groups.append(params)

    return shared_parameter_groups


def create_gradnorm_trainer(
        model,
        optimizer,
        gradnorm_optimizer,
        criterion,
        prepare_batch,
        shared_parameter_groups,
        warmup_epochs=0,
        device=None,
        deterministic=False,
):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        gradnorm_optimizer.zero_grad()

        x, y = prepare_batch(batch, device=device, non_blocking=False)
        preds = model(x)
        preds = reshape_multitask_tensor(preds, criterion.num_tasks)
        y = reshape_multitask_tensor(y, criterion.num_tasks)

        task_losses = criterion.compute_task_losses(preds, y)
        current_epoch = engine.state.epoch if engine.state.epoch is not None else 1
        warmup_active = current_epoch <= warmup_epochs

        # warmup_weights = torch.ones(criterion.num_tasks, device=preds.device)
        warmup_weights = torch.tensor([1.0, 1.0], device=preds.device)

        if warmup_active:
            ###2026.4.25修改权重weighted_loss = torch.sum(task_losses)
            weighted_loss = torch.sum(warmup_weights * task_losses)
            weighted_loss.backward()
            optimizer.step()
        else:
            criterion.maybe_store_initial_losses(task_losses)
            ema_task_losses, loss_reference = criterion.update_ema_losses(task_losses)
            weighted_loss = criterion.weighted_loss(task_losses)

            grad_norms = []
            for task_idx in range(criterion.num_tasks):
                layer_grad_norms = []
                for layer_parameters in shared_parameter_groups:
                    grads = torch.autograd.grad(
                        criterion.task_weights[task_idx] * task_losses[task_idx],
                        layer_parameters,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    grad_components = [grad.norm(2) for grad in grads if grad is not None]
                    if grad_components:
                        layer_grad_norms.append(torch.norm(torch.stack(grad_components), p=2))
                if layer_grad_norms:
                    grad_norm = torch.stack(layer_grad_norms).mean()
                else:
                    grad_norm = torch.zeros((), device=preds.device)
                grad_norms.append(grad_norm)

            grad_norms = torch.stack(grad_norms)
            ema_grad_norms = criterion.update_ema_grad_norms(grad_norms)
            with torch.no_grad():
                loss_ratios = ema_task_losses / loss_reference.clamp_min(1e-8)
                inverse_train_rates = loss_ratios / loss_ratios.mean()
                target_grad_norms = ema_grad_norms.mean() * inverse_train_rates.pow(criterion.alpha)

            gradnorm_loss = F.l1_loss(ema_grad_norms + (grad_norms - grad_norms.detach()), target_grad_norms, reduction="sum")
            gradnorm_grads = torch.autograd.grad(
                gradnorm_loss,
                criterion.task_weights,
                retain_graph=True,
            )[0]

            weighted_loss.backward()
            criterion.task_weights.grad = gradnorm_grads

            optimizer.step()
            gradnorm_optimizer.step()
            criterion.renormalize_()

        engine.state.task_losses = task_losses.detach().cpu()
        engine.state.task_weights = criterion.get_weights().cpu()
        engine.state.ema_task_losses = criterion.ema_task_losses.detach().cpu()
        engine.state.ema_grad_norms = criterion.ema_grad_norms.detach().cpu()
        engine.state.warmup_active = warmup_active
        return weighted_loss.item()

    return Engine(_update)

##新定义的多任务损失MAE
class MultiTaskMAE(Metric):
    def __init__(self, task_idx, scale=1.0, output_transform=lambda x: x):
        super().__init__(output_transform)
        self.task_idx = task_idx  # 保持为整数
        self.scale = float(scale)
        self.mae = 0.0
        self.count = 0

    def reset(self):
        self.mae = 0.0
        self.count = 0

    def update(self, output):
        preds, targets = output
        num_tasks = preds.shape[-1] if preds.dim() > 1 else targets.numel()
        targets = reshape_multitask_tensor(targets, num_tasks)
        preds = reshape_multitask_tensor(preds, num_tasks)
            
        preds_task = preds[:, self.task_idx]  # 直接索引
        targets_task = targets[:, self.task_idx]
        
        self.mae += torch.sum(torch.abs(preds_task - targets_task)).item() * self.scale
        self.count += preds_task.size(0)

    def compute(self):
        return self.mae / self.count if self.count != 0 else 0.0
##新定义的多任务损失MAEending

def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total Trainable Params: {total_params}")
    return total_params


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_pyg(
        config: Union[TrainingConfig, Dict[str, Any]],
        data_root: str = None,
        file_format: str = 'poscar',
        checkpoint: str = None,
        testing: bool = False,
        train_val_test_loaders=None,
):
    # print(config)
    config = TrainingConfig(**config)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir, config.checkpoint_dir)
    deterministic = False
    print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    pprint.pprint(tmp)  # , sort_dicts=False)

    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        random.seed(config.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if data_root:
        dataset_info = loadjson(os.path.join(data_root, "dataset_info.json"))
        if "n_train" in dataset_info:
            config.n_train = dataset_info['n_train']
        if "n_val" in dataset_info:
            config.n_val = dataset_info['n_val']
        if "n_test" in dataset_info:
            config.n_test = dataset_info['n_test']

        if "train_ratio" in dataset_info:
            config.train_ratio = dataset_info['train_ratio']
        if "val_ratio" in dataset_info:
            config.val_ratio = dataset_info['val_ratio']
        if "test_ratio" in dataset_info:
            config.test_ratio = dataset_info['test_ratio']

        config.keep_data_order = True
        config.target = "target"

        id_prop_dat = os.path.join(data_root, "id_prop.csv")
        with open(id_prop_dat, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
        # print(f"data={data}")
        dataset_array = []
        for i in data:
            info = {}
            file_name = i[0]
            file_path = os.path.join(data_root, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                # Note using 500 angstrom as box size
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                # Note using 500 angstrom as box size
                # Recommended install pytraj
                # conda install -c ambermd pytraj
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(
                    "File format not implemented", file_format
                )

            info["atoms"] = atoms.to_dict()
            info["jid"] = file_name

            tmp = [float(j) for j in i[1:]]  # float(i[1])
            if len(tmp) == 1:
                tmp = tmp[0]

            info["target"] = tmp  # float(i[1])
            dataset_array.append(info)
    else:
        dataset_array = None

    print('output_dir train', config.output_dir)
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        (
            train_loader,
            val_loader,
            test_loader,
            mean,
            std
        ) = get_train_val_loaders(
            dataset=config.dataset,
            root=config.output_dir,
            cachedir=config.cache_dir,
            processdir=config.process_dir,
            dataset_array=dataset_array,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            train_data_fraction=config.train_data_fraction,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            normalize=config.normalize,
            euclidean=config.euclidean,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            infinite_funcs=config.infinite_funcs,
            infinite_params=config.infinite_params,
            R=config.R,
            keep_data_order=config.keep_data_order,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        mean = 0.0
        std = 1.0

    # define network, optimizer, scheduler
    _model = {
        "potnet": PotNet,
    }
    config.model.euclidean = config.euclidean
    net = _model.get(config.model.name)(config.model)
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint)["model"])

    count_parameters(net)
    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        total_steps = max(1, config.epochs * steps_per_epoch)
        pct_start = config.warmup_steps / total_steps
        if pct_start >= 1.0:
            pct_start = 0.99
            print(
                "Clamping OneCycleLR pct_start to 0.99 because "
                f"warmup_steps={config.warmup_steps} exceeds total_steps={total_steps}."
            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.5
        )

    num_tasks = getattr(net, "num_outputs", 2)
    criterion = GradNormMultiTaskLoss(
        num_tasks=num_tasks,
        alpha=config.gradnorm_alpha,
        loss_name=config.criterion,
        ema_beta=config.gradnorm_ema_beta,
        reference_beta=config.gradnorm_reference_beta,
    ).to(device)
    gradnorm_optimizer = torch.optim.Adam(
        [criterion.task_weights],
        lr=config.gradnorm_lr or config.learning_rate,
    )
    shared_parameter_groups = get_shared_layer_parameter_groups(net)
    criterion.reset_initial_losses()


    # set up training engine and evaluators
    # metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std, "neg_mae": -1.0 * MeanAbsoluteError() * std}

    if isinstance(std, torch.Tensor):
        metric_scales = std.detach().cpu().tolist()
    elif isinstance(std, (list, tuple, np.ndarray)):
        metric_scales = list(std)
    else:
        metric_scales = [float(std)] * num_tasks

    if len(metric_scales) < num_tasks:
        metric_scales = metric_scales + [1.0] * (num_tasks - len(metric_scales))

    metrics = {
    "loss": Loss(criterion),
    "mae_task1": MultiTaskMAE(task_idx=0, scale=metric_scales[0]),
    "mae_task2": MultiTaskMAE(task_idx=1, scale=metric_scales[1]),
    "total_mae": MultiTaskMAE(task_idx=0, scale=metric_scales[0]) + MultiTaskMAE(task_idx=1, scale=metric_scales[1]),
    "neg_mae": -1.0 * (MultiTaskMAE(task_idx=0, scale=metric_scales[0]) + MultiTaskMAE(task_idx=1, scale=metric_scales[1]))
}

    trainer = create_gradnorm_trainer(
        net,
        optimizer,
        gradnorm_optimizer,
        criterion,
        prepare_batch=prepare_batch,
        shared_parameter_groups=shared_parameter_groups,
        warmup_epochs=config.gradnorm_warmup_epochs,
        device=device,
        deterministic=deterministic,
    )

    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "gradnorm_optimizer": gradnorm_optimizer,
            "loss_balancer": criterion,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=5,
            score_name="neg_mae",
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }
    history["train"]["task_weights"] = []
    history["validation"]["task_weights"] = []
    history["train"]["ema_task_losses"] = []
    history["validation"]["ema_task_losses"] = []
    history["train"]["ema_grad_norms"] = []
    history["validation"]["ema_grad_norms"] = []

    if config.store_outputs:
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics

            # 打印验证集结果
        print(f"Val MAE Task 1: {vmetrics['mae_task1']:.4f}")
        print(f"Val MAE Task 2: {vmetrics['mae_task2']:.4f}")
        print(f"Val Total MAE: {vmetrics['total_mae']:.4f}")
        current_weights = criterion.get_weights().cpu().tolist()
        current_ema_losses = criterion.ema_task_losses.detach().cpu().tolist()
        current_ema_grad_norms = criterion.ema_grad_norms.detach().cpu().tolist()
        if engine.state.warmup_active:
            print(f"GradNorm Warmup: epoch {engine.state.epoch}/{config.gradnorm_warmup_epochs}")
        elif engine.state.epoch == config.gradnorm_warmup_epochs + 1:
            print("GradNorm Warmup finished. Initial losses were reset after warmup.")
        print(f"GradNorm Weights: {[round(weight, 4) for weight in current_weights]}")
        print(f"GradNorm EMA Losses: {[round(loss, 4) for loss in current_ema_losses]}")
        print(f"GradNorm EMA GradNorms: {[round(norm, 4) for norm in current_ema_grad_norms]}")
        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()
            history["train"][metric].append(tmetrics[metric])
            history["validation"][metric].append(vmetrics[metric])
            # history["train"][metric].append(tm)
            # history["validation"][metric].append(vm)
        history["train"]["task_weights"].append(current_weights)
        history["validation"]["task_weights"].append(current_weights)
        history["train"]["ema_task_losses"].append(current_ema_losses)
        history["validation"]["ema_task_losses"].append(current_ema_losses)
        history["train"]["ema_grad_norms"].append(current_ema_grad_norms)
        history["validation"]["ema_grad_norms"].append(current_ema_grad_norms)

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, config.model.name + "_" + config.target + "_history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir,
                                      config.model.name + "_" + config.target + "_history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            # pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
            # pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")

    if config.n_early_stopping is not None:
        def default_score_fn(engine):
            score = engine.state.metrics["neg_mae"]
            return score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)

    if config.log_tensorboard:

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                # metric_names=["loss", "mae", "neg_mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    # train the model!
    if not testing:
        trainer.run(train_loader, max_epochs=config.epochs)

    if config.log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()

    print("Testing!")
    net.eval()
    t1 = time.time()
    mean_tensor = torch.as_tensor(mean, dtype=torch.get_default_dtype(), device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.get_default_dtype(), device=device)
    if config.write_predictions:
        f = open(
            os.path.join(config.output_dir,
                         config.model.name + "_" + config.target + "_prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
    targets = []
    predictions = []

    with torch.no_grad():
        ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
        ###新增加的
        targets_task1 = []  # 单独存储任务1的真实值
        targets_task2 = []  # 单独存储任务2的真实值
        preds_task1 = []    # 单独存储任务1的预测值
        preds_task2 = []    # 单独存储任务2的预测值
        ###新增加的ending
        for dat, id in tqdm(zip(test_loader, ids)):
            data = dat
            target = dat.label
            out_data = net(data.to(device))
            # print(out_data)
            # print(target)
            ###注释
           # out_data = out_data.cpu().numpy().tolist()
           # target = target.cpu().numpy().flatten().tolist()
            ###注释掉ending
            # pred_task1 = out_data[0].cpu().numpy().flatten()
            # pred_task2 = out_data[1].cpu().numpy().flatten()
        
            # # 分离两个任务的真实值
            # target_task1 = target[0].cpu().numpy().flatten()
            # target_task2 = target[1].cpu().numpy().flatten()
            
            out_data = out_data * std_tensor + mean_tensor
            pred_task1 = out_data[0, 0].detach().cpu().numpy().flatten()  # 取标量值
            pred_task2 = out_data[0, 1].detach().cpu().numpy().flatten()
            target_task1 = target[0].cpu().numpy().flatten()
            target_task2 = target[1].cpu().numpy().flatten()
            ##成功
            # pred_task1 = out_data[0, 0].item()  # 取标量值
            # pred_task2 = out_data[0, 1].item()
            # target_task1 = target[0].item()
            # target_task2 = target[1].item() 


            # preds_task1.append(pred_task1)
            # preds_task2.append(pred_task2)
            # targets_task1.append(target_task1)
            # targets_task2.append(target_task2)


        
            # 收集数据
            preds_task1.extend(pred_task1.tolist())
            preds_task2.extend(pred_task2.tolist())
            targets_task1.extend(target_task1.tolist())
            targets_task2.extend(target_task2.tolist())

            if isinstance(out_data, list) and len(out_data) == 1:
                # out_data = out_data[0]
                out_data = out_data.view(-1, 2)
            if len(target) == 1:
                # target = target[0]
                target = target.view(-1, 2)
            # if config.write_predictions:
            #     f.write("%s, %6f, %6f\n" % (id, target, out_data))

            targets.append(target)
            predictions.append(out_data)
    if config.write_predictions:
        f.close()
    from sklearn.metrics import mean_absolute_error
    mae_task1 = mean_absolute_error(targets_task1, preds_task1)
    mae_task2 = mean_absolute_error(targets_task2, preds_task2)
    total_mae = (mae_task1 + mae_task2)  
    
    t2 = time.time()

    print(f"Test MAE Task 1: {mae_task1:.4f}")
    print(f"Test MAE Task 2: {mae_task2:.4f}")
    print(f"Test Total MAE: {total_mae:.4f}")


    print("Test time(s):", t2 - t1)


    #后来注释掉的
    # from sklearn.metrics import mean_absolute_error

    # print(
    #     "Test MAE:",
    #     mean_absolute_error(np.array(targets), np.array(predictions)),
    # )
    #### return mean_absolute_error(np.array(targets), np.array(predictions))
    return total_mae


def train_prop_model(config: Dict, data_root: str = None, checkpoint: str = None, testing: bool = False, file_format: str = 'poscar'):
    if config["dataset"] == "megnet":
        config["id_tag"] = "id"
        if config["target"] == "e_form" or config["target"] == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4239

    result = train_pyg(config, data_root=data_root, file_format=file_format, checkpoint=checkpoint, testing=testing)
    return result
