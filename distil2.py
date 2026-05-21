import os
import pandas as pd
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
import grain.python as grain
from tqdm import tqdm
from pathlib import Path

from train_dataloader import BracketDataSource
from transformer import CausalTransformer

def create_train_state(model, rng, learning_rate, weight_decay, input_shape):
    variables = model.init(rng, jnp.ones(input_shape, dtype=jnp.uint8))
    params = variables["params"]

    # ---- build mask here ----
    def decay_mask_fn(params):
        def mask_fn(path, v):
            path_str = jax.tree_util.keystr(path, simple=True, separator="/")
            return not ("bias" in path_str or "scale" in path_str)
        return jax.tree_util.tree_map_with_path(mask_fn, params)

    decay_mask = decay_mask_fn(params)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mask=decay_mask,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
    )

    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params,
                                         tx=tx)

@jax.jit
def train_step(
    state,
    batch,
    teacher_state,
    temperature=2.0,
    kl_weight=0.99,
    ce_weight=0.01,
):

    x = batch["input_ids"]
    y = batch["labels"]
    loss_mask = batch["loss_mask"]

    # Frozen teacher forward pass
    teacher_logits = teacher_state.apply_fn(
        {"params": teacher_state.params},
        x,
    )

    def loss_fn(params):

        student_logits = state.apply_fn(
            {"params": params},
            x,
        )

        teacher_probs = jax.nn.softmax(
            teacher_logits / temperature,
            axis=-1,
        )

        student_log_probs = jax.nn.log_softmax(
            student_logits / temperature,
            axis=-1,
        )

        kl_per_token = jnp.sum(
            teacher_probs * (
                jnp.log(teacher_probs + 1e-9)
                - student_log_probs
            ),
            axis=-1,
        )

        masked_kl = kl_per_token * loss_mask

        kl_loss = (
            jnp.sum(masked_kl)
            / jnp.sum(loss_mask)
        ) * (temperature ** 2)

        ce_per_token = (
            optax.softmax_cross_entropy_with_integer_labels(
                student_logits,
                y,
            )
        )

        masked_ce = ce_per_token * loss_mask

        ce_loss = (
            jnp.sum(masked_ce)
            / jnp.sum(loss_mask)
        )

        loss = (
            kl_weight * kl_loss
            +
            ce_weight * ce_loss
        )

        aux = (
            kl_loss,
            ce_loss,
            student_logits,
        )

        return loss, aux

    grad_fn = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )

    (loss, (kl_loss, ce_loss, logits)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    pred = jnp.argmax(logits, axis=-1)

    correct_map = (pred == y)

    sequence_acc = jnp.mean(
        jnp.all(
            jnp.logical_or(
                correct_map,
                (loss_mask == 0),
            ),
            axis=1,
        )
    )

    char_acc = (
        jnp.sum(correct_map * loss_mask)
        / jnp.sum(loss_mask)
    )

    metrics = {
        "loss": loss,
        "sequence_acc": sequence_acc,
        "char_acc": char_acc,
        "kl_loss": kl_loss,
        "ce_loss": ce_loss,
    }

    return state, metrics


@jax.jit
def eval_step(
    state,
    batch,
    teacher_state,
    temperature=2.0,
    kl_weight=0.99,
    ce_weight=0.01,
):

    x = batch["input_ids"]
    y = batch["labels"]
    loss_mask = batch["loss_mask"]

    teacher_logits = teacher_state.apply_fn(
        {"params": teacher_state.params},
        x,
    )

    student_logits = state.apply_fn(
        {"params": state.params},
        x,
    )

    # -------------------------
    # Distillation KL loss
    # -------------------------

    teacher_probs = jax.nn.softmax(
        teacher_logits / temperature,
        axis=-1,
    )

    student_log_probs = jax.nn.log_softmax(
        student_logits / temperature,
        axis=-1,
    )

    kl_per_token = jnp.sum(
        teacher_probs * (
            jnp.log(teacher_probs + 1e-9)
            - student_log_probs
        ),
        axis=-1,
    )

    masked_kl = kl_per_token * loss_mask

    kl_loss = (
        jnp.sum(masked_kl)
        / jnp.sum(loss_mask)
    ) * (temperature ** 2)

    ce_per_token = (
        optax.softmax_cross_entropy_with_integer_labels(
            student_logits,
            y,
        )
    )

    masked_ce = ce_per_token * loss_mask

    ce_loss = (
        jnp.sum(masked_ce)
        / jnp.sum(loss_mask)
    )

    loss = (
        kl_weight * kl_loss
        +
        ce_weight * ce_loss
    )

    pred = jnp.argmax(student_logits, axis=-1)

    correct_map = (pred == y)

    sequence_acc = jnp.mean(
        jnp.all(
            jnp.logical_or(
                correct_map,
                (loss_mask == 0),
            ),
            axis=1,
        )
    )

    char_acc = (
        jnp.sum(correct_map * loss_mask)
        / jnp.sum(loss_mask)
    )

    metrics = {
        "loss": loss,
        "kl_loss": kl_loss,
        "ce_loss": ce_loss,
        "sequence_acc": sequence_acc,
        "char_acc": char_acc,
    }

    return metrics

def train():
    train_file = "/home/ashwin/Python/JAX/bracket_lm/processed_data/train.npy"
    val_file = "/home/ashwin/Python/JAX/bracket_lm/processed_data/val.npy"
    unmask_ids = [2, 3, 6]
    batch_size = 128
    epochs = 100
    learning_rate = 1e-5
    weight_decay = 0.0

    train_source = BracketDataSource(train_file, unmask_ids=unmask_ids)
    val_source = BracketDataSource(val_file, unmask_ids=unmask_ids)

    train_sampler = grain.IndexSampler(
        num_records=len(train_source),
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        num_epochs=1,
        shuffle=True,
        seed=0,
    )

    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,
        worker_count=4,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    val_sampler = grain.IndexSampler(
        num_records=len(val_source),
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        num_epochs=1,
        shuffle=False,
    )

    val_loader = grain.DataLoader(
        data_source=val_source,
        sampler=val_sampler,
        worker_count=4,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=False)
        ]
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1, 63), dtype=jnp.int32)
    ckpt_path = os.path.abspath("bracketlm_checkpoints")
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True,
                                           step_prefix="tiny_bracket_lm_distilled_2")

    num_val_steps = len(val_sampler) // batch_size

    student = CausalTransformer(vocab_size=8,
                                dim=10,
                                num_layers=2,
                                num_heads=1,
                                ff_dim=10,
                                PAD_ID=7)

    student_state = create_train_state(student, init_rng, learning_rate,
                                       weight_decay,
                                       dummy_input.shape)

    ckptr = ocp.Checkpointer(
        ocp.CompositeCheckpointHandler()
    )

    restored = ckptr.restore(
        Path(__file__).resolve().parent
        / "bracketlm_checkpoints"
        / "tiny_bracket_lm_distilled_286",
        args=ocp.args.Composite(
            default=ocp.args.PyTreeRestore(
                item={"params": student_state.params},
                partial_restore=True,
            ),
        ),
    )

    student_state = student_state.replace(
        params=restored["default"]["params"]
    )

    teacher = CausalTransformer(vocab_size=8,
                                dim=64,
                                num_layers=2,
                                num_heads=4,
                                ff_dim=128,
                                PAD_ID=7)

    teacher_state = create_train_state(teacher, init_rng, learning_rate,
                                       weight_decay,
                                       dummy_input.shape)

    restored = ckptr.restore(
        Path(__file__).resolve().parent
        / "bracketlm_checkpoints"
        / "bracket_lm_15",
        args=ocp.args.Composite(
            default=ocp.args.PyTreeRestore(
                item={"params": teacher_state.params},
                partial_restore=True,
            ),
        ),
    )

    teacher_state = teacher_state.replace(
        params=restored["default"]["params"]
    )

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(
        student_state.params))
    print(f"Parameter Count: {param_count/1e3:.2f}k")

    flop_analysis = jax.jit(student_state.apply_fn).lower(
        {"params": student_state.params},
        dummy_input,
    ).cost_analysis()

    flops = flop_analysis[0]["flops"] if isinstance(flop_analysis, list)\
        else flop_analysis.get("flops", 0)
    print(f"FLOPs (per forward pass): {flops/1e6:.4f}M")

    best_val_loss = float("inf")

    os.makedirs('logs', exist_ok=True)
    log_path = "logs/tiny_bracket_lm_distillation_metrics2.csv"
    df = pd.DataFrame(columns=[
        "epoch",
        "train_loss",
        "train_kl_loss",
        "train_ce_loss",
        "train_char_acc",
        "train_sequence_acc",
        "val_loss",
        "val_kl_loss",
        "val_ce_loss",
        "val_char_acc",
        "val_sequence_acc"
    ])
    df.to_csv(log_path, index=False)

    steps_per_epoch = len(train_source) // batch_size

    for epoch in range(epochs):

        running_loss = 0.0
        running_kl_loss = 0.0
        running_ce_loss = 0.0
        running_char_acc = 0.0
        running_seq_acc = 0.0

        pbar = tqdm(
                train_loader,
                total=steps_per_epoch,
                desc=f"epoch {epoch+1}/{epochs}",
        )

        for step, batch in enumerate(pbar):
            student_state, metrics = train_step(student_state,
                                                batch,
                                                teacher_state)

            loss = float(metrics["loss"])
            kl_loss = float(metrics["kl_loss"])
            ce_loss = float(metrics["ce_loss"])
            char_acc = float(metrics["char_acc"])
            seq_acc = float(metrics["sequence_acc"])

            running_loss += loss
            running_kl_loss += kl_loss
            running_ce_loss += ce_loss
            running_char_acc += char_acc
            running_seq_acc += seq_acc

            pbar.set_postfix({
                "loss": loss,
                "char_acc": char_acc,
            })

        # epoch averages
        avg_loss = running_loss / steps_per_epoch
        avg_kl_loss = running_kl_loss / steps_per_epoch
        avg_ce_loss = running_ce_loss / steps_per_epoch
        avg_char_acc = running_char_acc / steps_per_epoch
        avg_seq_acc = running_seq_acc / steps_per_epoch

        print(
            f"epoch {epoch+1}/{epochs} | "
            f"loss: {avg_loss:.4f} | "
            f"kl loss: {avg_kl_loss:.4f} | "
            f"ce loss: {avg_ce_loss:.4f} | "
            f"char_acc: {avg_char_acc:.4f} | "
            f"seq_acc: {avg_seq_acc:.4f}"
        )

        val_loss = 0.0
        val_kl_loss = 0.0
        val_ce_loss = 0.0
        val_char_acc = 0.0
        val_seq_acc = 0.0

        for batch in val_loader:
            metrics = eval_step(student_state, batch, teacher_state)

            val_loss += float(metrics["loss"])
            val_kl_loss += float(metrics["kl_loss"])
            val_ce_loss += float(metrics["ce_loss"])
            val_char_acc += float(metrics["char_acc"])
            val_seq_acc += float(metrics["sequence_acc"])

        avg_val_loss = val_loss / num_val_steps
        avg_val_kl_loss = val_kl_loss / num_val_steps
        avg_val_ce_loss = val_ce_loss / num_val_steps
        avg_val_char_acc = val_char_acc / num_val_steps
        avg_val_seq_acc = val_seq_acc / num_val_steps

        print(
            f"val | "
            f"loss: {avg_val_loss:.4f} | "
            f"kl loss: {avg_val_kl_loss:.4f} | "
            f"ce loss: {avg_val_ce_loss:.4f} | "
            f"char_acc: {avg_val_char_acc:.4f} | "
            f"seq_acc: {avg_val_seq_acc:.4f}"
        )

        if (avg_val_loss < best_val_loss):
            best_val_loss = avg_val_loss
            with ocp.CheckpointManager(ckpt_path, options=options) as mngr:
                mngr.save(
                    epoch+1,
                    args=ocp.args.StandardSave(student_state)
                )
            print(f"New best validation loss! Checkpoint saved to {ckpt_path}")

        row = pd.DataFrame([{
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "train_kl_loss": avg_kl_loss,
            "train_ce_loss": avg_ce_loss,
            "train_char_acc": avg_char_acc,
            "train_sequence_acc": avg_seq_acc,
            "val_loss": avg_val_loss,
            "val_kl_loss": avg_val_kl_loss,
            "val_ce_loss": avg_val_ce_loss,
            "val_char_acc": avg_val_char_acc,
            "val_seq_acc": avg_val_seq_acc,
        }])
        row.to_csv(
            log_path,
            mode="a",
            header=False,
            index=False
        )


if __name__ == "__main__":
    train()
