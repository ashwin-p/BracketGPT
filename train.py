import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
import grain.python as grain
from tqdm import tqdm

from train_dataloader import BracketDataSource
from transformer import CausalTransformer

def create_train_state(rng, learning_rate, weight_decay, total_steps, input_shape):
    model = CausalTransformer(vocab_size=8,
                              dim=64,
                              num_layers=2,
                              num_heads=4,
                              ff_dim=128,
                              PAD_ID=7)
    variables = model.init(rng, jnp.ones(input_shape, dtype=jnp.uint8))
    params = variables["params"]

    warmup_steps = int(0.05 * total_steps)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0
    )

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
            learning_rate=lr_schedule,
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
def train_step(state, batch):
    x = batch["input_ids"]
    y = batch["labels"]
    loss_mask = batch["loss_mask"]

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            x,
        )
        loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits,
                                                                         y)
        masked_loss = loss_per_token * loss_mask
        loss = jnp.sum(masked_loss) / jnp.sum(loss_mask)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    pred = jnp.argmax(logits, -1)
    correct_map = (pred == y)
    sequence_acc = jnp.mean(
        jnp.all(jnp.logical_or(correct_map, (loss_mask == 0)), axis=1)
    )
    char_acc = jnp.sum(correct_map * loss_mask) / jnp.sum(loss_mask)
    metrics = {
        "loss": loss,
        "sequence_acc": sequence_acc,
        "char_acc": char_acc
    }
    return state, metrics


@jax.jit
def eval_step(state, batch):
    x = batch["input_ids"]
    y = batch["labels"]
    loss_mask = batch["loss_mask"]

    logits = state.apply_fn(
        {"params": state.params},
        x
    )
    loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits,
                                                                         y)
    masked_loss = loss_per_token * loss_mask
    loss = jnp.sum(masked_loss) / jnp.sum(loss_mask)
    pred = jnp.argmax(logits, -1)
    correct_map = (pred == y)
    sequence_acc = jnp.mean(
        jnp.all(jnp.logical_or(correct_map, (loss_mask == 0)), axis=1)
    )
    char_acc = jnp.sum(correct_map * loss_mask) / jnp.sum(loss_mask)
    metrics = {
        "loss": loss,
        "sequence_acc": sequence_acc,
        "char_acc": char_acc
    }
    return metrics

def train():
    train_file = "/home/ashwin/Python/JAX/bracket_lm/processed_data/train.npy"
    val_file = "/home/ashwin/Python/JAX/bracket_lm/processed_data/val.npy"
    unmask_ids = [2, 3, 6]
    batch_size = 128
    epochs = 15
    learning_rate = 4e-4
    weight_decay = 0.001

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
    total_steps = (len(train_sampler) // batch_size) * epochs
    ckpt_path = os.path.abspath("bracketlm_checkpoints")
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True,
                                           step_prefix="bracket_lm")

    num_val_steps = len(val_sampler) // batch_size
    state = create_train_state(init_rng, learning_rate, weight_decay,
                               total_steps, dummy_input.shape)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Parameter Count: {param_count/1e3:.2f}k")

    flop_analysis = jax.jit(state.apply_fn).lower(
        {"params": state.params},
        dummy_input,
    ).cost_analysis()

    flops = flop_analysis[0]["flops"] if isinstance(flop_analysis, list)\
        else flop_analysis.get("flops", 0)
    print(f"FLOPs (per forward pass): {flops/1e6:.4f}M")

    best_val_loss = float("inf")

    os.makedirs('logs', exist_ok=True)
    log_path = "logs/bracket_lm_train_metrics.csv"
    df = pd.DataFrame(columns=[
        "epoch",
        "train_loss",
        "train_char_acc",
        "train_sequence_acc",
        "val_loss",
        "val_char_acc",
        "val_sequence_acc"
    ])
    df.to_csv(log_path, index=False)

    steps_per_epoch = len(train_source) // batch_size

    for epoch in range(epochs):

        running_loss = 0.0
        running_char_acc = 0.0
        running_seq_acc = 0.0

        pbar = tqdm(
                train_loader,
                total=steps_per_epoch,
                desc=f"epoch {epoch+1}/{epochs}",
        )

        for step, batch in enumerate(pbar):
            state, metrics = train_step(state, batch)

            loss = float(metrics["loss"])
            char_acc = float(metrics["char_acc"])
            seq_acc = float(metrics["sequence_acc"])

            running_loss += loss
            running_char_acc += char_acc
            running_seq_acc += seq_acc

            pbar.set_postfix({
                "loss": loss,
                "char_acc": char_acc,
            })

        # epoch averages
        avg_loss = running_loss / steps_per_epoch
        avg_char_acc = running_char_acc / steps_per_epoch
        avg_seq_acc = running_seq_acc / steps_per_epoch

        print(
            f"epoch {epoch+1}/{epochs} | "
            f"loss: {avg_loss:.4f} | "
            f"char_acc: {avg_char_acc:.4f} | "
            f"seq_acc: {avg_seq_acc:.4f}"
        )

        val_loss = 0.0
        val_char_acc = 0.0
        val_seq_acc = 0.0

        for batch in val_loader:
            metrics = eval_step(state, batch)

            val_loss += float(metrics["loss"])
            val_char_acc += float(metrics["char_acc"])
            val_seq_acc += float(metrics["sequence_acc"])

        avg_val_loss = val_loss / num_val_steps
        avg_val_char_acc = val_char_acc / num_val_steps
        avg_val_seq_acc = val_seq_acc / num_val_steps

        print(
            f"val | "
            f"loss: {avg_val_loss:.4f} | "
            f"char_acc: {avg_val_char_acc:.4f} | "
            f"seq_acc: {avg_val_seq_acc:.4f}"
        )

        if (avg_val_loss < best_val_loss):
            best_val_loss = avg_val_loss
            with ocp.CheckpointManager(ckpt_path, options=options) as mngr:
                mngr.save(
                    epoch+1,
                    args=ocp.args.StandardSave(state)
                )
            print(f"New best validation loss! Checkpoint saved to {ckpt_path}")

        row = pd.DataFrame([{
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "train_char_acc": avg_char_acc,
            "train_sequence_acc": avg_seq_acc,
            "val_loss": avg_val_loss,
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
