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

def create_train_state(rng, learning_rate, weight_decay, total_steps, input_shape):
    model = CausalTransformer(vocab_size=8,
                              dim=10,
                              num_layers=2,
                              num_heads=1,
                              ff_dim=10,
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
def generate_step(state, batch):
    batch_size, seq_len = batch["input_ids"].shape
    loss_mask = batch["loss_mask"]
    labels = batch["labels"]

    # Initialize our working output to the ground truth labels.
    # We will dynamically overwrite the masked (generated) positions.
    gen_labels = labels
    has_eos = jnp.zeros((batch_size,), dtype=jnp.bool_)

    def body_fn(t, val):
        current_gen_labels, current_has_eos = val

        # Construct the input sequence for the current step t.
        # input_ids[:, 0:1] is the BOS token.
        # current_gen_labels[:, :-1] acts as the rest of the input sequence.
        current_input = jnp.concatenate(
            [batch["input_ids"][:, 0:1], current_gen_labels[:, :-1]], axis=1
        )

        # Forward pass (computing logits for the whole sequence up to t)
        logits = state.apply_fn({"params": state.params}, current_input)

        # Greedily pick the predicted next token for position t
        next_token = jnp.argmax(logits[:, t], axis=-1)

        # Check if we are currently in the generation phase for this batch element
        is_gen_step = (loss_mask[:, t] == 1)

        # Update EOS tracker (Token 6 is EOS)
        new_has_eos = current_has_eos | (is_gen_step & (next_token == 6))

        # If we've already hit EOS in a previous step, force padding (Token 7)
        # Otherwise, use the model's prediction.
        token_to_write = jnp.where(current_has_eos, 7, next_token)

        # Overwrite the generated token, ONLY if this is a generation step.
        # If it's a prompt step (mask==0), it correctly leaves the ground truth alone.
        updated_gen_labels = current_gen_labels.at[:, t].set(
            jnp.where(is_gen_step, token_to_write, current_gen_labels[:, t])
        )

        return updated_gen_labels, new_has_eos

    # Run the autoregressive loop
    final_gen_labels, final_has_eos = jax.lax.fori_loop(
        0, seq_len, body_fn, (gen_labels, has_eos)
    )

    # -----------------------------
    # Metrics Calculation
    # -----------------------------
    correct_map = (final_gen_labels == labels)

    # 1. char_acc: Accuracy on the generated tokens only
    char_acc = jnp.sum(correct_map * loss_mask) / jnp.sum(loss_mask)

    # 2. seq_acc: Exact match for the entire generated sequence. 
    # If loss_mask == 0, we consider it automatically correct (we only judge the generation).
    seq_acc = jnp.mean(
        jnp.all(jnp.logical_or(correct_map, (loss_mask == 0)), axis=1)
    )

    # 3. num_sequences_without_eos: Count of sequences that never generated token 6
    num_sequences_without_eos = jnp.sum(~final_has_eos)

    metrics = {
        "char_acc": char_acc,
        "sequence_acc": seq_acc,
        "num_sequences_without_eos": num_sequences_without_eos
    }

    return final_gen_labels, metrics


def debug_generate(state, opening_str, tok2id, id2tok, max_new_tokens=30):
    # Prepare the prompt
    tokens = [tok2id["BOS"]] + [tok2id[c] for c in opening_str] + [tok2id["SEP"]]

    print("--- Debugging Generation ---")
    print(f"Input String: {opening_str}")
    print(f"Prompt IDs:   {tokens}")
    print("-" * 30)

    generated_indices = []

    for i in range(max_new_tokens):
        input_tensor = jnp.array([tokens], dtype=jnp.int32)

        # Forward pass
        logits = state.apply_fn({"params": state.params}, input_tensor)

        # Get logits for the LAST token
        next_token_logits = logits[0, -1, :]

        # Get top 3 candidates
        top_k_vals, top_k_ids = jax.lax.top_k(next_token_logits, k=3)

        # Pick the winner
        next_id = int(top_k_ids[0])

        # FIX: Remove str() conversion
        next_char = id2tok[next_id]

        # Print process
        candidates = ", ".join([f"'{id2tok[int(idx)]}' ({val:.2f})"
                                for idx, val in zip(top_k_ids, top_k_vals)])
        print(f"Step {i+1}: Picked '{next_char}' | Candidates: {candidates}")

        if next_id == tok2id["EOS"]:
            print("--- Hit EOS ---")
            break

        tokens.append(next_id)
        generated_indices.append(next_id)

    res_str = "".join([id2tok[idx] for idx in generated_indices if id2tok[idx] in [")", "]"]])
    print("-" * 30)
    print(f"Final Generated Sequence: {res_str}")
    return res_str
# Usage:
# debug_generate(state, "([", token_to_id, id_to_token)


def evaluate():
    unmask_ids = [2, 3, 6]
    batch_size = 1024
    learning_rate = 4e-4
    weight_decay = 0.0

    test_source = BracketDataSource("/home/ashwin/Python/JAX/bracket_lm/processed_data/test_32.npy", unmask_ids)

    test_sampler = grain.IndexSampler(
        num_records=len(test_source),
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        num_epochs=1,
        shuffle=False,
    )

    test_loader = grain.DataLoader(
        data_source=test_source,
        sampler=test_sampler,
        worker_count=4,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=False)
        ]
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1, 63), dtype=jnp.int32)

    num_test_steps = len(test_sampler) // batch_size
    state = create_train_state(init_rng, learning_rate, weight_decay,
                               num_test_steps, dummy_input.shape)
    ckptr = ocp.Checkpointer(
        ocp.CompositeCheckpointHandler()
    )

    restored = ckptr.restore(
        Path(__file__).resolve().parent
        / "bracketlm_checkpoints"
        / "tiny_bracket_lm_distilled_2_100",
        args=ocp.args.Composite(
            default=ocp.args.PyTreeRestore(
                item={"params": state.params},
                partial_restore=True,
            ),
        ),
    )

    state = state.replace(
        params=restored["default"]["params"]
    )

    num_batches = len(test_sampler) // batch_size

    test_char_acc = 0.0
    test_seq_acc = 0.0
    test_eos_miss = 0.0
    # print(debug_generate(state, "((([(([(((((([([", tok2id, id2tok))

    pbar = tqdm(test_loader, total=num_batches, desc="testing")
    for batch_idx, batch in enumerate(pbar, start=1):
        _, metrics = generate_step(state, batch)

        char_acc = float(metrics["char_acc"])
        seq_acc = float(metrics["sequence_acc"])
        eos_miss = float(metrics["num_sequences_without_eos"])

        test_char_acc += char_acc
        test_seq_acc += seq_acc
        test_eos_miss += eos_miss

        pbar.set_postfix({
            "batch": f"{batch_idx}/{num_batches}",
            "char_acc": f"{char_acc:.4f}",
            "seq_acc": f"{seq_acc:.4f}",
        })

    avg_test_char_acc = test_char_acc / num_test_steps
    avg_test_seq_acc = test_seq_acc / num_test_steps
    avg_eos_miss = test_eos_miss / (num_test_steps * batch_size)

    print(
        f"test | "
        f"char_acc: {avg_test_char_acc} | "
        f"seq_acc: {avg_test_seq_acc} | "
        f"eos miss: {avg_eos_miss}"
    )


if __name__ == "__main__":
    evaluate()
    # test | loss: 3.96954030179586e-05 | char_acc: 1.0 | seq_acc: 1.0
