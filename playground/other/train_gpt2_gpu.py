"""Train gpt2 with alpa on gpu."""

from math import isnan
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
# import matplotlib.pyplot as plt
from get_wikitext import load_wikitext2, prepare_batch
from test_export_hlo import create_train_state, get_train_step, compute_gpt_parameter_count

import alpa
from alpa import parallelize, global_config, set_parallelize_options, LocalPhysicalDeviceMesh
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule, TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.util import map_to_shape, count_communication_primitives, print_used_time, GB

import pickle
import time

NUM_EPOCHS = 3

as_option = global_config.default_autosharding_option


def train_loop_gpt_bert(model_type, benchmark_case):
    print_used_time(None)

    # Model configs
    (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
     l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, num_micro_batches, force_batch_dim_mapping,
     use_remat, prefer_reduce_scatter, other, overwrite_global_config_dict) = benchmark_case
 
    dtype = jnp.float16

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = alpa.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    if force_batch_dim_mapping:
        # Always map batch dim to mesh dim 0
        as_option.force_batch_dim_to_mesh_dim = 0
    as_option.prefer_reduce_scatter = prefer_reduce_scatter

    if other == "zero-3":
        as_option.force_zero_stage_3 = True
    elif other in ["shard-largest"]:
        as_option.force_simple_heuristic = other
        global_config.remat_using_while = True


    print_used_time("Setup device mesh")

    # Init train state
    if model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ), dtype=dtype)
    elif model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")


    train_step = get_train_step(grad_func, num_layers, dtype)

    # Run
    losses = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Prepare input batch
        batches = load_wikitext2(batch_size, seq_len)
        # print_used_time("Prepare input")

        rngkey = jax.random.PRNGKey(0)
        batch = next(batches)
        prepare_batch(batch)
        if epoch == 0:
            state = create_train_state(rngkey, model, batch, dtype)
        # print_used_time("Create train state")
        
        i = 0
        for batch in batches:
            prepare_batch(batch)
            result = train_step(state, batch, rngkey)
            loss, state = result[0], result[1]
            if i % 1 == 0:
                print(i, "- LOSS:", loss)
            if np.isnan(loss):
                raise Exception(f"NaN found in epoch {epoch} iter {i}")
            losses.append(float(loss))
            i += 1

        epoch_time = time.time() - start_time
        print(f"EPOCH {epoch} -", f"TIME: {epoch_time}s", "- LOSS:", loss)

    print_used_time("Run")

    return losses



if __name__ == "__main__":
    model_type = "gpt"

    _ = None
    num_nodes = _  # machines
    num_devices_per_node = _  # cores

    # Define a model with 1.3B parameters

    # B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
    # #head = num_heads, LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
    # PD0 = physical_mesh_dimension_0, PD1 = physical_mesh_dimension_1,
    # NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
    # RS = prefer_reduce_scatter
    benchmark_case = (
        #B, S,     H      L,  #head, V,     LD0,       LD1,  
        # hidden size and vocab size should be multiples of num_devices
        # 12M testing
        # 16,  512,  512,  2,  32,    12800, num_nodes, num_devices_per_node, 
        # 300M
        # 8,  128,  1024,  24,  32,    25600,   _,         _,
        # 1.3B
        1,  128,  2048,  24,  16,    32032, num_nodes, num_devices_per_node, 
        #_,_,  PP,  NB, FM,   Remat, RS,    _  _
        _, _,  1,   1,  True, False, False, _, _)

    num_layers, hidden_size, vocab_size = (benchmark_case[3], benchmark_case[2],
                                           benchmark_case[5])
    param_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                              vocab_size)
    # print(f"Param count: {param_count/1e9:.2f} B")
    print(f"Param count: {param_count}")

    # Run train loop for gpt2
    losses = train_loop_gpt_bert(model_type, benchmark_case)
    # print(losses)

    # plt.plot(range(len(losses)), losses)

    with open("losses", "wb") as fp:   #Pickling
        pickle.dump(losses, fp)

