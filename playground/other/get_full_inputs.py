"""Benchmark one case of intra-op only parallelism."""
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from get_wikitext import load_wikitext2, prepare_batch
from test_export_hlo import create_train_state, get_train_step, compute_gpt_parameter_count

import alpa
from alpa import parallelize, global_config, set_parallelize_options, LocalPhysicalDeviceMesh
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule, TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.util import map_to_shape, count_communication_primitives, print_used_time, GB

import os

as_option = global_config.default_autosharding_option

ds = None

DIR_NAME = "inputs_test/"

BATCH_INPUTS_SIZE = 5


def benchmark_2d_one_case_gpt_bert(physical_mesh, model_type, benchmark_case):
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

    logical_mesh = physical_mesh.get_logical_mesh([l_dim0, l_dim1])
    set_parallelize_options(devices=logical_mesh, num_micro_batches=num_micro_batches)

    print_used_time("Setup device mesh")

    # Prepare input batch
    global ds
    if ds == None:
        ds = load_wikitext2(batch_size, seq_len)
    batch = next(ds)
    prepare_batch(batch)

    print_used_time("Prepare input")

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

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch, dtype)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, dtype)
    executable, args_flat = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    return executable, args_flat


def main(iter=0):
    model_type = "gpt"

    num_nodes = 1  # machines
    num_devices_per_node = 2  # cores
    _ = None

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
        1,  128,  1024,  24,  32,    25600,   num_nodes,  num_devices_per_node,
        # 1.3B
        # 1,  128,  2048,  24,  16,    32032, num_nodes, num_devices_per_node, 
        #_,_,  PP,  NB, FM,   Remat, RS,    _  _
        _, _,  1,   1,  True, False, False, _, _)

    num_layers, hidden_size, vocab_size = (benchmark_case[3], benchmark_case[2],
                                           benchmark_case[5])
    param_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                              vocab_size)
    # print(f"Param count: {param_count/1e9:.2f} B")
    print(f"Param count: {param_count}")

    # Device a fake physical mesh
    num_devices = num_nodes * num_devices_per_node
    physical_mesh = LocalPhysicalDeviceMesh(devices=[None] * num_devices)

    # Compile a mesh executable
    executable, args_flat = benchmark_2d_one_case_gpt_bert(physical_mesh, model_type, benchmark_case)

    # Checking stuff, ignore these
    # # print(sharded_args[4].device_buffers)

    print(f"Write args to {DIR_NAME}")
    sharded_args = executable.preshard_dynamic_args(*args_flat)
    num_inputs = len(sharded_args)
    if iter > 0:
        sharded_args = sharded_args[num_inputs - BATCH_INPUTS_SIZE:]
    cwd = os.getcwd()
    main_dir = os.path.join(DIR_NAME, "iter_" + str(iter))
    partition_dir_name="partition_id"
    ins_folder = os.path.join(cwd, main_dir)
    partition_id_folder = os.path.join(ins_folder, partition_dir_name)
    if not os.path.exists(ins_folder):
        os.makedirs(ins_folder)
    if not os.path.exists(partition_id_folder):
        os.makedirs(partition_id_folder)
    for i, arg in enumerate(sharded_args):
        if iter > 0:
            i += num_inputs - BATCH_INPUTS_SIZE
        # print(i, ":", type(arg), arg.shape)
        for j, device in enumerate(arg.device_buffers):
            device_folder = os.path.join(ins_folder, "dev_" + str(j))
            if not os.path.exists(device_folder):
                os.makedirs(device_folder)
            arg_path = os.path.join(device_folder, str(i))
            # print(i, ":", f"dev_{j}", ":", type(arg.device_buffers[j]), arg.device_buffers[j].shape)
            # print(arg.device_buffers[j])
            jnp.save(arg_path, arg.device_buffers[j])
    for i in range(32):
        partition_id = jnp.array([i], dtype=np.uint32)
        arg_path = os.path.join(partition_id_folder, "partition_id_" + str(i))
        jnp.save(arg_path, partition_id)

    print("DONE")

if __name__ == "__main__":
    i = 0
    while True:
        print("Generating iter:", i)
        try:
            main(i)
        except StopIteration:
            break
        i += 1
