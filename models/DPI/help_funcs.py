import torch
import math

# def add_uniform_noise_to_all(attend_logits, noise_scale=0.01):
#     noise = torch.rand_like(attend_logits) * noise_scale
#     attend_logits = attend_logits + noise
#     return attend_logits

# def add_gaussian_noise_to_all(attend_logits, noise_mean=0.0, noise_std=0.01):
#     noise = torch.randn_like(attend_logits) * noise_std + noise_mean
#     attend_logits = attend_logits + noise
#     return attend_logits

def add_uniform_noise_to_rel_coef(attend_logits, rel_coef, addon_relcoef = 0, scale_max = 0, scale_std = 0, std_base = 1.0):
    if scale_max == 0:
        noise_max = 1
    elif scale_max == 1:
        noise_max = attend_logits.max(dim=-1,keepdim=True)[0]

    if scale_std == 0:
        noise_std = 1
    elif scale_std == 1:
        distance_ids = torch.arange(0,rel_coef.shape[-1]).unsqueeze(0).to(attend_logits.device)
        distance_matrix = distance_ids.unsqueeze(2) - distance_ids.unsqueeze(1)
        noise_std = distance_matrix[:, -rel_coef.shape[1]:, :]/rel_coef.shape[-1] * std_base

    # print("attend_logits ",attend_logits.shape,noise_std.shape,noise_max.shape)
    noise = torch.rand_like(attend_logits) * noise_std * noise_max
    if addon_relcoef == 0:
        attend_logits.add_(noise)
    else:
        mask = (rel_coef != 0)
        noise.mul_(mask)
        attend_logits.add_(noise)
    return attend_logits
 
def add_gaussian_noise_to_rel_coef(attend_logits, rel_coef, addon_relcoef = 0, scale_mean = 0, scale_std = 0, std_base = 1.0):
    if scale_mean == 0:
        noise_mean = 0
    elif scale_mean == 1:
        noise_mean = attend_logits.mean(dim=-1,keepdim=True)
    qlen = rel_coef.shape[1]
    if scale_std == 0:
        noise_std = 1
    elif scale_std == 1:
        distance_ids = torch.arange(0,rel_coef.shape[-1], dtype=attend_logits.dtype).unsqueeze(0).to(attend_logits.device)
        distance_matrix = distance_ids[:,-qlen:].unsqueeze(2) - distance_ids.unsqueeze(1)
        noise_std = distance_matrix/rel_coef.shape[-1] * std_base

    # print("attend_logits ",attend_logits.shape, noise_std.shape, noise_mean.shape)
    noise = torch.randn_like(attend_logits, dtype=attend_logits.dtype)
    noise.mul_(noise_std)
    noise.add_(noise_mean)
    if addon_relcoef == 0:
        attend_logits.add_(noise)
    else:
        mask = (rel_coef != 0) 
        noise.mul_(mask)
        attend_logits.add_(noise)

    return attend_logits




def wrap_with_model_special_input(model_name, prompt, tokenizer, device, dataset, sys_prompt=None, apply_tmp = True):
    if "Llama-2" in model_name:
        return tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    elif "Llama-3" in model_name:
        if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            return tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            # print("prompt", len(prompt))
            if sys_prompt is None:
                sys_prompt = "You are a chatbot who always responds in polite speak."

            messages = [
            {"role": "system", "content": "{}".format(sys_prompt)},
            {"role": "user", "content": "{}".format(prompt)},]
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)


def get_chunk_size_list(cur_len, max_position_embeddings, chunk_coe):
    print("cur_len, max_position_embeddings, chunk_coe ",cur_len, max_position_embeddings, chunk_coe)
    chunk_size_list = [max_position_embeddings]
    sum_len = max_position_embeddings
    while sum_len < cur_len:
        chunk_size = int(sum_len*chunk_coe) if type(chunk_coe) == float else chunk_coe
        chunk_size_list.append(chunk_size)
        sum_len += chunk_size
    chunk_size_list[-1] = chunk_size_list[-1] - (sum_len - cur_len)   
    return chunk_size_list

def in_place_softmax(x, dim=-1):
    """
    In-place softmax implementation in PyTorch.
    This modifies the input tensor `x` directly.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which softmax will be computed.
    Returns:
        torch.Tensor: The same tensor after applying in-place softmax.
    """
    # Subtract the max value for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x.sub_(x_max)

    # Apply exponentials in-place
    x.exp_()

    # Normalize by the sum of exponentials in-place
    sum_exp = torch.sum(x, dim=dim, keepdim=True)
    x.div_(sum_exp)

    return x

def chunk_softmax(attn_weights):
    bs, head, length, length = attn_weights.shape
    interval = int(length/5)
    for i in range(0, length, interval):
        attn_weights[:,:,i:i+interval,:] = torch.nn.functional.softmax(attn_weights[:,:,i:i+interval,:], dim=-1)
    return attn_weights

def compute_tensor_size(tensor):

    # 计算内存占用
    num_elements = tensor.numel()
    bytes_per_element = tensor.element_size()
    total_memory_bytes = num_elements * bytes_per_element
    total_memory_mb = total_memory_bytes / (1024 ** 3)
    return total_memory_mb
    # print(f"张量占用内存：{total_memory_mb:.2f} GB")


def compute_attention_scores(query, key, chunk_size=None, scale_factor=None):
    """
    Computes scaled dot-product attention scores for multi-head attention.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        scale_factor: Optional scaling factor for numerical stability (default is 1/sqrt(head_dim))

    Returns:
        Tensor: Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
    """
    if scale_factor is None:
        scale_factor = query.size(-1) ** 0.5


    batch_size, num_heads, seq_len, head_dim = query.shape
    if chunk_size is None:
        chunk_size = int(seq_len/2)
    attention_scores = []
    for i in range(0, seq_len, chunk_size):
        query_chunk = query[:, :, i:i + chunk_size, :]  # Select chunk from query
        # Compute scaled dot-product attention for the chunk
        scores_chunk = torch.einsum(
            'bhqd,bhkd->bhqk', query_chunk, key
        )  # (batch_size, num_heads, chunk_size, seq_len)
        scores_chunk = scores_chunk.div_(scale_factor)  # Scale scores
        attention_scores.append(scores_chunk)

    # Concatenate along the sequence length dimension
    return torch.cat(attention_scores, dim=2)

decode_rec = []

def get_importance(complete_attn, device):
    # print("complete_attn ", complete_attn.shape)
    if len(complete_attn.shape) == 4:
        complete_attn = complete_attn.mean(dim=1) # mean along the head -> bs, q, k
    max_k_len = complete_attn.shape[-1]
    lower_triangular_matrix = torch.tril(torch.ones(max_k_len,max_k_len)).to(device)
    scaling = torch.arange(max_k_len).to(device).unsqueeze(1) + 1
    scaling = lower_triangular_matrix*scaling
    norm = scaling.sum(dim = 0)
    norm[norm==0] = 1
    
    importance = (complete_attn * lower_triangular_matrix * scaling).cumsum(dim=1) / norm
    return max_k_len, importance

def get_importance_norm(complete_attn, device):
    if len(complete_attn.shape) == 4:
        complete_attn = complete_attn.mean(dim=1) # mean along the head -> bs, q, k

    max_k_len = complete_attn.shape[-1]
    # print("importance 1 ", importance)
    # importance = importance.cumsum(dim=1)
    # print("importance 2 ", importance)

    lower_triangular_matrix = torch.tril(torch.ones(max_k_len,max_k_len)).to(device)
    norm = lower_triangular_matrix.cumsum(dim=0)
    norm[norm==0] = 1
    # print("norm ",norm)
    importance = (complete_attn * lower_triangular_matrix).cumsum(dim=1)
    # print("importance 2 ",importance)
    importance = importance / norm # bs, len, len
    return max_k_len, importance
    
def construct_new_pi(cur_len, max_position_embeddings, device, rec, add_token=1,local_window=0, start_pos = 0):
    # start_pos = 100
    target_len = cur_len + add_token
    min_group_size = math.ceil((target_len-local_window-start_pos)/ (max_position_embeddings-local_window - start_pos))
    interval = 1/min_group_size
    ori_len = max_position_embeddings - start_pos
    new_pi = []
    i = start_pos
    while ori_len < target_len:
        new_pi += [i +interval * j for j in range(min_group_size)]
        i += 1
        ori_len = max_position_embeddings - i + len(new_pi)
        print("ori_len ",ori_len,target_len,len(new_pi))
    seg_1 = [j for j in range(0, start_pos)]
    seg_2 = [j for j in range(i+start_pos, max_position_embeddings)]
    new_pi = seg_1 + new_pi[:target_len - len(seg_1) - len(seg_2)] +  seg_2
    new_pi = torch.tensor(new_pi, device=device)
    rec.append(new_pi)
    return new_pi


def decode_interpolation(position_ids, max_position_embeddings, unfinished_sequences, device,local_window):
    assert len(position_ids) == 1, "currently only support batch_size == 1 "
    new_position_ids = []
    # global decode_rec
    for i, one_old_position_ids in enumerate(position_ids):
        if unfinished_sequences[i] == 0: # finished sentences
            new_position_ids.append(torch.cat([one_old_position_ids, torch.tensor([0])]))
        else:
            # the one_old_position_ids may be [1,1,0,1,2,3,4]/[1,1,0,1] in the case of batch inference
            start_pos = torch.where(one_old_position_ids == 0)[0][0]
            cur_len = len(one_old_position_ids[start_pos:])
            # print("cur_len ",cur_len, start_pos)
            if cur_len < max_position_embeddings: # In the case of batch inference, some sentences don't exceed the training context length
                new_position_ids.append(torch.cat([one_old_position_ids, torch.tensor([cur_len])]))
            else:
                new_pi = construct_new_pi(cur_len, max_position_embeddings, device, decode_rec, 1, local_window)
                new_position_ids.append(new_pi)
    return torch.stack(new_position_ids,dim=0) # bs, current length

def prefill_interpolation(cur_input_ids, position_ids, max_position_embeddings, unfinished_sequences, device, local_window):
    assert len(position_ids) == 1, "currently only support batch_size == 1 "

    new_position_ids = []
    for i, one_old_position_ids in enumerate(position_ids):
        if unfinished_sequences[i] == 0: # finished sentences
            new_position_ids.append(torch.cat([one_old_position_ids, torch.tensor([0])]))
        else:
            # the one_old_position_ids may be [1,1,0,1,2,3,4]/[1,1,0,1] in the case of batch inference
            start_pos = torch.where(one_old_position_ids == 0)[0][0]
            last_len = len(one_old_position_ids[start_pos:])
            # print("last_len ",last_len, start_pos)
            cur_max_add_tokens = cur_input_ids.shape[1] - last_len
            if last_len < max_position_embeddings: # In the case of batch prefill, some sentences don't exceed the training context length
                # to do 
                # new_position_ids.append(torch.cat([one_old_position_ids, torch.tensor([last_len])]))
                pass
            else:
                new_pi = construct_new_pi(last_len, max_position_embeddings, device, decode_rec, cur_max_add_tokens, local_window)
                new_position_ids.append(new_pi)
                
    return torch.stack(new_position_ids,dim=0) # bs, current length