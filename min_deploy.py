def get_load_time_and_layer(load_time_list, avai_load_time, load_begin_idx):
    load_layer = 0
    load_time = 0
    load_idx = load_begin_idx
    total_layer_num = len(load_time_list)
    while load_idx < total_layer_num:
        if load_time + load_time_list[load_idx] > avai_load_time:
            break
        load_time += load_time_list[load_idx]
        load_layer += 1
        load_idx += 1

    return load_time, load_layer

def cal_lat_and_gpu_num(deploy_layers_num, total_layer_num, stall_num, single_layer_time):
    total_cal_layer_num = 0
    left_time = 0
    load_layer = deploy_layers_num
    total_inference_time = 0
    avai_load_time = 0
    use_gpu_num = 1

    load_index = [deploy_layers_num]

    cal_time_list = single_layer_time.cal_time
    trans_time_list = single_layer_time.trans_time
    load_time_list = single_layer_time.load_time

    cal_begin_idx_cur_round = 0
    load_begin_idx_cur_round = cal_begin_idx_cur_round + load_layer

    if stall_num > 0:
        while True:
            cal_time = sum(
                cal_time_list[
                    cal_begin_idx_cur_round : cal_begin_idx_cur_round + load_layer
                ]
            )
            total_cal_layer_num += load_layer
            total_inference_time += cal_time

            if total_cal_layer_num == total_layer_num:
                load_index.append(total_layer_num)
                return total_inference_time, use_gpu_num, load_index

            avai_load_time = cal_time + left_time
            if avai_load_time < load_time_list[load_begin_idx_cur_round]:
                total_cal_layer_num += stall_num
                total_inference_time += (
                    (load_time_list[load_begin_idx_cur_round] - avai_load_time)
                    + sum(
                        load_time_list[
                            load_begin_idx_cur_round : load_begin_idx_cur_round
                            + stall_num
                            - 1
                        ]
                    )
                    + cal_time_list[load_begin_idx_cur_round + stall_num - 1]
                    + trans_time_list[load_begin_idx_cur_round + stall_num - 1]
                )
                load_index.append(total_cal_layer_num)
                if total_cal_layer_num == total_layer_num:
                    return total_inference_time, use_gpu_num, load_index
                avai_load_time = total_inference_time
                use_gpu_num += 1
                break                                                                                                      

            load_time_cur_round, load_layer = get_load_time_and_layer(load_time_list, avai_load_time, load_begin_idx_cur_round)

            if total_cal_layer_num + load_layer >= total_layer_num:
                total_inference_time += sum(cal_time_list[load_begin_idx_cur_round:])
                load_index.append(total_layer_num)
                return total_inference_time, use_gpu_num, load_index

            cal_begin_idx_cur_round = total_cal_layer_num
            load_begin_idx_cur_round = cal_begin_idx_cur_round + load_layer
            left_time = avai_load_time - load_time_cur_round

        load_time_cur_round, load_layer = get_load_time_and_layer(load_time_list, avai_load_time, load_begin_idx_cur_round)
        cal_begin_idx_cur_round = total_cal_layer_num
        load_begin_idx_cur_round = cal_begin_idx_cur_round + load_layer
        left_time = 0

    # if cal_begin_idx_cur_round == 0 and load_layer == 0:
    #     return 

    while total_cal_layer_num < total_layer_num:
        if total_cal_layer_num + load_layer >= total_layer_num:
            total_inference_time += sum(cal_time_list[load_begin_idx_cur_round:])
            break
        
        cal_time = sum(
            cal_time_list[
                cal_begin_idx_cur_round : cal_begin_idx_cur_round + load_layer
            ]
        )
        total_cal_layer_num += load_layer
        total_inference_time += cal_time

        if total_cal_layer_num == total_layer_num:
            load_index.append(total_layer_num)
            return total_inference_time, use_gpu_num, load_index

        avai_load_time = cal_time + left_time
        if avai_load_time < load_time_list[load_begin_idx_cur_round]:
            total_inference_time += trans_time_list[total_cal_layer_num]
            avai_load_time = total_inference_time
            use_gpu_num += 1
            load_index.append(total_cal_layer_num)

        load_time_cur_round, load_layer = get_load_time_and_layer(load_time_list, avai_load_time, load_begin_idx_cur_round)

        if total_cal_layer_num + load_layer >= total_layer_num:
            total_inference_time += sum(cal_time_list[load_begin_idx_cur_round:])
            break

        cal_begin_idx_cur_round = total_cal_layer_num
        load_begin_idx_cur_round = cal_begin_idx_cur_round + load_layer
        left_time = avai_load_time - load_time_cur_round

        if load_layer == 0:
            total_inference_time = float('inf')
            use_gpu_num = float('inf')
            break

    load_index.append(total_layer_num)
    return total_inference_time, use_gpu_num, load_index


def cal_lat_by_deploynum_and_avaigpu(deploy_layers_num, total_layer_num, avai_gpu_num, single_layer_time):
    stall_num = 0
    while True:
        if deploy_layers_num == 0 and stall_num == 0:
            stall_num += 1
            continue

        inf_lat, use_gpu_num, load_index = cal_lat_and_gpu_num(deploy_layers_num, total_layer_num, stall_num, single_layer_time)
        if use_gpu_num <= avai_gpu_num:
            break
        stall_num += 1
    return inf_lat, load_index


def min_deploy(total_layer_num, slo, avai_gpu_num, single_layer_time):
    if sum(single_layer_time.cal_time) >= slo:
        return [total_layer_num, total_layer_num]

    l = 0
    r = total_layer_num
    while l < r:
        mid = (int)(l + (r - l) / 2)
        # print(l, r, mid)
        total_inference_time, use_gpu_num, load_index = cal_lat_and_gpu_num(mid, total_layer_num, 0, single_layer_time)
        # print(mid, total_inference_time, use_gpu_num, load_index)
        if total_inference_time > slo:
            l = mid + 1
        else:
            r = mid
            deploy_layers_num = mid

    stall_num = 1
    while use_gpu_num > avai_gpu_num:
        total_inference_time, use_gpu_num, load_index = cal_lat_and_gpu_num(deploy_layers_num, total_layer_num, stall_num, single_layer_time)
        # print(total_inference_time, use_gpu_num, load_index, stall_num)

        if use_gpu_num <= avai_gpu_num:
            break
        
        if total_inference_time > slo:
            if stall_num == 0:
                break
            stall_num = 0
            deploy_layers_num += 1
            continue

        stall_num += 1
    # print(total_inference_time, use_gpu_num, load_index)
    return load_index
